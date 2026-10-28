"""Generate lexicon entries from open data: ``uv run --group build python -m tools.build_lexicon``.

Reads ``data/sources.toml``, downloads each source into ``data/raw/`` (git-ignored),
verifies its pinned sha256, and writes:

* ``data/generated/lexicon.json`` - country names (English, Hinglish, Hindi script),
  demonyms, city names -> country, job titles -> profession bucket
* ``evals/oos.jsonl`` - off-topic messages from CLINC150 for the off-topic test

Hand-written ``lexicon.toml`` always wins: anything already there is skipped. An
alias that could mean two different things is dropped rather than guessed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import ssl
import sys
import tarfile
import tomllib
import urllib.request
from collections import defaultdict
from collections.abc import Iterable, Iterator
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Final

from app.brains.lexicon import load_lexicon
from app.paths import default_lexicon_path, default_tree_path
from app.text import tokenize
from app.tree import LoadedTree, load_tree

BACKEND: Final = Path(__file__).resolve().parents[1]
RAW: Final = BACKEND / "data" / "raw"
OUT: Final = BACKEND / "data" / "generated" / "lexicon.json"
OOS_OUT: Final = BACKEND / "evals" / "oos.jsonl"
MAX_DOWNLOAD: Final = 50_000_000

CITY_MIN_POP: Final = 100_000  # main city names
ALT_MIN_POP: Final = 500_000  # alternate spellings only for big cities
ALT_SIMILARITY: Final = 0.75  # alternate city names must look like the main name
CITY_DOMINANCE: Final = 5  # same name in two countries: keep only a 5x bigger one
COMMON_WORD_ZIPF: Final = 4.0  # "Nice", "Reading", "Split" are words first
BIG_CITY_POP: Final = 1_000_000  # ...unless the city is huge
EXCLUDED_CC: Final = frozenset({"IN"})  # travellers start in India; Indian cities aren't trips
OOS_IN_SCOPE_SAMPLE: Final = 1000
COMMON_WORDS: Final = 40_000  # real English words are never treated as typos
# CLINC intents that are about travel/visas, so not off-topic for this bot.
CLINC_ON_TOPIC: Final = frozenset(
    {
        "international_visa", "travel_suggestion", "travel_alert", "travel_notification",
        "book_flight", "book_hotel", "plug_type", "exchange_rate", "flight_status",
        "carry_on", "lost_luggage", "vaccines", "timezone", "translate",
    }
)  # fmt: skip

# ------------------------------------------------------------------ job titles

JOB_HEADS: Final = frozenset(
    """engineer manager developer analyst officer assistant technician teacher nurse doctor
    driver clerk accountant consultant designer specialist executive director supervisor
    operator worker administrator architect scientist lawyer attorney pharmacist therapist
    coordinator representative agent associate inspector mechanic electrician plumber chef cook
    writer editor producer artist photographer pilot attendant programmer tester auditor banker
    broker trader owner partner founder professor lecturer tutor instructor researcher intern
    trainee surgeon physician dentist sailor seaman soldier farmer labourer laborer guard cashier
    salesman salesperson receptionist secretary carpenter painter tailor barber waiter steward
    stewardess captain advisor adviser planner strategist counsellor counselor manager head
    lead chief president vice principal dean librarian journalist reporter anchor actor dancer
    musician singer chemist biologist physicist economist statistician mathematician surveyor
    contractor foreman helper handler packer loader picker sorter welder fitter machinist
    technologist practitioner paramedic midwife caregiver housekeeper janitor cleaner gardener
    beautician stylist trainer coach referee officer constable inspector investigator detective
    firefighter ranger""".split()
)
JOB_SUFFIXES: Final = ("er", "or", "ist", "ian", "ant", "ent", "man", "woman", "person", "ee")
COMMON_SINGLE_JOBS: Final = frozenset(
    "driver cook chef nurse pilot guard clerk tailor barber electrician plumber mechanic "
    "carpenter painter waiter cashier cleaner security peon maid gardener".split()
)
NOT_IN_TITLES: Final = frozenset({"and", "or", "with", "to", "for", "&", "my", "me", "i"})

# Ordered: the first matching rule decides the bucket.
BUCKET_RULES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("retired", ("retired",)),
    ("seafarer", ("seaman", "seafarer", "sailor", "deckhand", "mariner", "boatswain", "bosun",
                  "merchant navy", "deck officer", "marine engineer", "ship")),
    ("defence_personnel", ("army", "navy", "naval", "military", "soldier", "air force", "infantry",
                           "artillery", "sepoy", "paratrooper", "commando")),
    ("government_employee", ("government", "govt", "municipal", "federal", "ministry", "civil service",
                             "public sector", "railway", "postal", "postmaster", "police", "customs")),
    ("self_employed_professional", ("physician", "doctor", "surgeon", "dentist", "lawyer", "attorney",
                                    "advocate", "barrister", "solicitor", "chartered accountant",
                                    "veterinarian", "orthodontist", "psychiatrist",
                                    "cardiologist", "neurologist", "radiologist", "dermatologist",
                                    "oncologist", "pediatrician", "paediatrician", "gynecologist",
                                    "gynaecologist", "anesthesiologist", "anaesthetist", "allergist",
                                    "urologist", "nephrologist", "ophthalmologist", "pathologist",
                                    "orthopedic surgeon", "ent specialist")),
    ("company_director", ("director", "ceo", "cfo", "coo", "cto", "founder", "cofounder", "co founder",
                          "chairman", "chairperson", "managing director", "board member", "partner")),
    ("business_owner", ("owner", "proprietor", "shopkeeper", "merchant", "trader", "entrepreneur",
                        "businessman", "businesswoman", "dealer", "retailer", "wholesaler")),
    ("freelancer", ("freelance", "freelancer", "consultant", "contractor", "self employed")),
    ("farmer", ("farmer", "farm", "agricultural", "rancher", "grower", "cultivator", "orchard")),
    ("landlord_investor", ("landlord", "investor")),
    ("university_student", ("student",)),
    ("homemaker", ("homemaker", "housewife", "househusband")),
)  # fmt: skip


def bucket_for(title: str) -> str:
    words = f" {title} "
    for bucket, keys in BUCKET_RULES:
        if any(f" {k} " in words for k in keys):
            return bucket
    return "employed"


# ------------------------------------------------------------------ fetching


def _ssl_context() -> ssl.SSLContext:
    bundle = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    return ssl.create_default_context(cafile=bundle) if bundle else ssl.create_default_context()


def fetch(source: dict[str, Any], *, pin: bool) -> bytes:
    """Return the source's bytes from data/raw, downloading if needed; verify sha256."""
    url: str = source["url"]
    path: Path = RAW / str(source["id"]) / url.rsplit("/", 1)[-1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {url}", file=sys.stderr)
        req = urllib.request.Request(url, headers={"User-Agent": "gvs-chatbot-build"})  # noqa: S310
        with urllib.request.urlopen(req, timeout=120, context=_ssl_context()) as resp:  # noqa: S310
            body: bytes = resp.read(MAX_DOWNLOAD + 1)
        if len(body) > MAX_DOWNLOAD:
            raise SystemExit(f"{url}: larger than {MAX_DOWNLOAD} bytes")
        path.write_bytes(body)
    data: bytes = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    expected = source.get("sha256")
    if expected and digest != expected and not pin:
        raise SystemExit(f"{source['id']}: sha256 {digest} != pinned {expected}; refusing to build")
    if pin:
        print(f'  {source["id"]}: sha256 = "{digest}"', file=sys.stderr)
    return data


def tar_member(blob: bytes, name: str) -> bytes:
    """Read one named file from a .tgz without extracting anything to disk."""
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        member = tar.getmember(name)
        handle = tar.extractfile(member)
        if handle is None:
            raise SystemExit(f"{name}: not a regular file")
        data: bytes = handle.read()
        return data


# ------------------------------------------------------------------ cleaning


def norm(alias: str) -> str:
    """Canonical alias text: our own tokenizer (which folds accents), joined by spaces."""
    alias = re.sub(r"\(.*?\)", " ", alias).replace("&", " and ")
    alias = re.sub(r"^the\s+", "", alias.strip(), flags=re.IGNORECASE)
    return " ".join(tokenize(alias))


def is_devanagari(text: str) -> bool:
    return any("ऀ" <= ch <= "ॿ" for ch in text)


def usable(alias: str, *, max_tokens: int = 4) -> bool:
    tokens = alias.split()
    if not tokens or len(tokens) > max_tokens or any(t in NOT_IN_TITLES for t in tokens):
        return False
    if is_devanagari(alias):
        return all(is_devanagari(t) for t in tokens)
    if not all(re.fullmatch(r"[a-z][a-z'\-]*", t) for t in tokens):
        return False
    return len(alias) >= 4  # "uk", "jp", "in", "us" are never safe as generated aliases


def zipf(alias: str) -> float:
    from wordfreq import zipf_frequency  # build-only dependency (uv --group build)

    return 0.0 if is_devanagari(alias) else float(zipf_frequency(alias, "en"))


# ------------------------------------------------------------------ sources -> aliases


def country_names(sources: dict[str, dict[str, Any]], pin: bool) -> Iterator[tuple[str, str, str]]:
    """(iso2, alias, origin) from CLDR, i18n-iso-countries and world-countries."""
    cldr = sources["cldr-localenames"]
    blob = fetch(cldr, pin=pin)
    for loc in cldr["locales"]:
        raw = tar_member(blob, cldr["member"].format(locale=loc))
        terr = json.loads(raw)["main"][loc]["localeDisplayNames"]["territories"]
        for key, name in terr.items():
            code = key.split("-")[0]
            if re.fullmatch(r"[A-Z]{2}", code):
                yield code, name, f"cldr:{loc}"

    i18n = sources["i18n-iso-countries"]
    blob = fetch(i18n, pin=pin)
    for loc in i18n["locales"]:
        data = json.loads(tar_member(blob, i18n["member"].format(locale=loc)))["countries"]
        for code, names in data.items():
            for name in [names] if isinstance(names, str) else names:
                yield code, name, f"i18n:{loc}"

    wc = sources["world-countries"]
    for c in json.loads(tar_member(fetch(wc, pin=pin), wc["member"])):
        code = c["cca2"]
        yield code, c["name"]["common"], "world-countries"
        yield code, c["name"]["official"], "world-countries"
        for alt in c.get("altSpellings", []):
            yield code, alt, "world-countries"
        for demonym in (c.get("demonyms", {}).get("eng") or {}).values():
            yield code, demonym, "world-countries:demonym"


def city_rows(
    sources: dict[str, dict[str, Any]], pin: bool
) -> Iterator[tuple[str, str, int, bool]]:
    """(iso2, alias, population, is_alternate) from GeoNames cities15000."""
    src = sources["geonames-cities15000"]
    text = tar_member(fetch(src, pin=pin), src["member"]).decode("utf-8")
    for line in text.splitlines():
        cols = line.split("\t")
        if len(cols) < 15:
            continue
        name, ascii_name, alternates, feature, cc = cols[1], cols[2], cols[3], cols[7], cols[8]
        pop = int(cols[14] or 0)
        capital = feature == "PPLC"
        if pop >= CITY_MIN_POP or capital:
            yield cc, name, pop, False
            yield cc, ascii_name, pop, False
        if pop >= ALT_MIN_POP:
            base = norm(ascii_name)
            for alt in alternates.split(","):
                # Keep spelling variants ("Capporo", "Lizboa") and Hindi script; drop
                # romanisations from other scripts ("ba wang zi shi"), which are noise.
                if is_devanagari(alt) or _similar(norm(alt), base):
                    yield cc, alt, pop, True


def _similar(a: str, b: str) -> bool:
    return bool(a and b and a[0] == b[0]) and SequenceMatcher(None, a, b).ratio() >= ALT_SIMILARITY


def job_titles(sources: dict[str, dict[str, Any]], pin: bool) -> Iterator[tuple[str, str]]:
    src = sources["job-titles"]
    for title in json.loads(fetch(src, pin=pin))["job-titles"]:
        alias = norm(title)
        tokens = alias.split()
        if not usable(alias) or not tokens:
            continue
        head = tokens[-1]
        if head not in JOB_HEADS and not head.endswith(JOB_SUFFIXES):
            continue
        if len(tokens) == 1 and head not in COMMON_SINGLE_JOBS and zipf(alias) >= 3.5:
            continue  # "model", "host", "guide": words before job titles
        yield alias, bucket_for(alias)


# ------------------------------------------------------------------ build


class Builder:
    def __init__(self, tree: LoadedTree, taken: set[str]) -> None:
        self.tree = tree
        self.taken = taken  # aliases already in the hand lexicon
        self.by_iso: dict[str, tuple[str, str]] = {}
        for section, table in (
            ("countries", tree.data.countries),
            ("other_countries", tree.data.other_countries),
        ):
            for cid, c in table.items():
                if c.iso2:
                    self.by_iso[c.iso2] = (section, cid)
        self.place: dict[str, dict[tuple[str, str], int]] = defaultdict(dict)
        self.jobs: dict[str, set[str]] = defaultdict(set)
        self.stats: dict[str, int] = defaultdict(int)

    def add_place(self, iso2: str, raw: str, weight: int) -> None:
        target = self.by_iso.get(iso2)
        alias = norm(raw)
        if target is None or iso2 in EXCLUDED_CC or not usable(alias, max_tokens=5):
            self.stats["place_skipped"] += 1
            return
        if alias in self.taken:
            self.stats["place_in_hand_lexicon"] += 1
            return
        self.place[alias][target] = max(self.place[alias].get(target, 0), weight)

    def resolve_places(self) -> dict[str, dict[str, list[str]]]:
        out: dict[str, dict[str, list[str]]] = {"countries": {}, "other_countries": {}}
        for alias, targets in self.place.items():
            ranked = sorted(targets.items(), key=lambda kv: kv[1], reverse=True)
            if len(ranked) > 1 and ranked[0][1] < CITY_DOMINANCE * ranked[1][1]:
                self.stats["place_ambiguous"] += 1
                continue
            (section, cid), _ = ranked[0]
            out[section].setdefault(cid, []).append(alias)
        return out


def build(pin: bool) -> dict[str, Any]:
    manifest = tomllib.loads((BACKEND / "data" / "sources.toml").read_text(encoding="utf-8"))
    sources = {s["id"]: s for s in manifest["source"]}
    tree = load_tree(default_tree_path())
    hand = load_lexicon(default_lexicon_path(), tree, generated_path=None)
    taken = {" ".join(tokens) for tokens in hand.phrases}
    b = Builder(tree, taken)

    huge = 10**9  # country names always beat city names
    for iso2, name, _origin in country_names(sources, pin):
        b.add_place(iso2, name, huge)
    for iso2, name, pop, alternate in city_rows(sources, pin):
        alias = norm(name)
        limit = COMMON_WORD_ZIPF if pop < BIG_CITY_POP else COMMON_WORD_ZIPF + 1
        if zipf(alias) >= (COMMON_WORD_ZIPF - 1.0 if alternate else limit):
            b.stats["city_common_word"] += 1
            continue
        b.add_place(iso2, name, pop)
    places = b.resolve_places()

    professions: dict[str, list[str]] = defaultdict(list)
    for alias, bucket in job_titles(sources, pin):
        if alias in taken or alias in b.place:
            b.stats["job_in_use"] += 1
            continue
        b.jobs[alias].add(bucket)
    for alias, buckets in sorted(b.jobs.items()):
        if len(buckets) == 1:
            professions[next(iter(buckets))].append(alias)

    write_oos(sources, pin)
    sections: dict[str, Any] = {
        "countries": {k: sorted(v) for k, v in sorted(places["countries"].items())},
        "other_countries": {k: sorted(v) for k, v in sorted(places["other_countries"].items())},
        "professions": {k: sorted(v) for k, v in sorted(professions.items())},
    }
    counts = {s: sum(len(v) for v in sections[s].values()) for s in sections}
    from wordfreq import top_n_list  # build-only dependency (uv --group build)

    common = sorted({w for w in top_n_list("en", COMMON_WORDS) if w.isalpha() and len(w) >= 3})
    counts["common_words"] = len(common)
    return {
        "_comment": "GENERATED by tools/build_lexicon.py from data/sources.toml. Do not edit.",
        "sources": sorted(s for s in sources if s not in ("clinc150", "wordfreq")),
        "counts": counts,
        "filtered": dict(sorted(b.stats.items())),
        **sections,
        "common_words": common,
    }


def write_oos(sources: dict[str, dict[str, Any]], pin: bool) -> None:
    data = json.loads(fetch(sources["clinc150"], pin=pin))
    rng = random.Random(150)  # deterministic sample  # noqa: S311
    in_scope = [t for t, intent in data["test"] if intent not in CLINC_ON_TOPIC]
    rng.shuffle(in_scope)
    texts = [t for t, _ in data["oos_test"]] + in_scope[:OOS_IN_SCOPE_SAMPLE]
    lines = [
        "// GENERATED from CLINC150 (CC BY 3.0) by tools/build_lexicon.py. Off-topic for this bot:"
    ]
    lines.append("// the form must stay empty. Travel/visa intents are excluded.")
    lines += [
        json.dumps({"id": f"oos-{i:04d}", "message": t, "expect": {"form": {}, "on_topic": False}})
        for i, t in enumerate(texts)
    ]
    OOS_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    p.add_argument("--pin", action="store_true", help="print sha256 of each source, skip checks")
    args = p.parse_args(list(argv) if argv is not None else None)
    result = build(pin=args.pin)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(BACKEND)}: {result['counts']}  filtered: {result['filtered']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
