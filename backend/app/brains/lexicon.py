"""Load and validate ``lexicon.toml`` — the vocabulary of the rules brain."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from app.paths import default_generated_path
from app.text import tokenize
from app.tree import LoadedTree

MAX_LEXICON_BYTES: Final = 262_144
MAX_PHRASE_TOKENS: Final = 5
MAX_GENERATED_BYTES: Final = 16_000_000
GENERATED_SECTIONS: Final = ("countries", "other_countries", "professions")

VALUE_SECTIONS: Final = ("countries", "other_countries", "purposes", "professions", "relations")
MARKERS: Final = frozenset(
    {
        "we",
        "couple",
        "parents",
        "two_of_us",
        "solo",
        "plural",
        "count_noun",
        "count_of",
        "age_unit",
        "month_unit",
        "age_word",
        "confirm",
        "deny",
        "correct_cue",
        "restart",
        "remove",
        "greeting",
        "business_word",
        "pronoun",
        "conj",
        "weak_number",
        "off_topic",
        "help",
        "office_question",
        "with_others",
        "no_more",
        "one_of",
        "other_of",
    }
)
# Section name -> the kind of span it produces.
_KIND: Final = {
    "countries": "country",
    "other_countries": "other_country",
    "purposes": "purpose",
    "professions": "profession",
    "relations": "relation",
}


class LexiconError(ValueError):
    pass


VALUE_KINDS: Final = frozenset(_KIND.values())
FUZZY_MIN_LEN: Final = 4  # shortest word we try to correct
FUZZY_MIN_ALIAS: Final = 5  # shortest alias a typo may resolve to


def edit_distance(a: str, b: str, limit: int) -> int:
    """Optimal-string-alignment distance (typos incl. swapped letters), capped at limit + 1."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2: list[int] = []
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1]


@dataclass(frozen=True, slots=True)
class Lexicon:
    phrases: Mapping[tuple[str, ...], tuple[str, str]]  # tokens -> (kind, value)
    numbers: Mapping[str, int]
    weak_numbers: frozenset[str]
    ask_age_for: frozenset[str]
    plural_relations: frozenset[str]
    max_phrase: int
    # "newzealand" -> new zealand; built from multi-word value aliases.
    compact: Mapping[str, tuple[str, str]] = field(default_factory=dict)
    # first letter -> single-word value aliases, for typo matching.
    fuzzy_pool: Mapping[str, tuple[tuple[str, tuple[str, str]], ...]] = field(default_factory=dict)
    # Hindi-script words that appear in aliases; kept as-is instead of romanized.
    script_words: frozenset[str] = frozenset()
    # Real English words: never "corrected" ("suit" is not a typo of the city Suita).
    common_words: frozenset[str] = frozenset()

    def lookup(self, tokens: tuple[str, ...]) -> tuple[str, str] | None:
        return self.phrases.get(tokens)

    def guess(self, token: str) -> tuple[str, str] | None:
        """Best value for an unknown word: joined phrase, or one typo away (two if long).

        Requires the same first letter and a unique best match, so short or
        ambiguous words are never guessed.
        """
        if hit := self.compact.get(token):
            return hit
        for suffix in ("s", "es"):  # "teachers", "dentists": plurals are grammar, not typos
            stem = (
                self.phrases.get((token.removesuffix(suffix),)) if token.endswith(suffix) else None
            )
            if stem is not None and stem[0] in VALUE_KINDS:
                return stem
        if len(token) < FUZZY_MIN_LEN or not token.isalpha() or token in self.common_words:
            return None
        limit = 1 if len(token) <= 7 else 2
        best: dict[tuple[str, str], int] = {}
        for alias, hit in self.fuzzy_pool.get(token[0], ()):
            d = edit_distance(token, alias, limit)
            if d <= limit and d < best.get(hit, limit + 1):
                best[hit] = d
        if not best:
            return None
        top = min(best.values())
        winners = [hit for hit, d in best.items() if d == top]
        return winners[0] if len(winners) == 1 else None


def _str_list(value: object, where: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise LexiconError(f"{where}: expected a list of non-empty strings")
    return [v.strip().lower() for v in value]


def _typo_index(
    phrases: Mapping[tuple[str, ...], tuple[str, str]],
) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[tuple[str, tuple[str, str]], ...]]]:
    """Joined multi-word aliases ("newzealand") and per-letter pools for typo matching."""
    compact: dict[str, tuple[str, str]] = {}
    clashes: set[str] = set()
    pool: dict[str, list[tuple[str, tuple[str, str]]]] = {}
    for tokens, hit in phrases.items():
        if hit[0] not in VALUE_KINDS and hit[0] != "business_word":
            continue
        if len(tokens) > 1:
            joined = "".join(tokens)
            if (joined,) in phrases or (joined in compact and compact[joined] != hit):
                clashes.add(joined)
            compact[joined] = hit
        elif tokens[0].isalpha() and len(tokens[0]) >= (
            FUZZY_MIN_ALIAS + 1 if hit[0] == "other_country" else FUZZY_MIN_ALIAS
        ):
            # Short aliases are never typo targets: "male" must not become Mali.
            pool.setdefault(tokens[0][0], []).append((tokens[0], hit))
    for joined in clashes:
        compact.pop(joined, None)
    return compact, {k: tuple(v) for k, v in pool.items()}


def parse_lexicon(raw: bytes, tree: LoadedTree, generated: bytes | None = None) -> Lexicon:
    if len(raw) > MAX_LEXICON_BYTES:
        raise LexiconError(f"lexicon is {len(raw)} bytes; limit is {MAX_LEXICON_BYTES}")
    try:
        doc: dict[str, Any] = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise LexiconError(str(exc)) from exc

    known = {"settings", "fillers", "numbers", "markers", *VALUE_SECTIONS}
    if unknown := set(doc) - known:
        raise LexiconError(f"unknown sections: {sorted(unknown)}")

    data = tree.data
    tables: dict[str, Mapping[str, object]] = {
        "countries": data.countries,
        "purposes": data.purposes,
        "professions": data.professions,
        "relations": data.relations,
        "other_countries": data.other_countries,
    }
    errors: list[str] = []
    phrases: dict[tuple[str, ...], tuple[str, str]] = {}
    origin: dict[tuple[str, ...], str] = {}

    def add(alias: str, kind: str, value: str, where: str) -> None:
        tokens = tuple(tokenize(alias))
        if not tokens or len(tokens) > MAX_PHRASE_TOKENS:
            errors.append(f"{where}: alias {alias!r} must be 1-{MAX_PHRASE_TOKENS} words")
            return
        if tokens in phrases:
            errors.append(f"{where}: alias {alias!r} already used in {origin[tokens]}")
            return
        phrases[tokens] = (kind, value)
        origin[tokens] = where

    for section in VALUE_SECTIONS:
        for key, aliases in doc.get(section, {}).items():
            if key not in tables[section]:
                errors.append(f"{section}.{key}: not an id in visa-tree.json")
                continue
            for alias in _str_list(aliases, f"{section}.{key}"):
                add(alias, _KIND[section], key, f"{section}.{key}")

    markers = doc.get("markers", {})
    if unknown_markers := set(markers) - MARKERS:
        errors.append(f"markers: unknown roles {sorted(unknown_markers)}")
    weak = frozenset(_str_list(markers.get("weak_number", []), "markers.weak_number"))
    for role, words in markers.items():
        if role in MARKERS and role != "weak_number":
            for alias in _str_list(words, f"markers.{role}"):
                add(alias, role, role, f"markers.{role}")
    for alias in _str_list(doc.get("fillers", {}).get("words", []), "fillers.words"):
        add(alias, "filler", "filler", "fillers")

    numbers: dict[str, int] = {}
    for word, value in doc.get("numbers", {}).items():
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 110:
            errors.append(f"numbers.{word}: must be a whole number 0-110")
        elif (word,) in phrases:
            errors.append(f"numbers.{word}: also used as an alias in {origin[(word,)]}")
        else:
            numbers[word] = value
    errors += [f"markers.weak_number: {w!r} is not in [numbers]" for w in weak if w not in numbers]

    settings = doc.get("settings", {})
    if unknown_settings := set(settings) - {"ask_age_for", "plural_relations"}:
        errors.append(f"settings: unknown keys {sorted(unknown_settings)}")
    ask_age_for = frozenset(_str_list(settings.get("ask_age_for", []), "settings.ask_age_for"))
    errors += [
        f"settings.ask_age_for: unknown relation {r!r}"
        for r in sorted(ask_age_for - set(data.relations))
    ]
    plural_relations = frozenset(
        _str_list(settings.get("plural_relations", []), "settings.plural_relations")
    )
    errors += [
        f"settings.plural_relations: {w!r} is not a relation alias"
        for w in sorted(plural_relations)
        if phrases.get(tuple(tokenize(w)), ("", ""))[0] != "relation"
    ]

    if errors:
        raise LexiconError("; ".join(errors))

    no_typos: set[tuple[str, ...]] = set()
    common: frozenset[str] = frozenset()
    if generated is not None:
        no_typos, common = _merge_generated(generated, phrases, tables)
    compact, pool = _typo_index({k: v for k, v in phrases.items() if k not in no_typos})

    return Lexicon(
        phrases=phrases,
        numbers=numbers,
        weak_numbers=weak,
        ask_age_for=ask_age_for,
        plural_relations=plural_relations,
        max_phrase=max(len(t) for t in phrases),
        compact=compact,
        fuzzy_pool=pool,
        script_words=frozenset(t for tokens in phrases for t in tokens if _devanagari(t)),
        common_words=common,
    )


def _devanagari(token: str) -> bool:
    return any("\u0900" <= ch <= "\u097f" for ch in token)


def _merge_generated(
    raw: bytes,
    phrases: dict[tuple[str, ...], tuple[str, str]],
    tables: Mapping[str, Mapping[str, object]],
) -> tuple[set[tuple[str, ...]], frozenset[str]]:
    """Add generated aliases. The hand lexicon wins every clash; ids must still exist.

    Returns (generated phrases, common English words). Generated aliases match
    exactly or not at all: typo correction only targets the curated lexicon.
    """
    if len(raw) > MAX_GENERATED_BYTES:
        raise LexiconError(f"generated lexicon is {len(raw)} bytes; limit is {MAX_GENERATED_BYTES}")
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LexiconError(f"generated lexicon: {exc}") from exc
    if not isinstance(doc, dict):
        raise LexiconError("generated lexicon: expected an object")
    errors: list[str] = []
    no_typos: set[tuple[str, ...]] = set()
    for section in GENERATED_SECTIONS:
        entries = doc.get(section, {})
        if not isinstance(entries, dict):
            errors.append(f"generated.{section}: expected an object")
            continue
        for key, aliases in entries.items():
            if key not in tables[section]:
                errors.append(f"generated.{section}.{key}: not an id in visa-tree.json")
                continue
            for alias in _str_list(aliases, f"generated.{section}.{key}"):
                tokens = tuple(tokenize(alias))
                if not tokens or len(tokens) > MAX_PHRASE_TOKENS or tokens in phrases:
                    continue  # the hand-written lexicon already decides this one
                phrases[tokens] = (_KIND[section], key)
                no_typos.add(tokens)
    words = doc.get("common_words", [])
    if not isinstance(words, list) or not all(isinstance(w, str) for w in words):
        errors.append("generated.common_words: expected a list of strings")
        words = []
    if errors:
        raise LexiconError("; ".join(errors[:20]))
    return no_typos, frozenset(words)


_CACHE: dict[tuple[object, ...], Lexicon] = {}
_CACHE_SIZE: Final = 4


def _signature(path: Path | None) -> tuple[object, ...]:
    if path is None:
        return (None,)
    st = path.stat()
    return (str(path.resolve()), st.st_mtime_ns, st.st_size)


def load_lexicon(
    path: Path, tree: LoadedTree, *, generated_path: Path | Literal["default"] | None = "default"
) -> Lexicon:
    """Load lexicon.toml, plus the generated lexicon (default location) when it exists.

    Results are cached per (files, tree version): building ~70k phrases takes about
    a second, and the result is immutable. Editing either file invalidates the cache.
    """
    if generated_path == "default":
        default = default_generated_path()
        generated_path = default if default.exists() else None
    try:
        key = (_signature(path), _signature(generated_path), tree.version)
        if (cached := _CACHE.get(key)) is not None:
            return cached
        raw = path.read_bytes()
        generated = generated_path.read_bytes() if generated_path is not None else None
    except OSError as exc:
        raise LexiconError(f"cannot read lexicon: {exc}") from exc
    lexicon = parse_lexicon(raw, tree, generated)
    if len(_CACHE) >= _CACHE_SIZE:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = lexicon
    return lexicon
