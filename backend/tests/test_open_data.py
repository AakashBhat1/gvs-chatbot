"""Generated lexicon (open data), Hindi script, accents and the off-topic gate."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

from app.brains.factory import build_cascade
from app.brains.lexicon import Lexicon, LexiconError, load_lexicon, parse_lexicon
from app.brains.rules import RulesBrain
from app.chat import ChatEngine
from app.evaluate import load_cases, run_case
from app.extraction import Context
from app.form import TripForm
from app.hindi import romanize
from app.paths import default_generated_path, default_lexicon_path
from app.text import tokenize
from app.tree import LoadedTree
from tools import build_lexicon as B

EVALS = Path(__file__).resolve().parents[1] / "evals"
HAND = default_lexicon_path().read_bytes()


@pytest.fixture(scope="module")
def lex(real_tree_module: LoadedTree) -> Lexicon:
    return load_lexicon(default_lexicon_path(), real_tree_module)


@pytest.fixture(scope="module")
def brain(lex: Lexicon) -> RulesBrain:
    return RulesBrain(lex)


def understand(brain: RulesBrain, msg: str) -> Any:
    return brain.understand(msg, Context(TripForm(), None, "collecting")).extraction


# ----------------------------------------------------------------- Hindi script


@pytest.mark.parametrize(
    ("word", "spellings"),
    [
        ("घूमने", ["ghoomne", "ghumne"]),  # medial schwa deletion
        ("जापान", ["jaapaan", "japan"]),  # final schwa deletion
        ("मुझे", ["mujhe"]),
        ("माँ", ["maa", "ma"]),  # candrabindu only nasalises
        ("हैं", ["hain"]),
        ("फ़्लाइट", ["flaait", "flait"]),  # precomposed nukta letter
        ("फ़्लाइट", ["flaait", "flait"]),  # combining nukta
        ("स्कूल", ["skool", "skul"]),  # virama
        ("।", []),  # punctuation only
    ],
)
def test_romanize(word: str, spellings: list[str]) -> None:
    assert romanize(word) == spellings


def test_hindi_script_message(brain: RulesBrain) -> None:
    e = understand(brain, "मुझे जापान घूमने जाना है, मैं और मेरी पत्नी")
    assert (e.country, e.purpose) == ("japan", "tourism")
    assert [p.relation for p in e.travellers] == ["self", "spouse"]


def test_hindi_script_conversation(lex: Lexicon, real_tree_module: LoadedTree) -> None:
    engine = ChatEngine(lambda: real_tree_module, RulesBrain(lex), ask_age_for=lex.ask_age_for)
    state, _ = engine.start()
    for line in ("मुझे जापान घूमने जाना है, मैं और मेरी पत्नी", "मैं डॉक्टर हूँ और पत्नी टीचर है"):
        state, reply = engine.handle(state, text=line)
    assert state.stage == "confirming"
    assert [t.profession for t in state.form.travellers] == [
        "self_employed_professional",
        "employed",
    ]
    state, reply = engine.handle(state, text="हाँ")
    assert reply.checklist is not None


def test_hindi_city_names_match_exactly(brain: RulesBrain) -> None:
    assert understand(brain, "पेरिस").country == "france"
    assert understand(brain, "दुबई").country == "uae-dubai"


# ----------------------------------------------------------------- text normalisation


def test_tokenize_folds_accents_and_hindi_digits() -> None:
    assert tokenize("Montréal, Zürich, São Paulo") == [
        "montreal",
        ",",
        "zurich",
        ",",
        "sao",
        "paulo",
    ]
    assert tokenize("१२ साल") == ["12", "साल"]


# ----------------------------------------------------------------- generated data in the bot


@pytest.mark.parametrize(
    ("message", "field", "value"),
    [
        ("going to phuket", "country", "thailand"),
        ("conference in zurich", "country", "switzerland"),
        ("trip to Zürich", "country", "switzerland"),
        ("we fly to calgary", "country", "canada"),
        ("I need a British visa", "country", "uk"),
        ("Nippon trip", "country", "japan"),
    ],
)
def test_generated_places(brain: RulesBrain, message: str, field: str, value: str) -> None:
    assert getattr(understand(brain, message), field) == value


def test_city_in_country_without_checklist(brain: RulesBrain) -> None:
    assert understand(brain, "holiday in colombo").other_country == "sri-lanka"


@pytest.mark.parametrize(
    ("job", "bucket"),
    [
        ("software tester", "employed"),
        ("data analyst", "employed"),
        ("videographer", "employed"),
        ("cardiologist", "self_employed_professional"),
    ],
)
def test_generated_job_titles(brain: RulesBrain, job: str, bucket: str) -> None:
    e = understand(brain, f"Japan tourism, just me, I am a {job}")
    assert [p.profession for p in e.travellers] == [bucket]


def test_plurals_resolve_but_common_words_are_not_typos(lex: Lexicon) -> None:
    assert lex.guess("teachers") == ("profession", "employed")
    assert lex.guess("dentists") == ("profession", "self_employed_professional")
    for word in ("suit", "begin", "submit", "mine", "sales", "written"):
        assert lex.guess(word) is None, word
    assert lex.guess("japn") == ("country", "japan")  # real typos still corrected


def test_generated_aliases_are_never_typo_targets(lex: Lexicon) -> None:
    assert lex.guess("calgarry") is None  # 'calgary' is generated: exact match only


# ----------------------------------------------------------------- merge rules


def _gen(**sections: Any) -> bytes:
    return json.dumps(sections).encode()


def test_hand_lexicon_wins_clashes(real_tree_module: LoadedTree) -> None:
    lex = parse_lexicon(HAND, real_tree_module, _gen(countries={"uk": ["japan", "zzland"]}))
    assert lex.lookup(("japan",)) == ("country", "japan")  # hand alias untouched
    assert lex.lookup(("zzland",)) == ("country", "uk")


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (_gen(countries={"atlantis": ["x"]}), "not an id"),
        (_gen(countries=["x"]), "expected an object"),
        (_gen(common_words="nope"), "list of strings"),
        (b"[1, 2]", "expected an object"),
        (b"{nope", "generated lexicon"),
        (b" " * 16_000_001, "limit"),
    ],
)
def test_bad_generated_lexicon_rejected(
    real_tree_module: LoadedTree, raw: bytes, message: str
) -> None:
    with pytest.raises(LexiconError, match=message):
        parse_lexicon(HAND, real_tree_module, raw)


def test_bot_works_without_generated_file(
    real_tree_module: LoadedTree, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GVS_GENERATED_LEXICON_PATH", str(tmp_path / "missing.json"))
    lex = load_lexicon(default_lexicon_path(), real_tree_module)
    assert lex.lookup(("phuket",)) == ("country", "thailand")  # hand alias still there
    assert lex.lookup(("calgary",)) is None
    with pytest.raises(LexiconError, match="cannot read"):
        load_lexicon(default_lexicon_path(), real_tree_module, generated_path=tmp_path / "x.json")


def test_generated_file_is_committed_and_valid() -> None:
    doc = json.loads(default_generated_path().read_text(encoding="utf-8"))
    assert doc["counts"]["countries"] > 1000
    assert doc["counts"]["professions"] > 10_000
    assert "geonames-cities15000" in doc["sources"]


# ----------------------------------------------------------------- off-topic gate


def test_off_topic_messages_rarely_touch_the_form(
    lex: Lexicon, real_tree_module: LoadedTree
) -> None:
    """2,000 CLINC150 messages (banking, music, recipes...): the form should stay empty."""
    cascade = build_cascade(lex)
    cases = load_cases(EVALS / "oos.jsonl")
    changed = sum(run_case(cascade, c, real_tree_module).form != TripForm() for c in cases)
    assert len(cases) == 2000
    assert changed / len(cases) <= 0.08, f"{changed} off-topic messages changed the form"


# ----------------------------------------------------------------- build tool helpers


@pytest.mark.parametrize(
    ("title", "bucket"),
    [
        ("army doctor", "defence_personnel"),
        ("government doctor", "government_employee"),
        ("merchant navy officer", "seafarer"),
        ("cardiologist", "self_employed_professional"),
        ("solutions architect", "employed"),
        ("retired teacher", "retired"),
        ("shop owner", "business_owner"),
        ("data analyst", "employed"),
    ],
)
def test_bucket_rules(title: str, bucket: str) -> None:
    assert B.bucket_for(title) == bucket


def test_alias_cleaning() -> None:
    assert B.norm("The Bahamas") == "bahamas"
    assert B.norm("Côte d'Ivoire (Ivory Coast)") == "cote d'ivoire"
    assert B.usable("new york") and B.usable("दुबई")
    assert not B.usable("jp")  # short codes are never generated
    assert not B.usable("a and p mechanic")
    assert B._similar("fukuoko", "fukuoka") and not B._similar("ba wang zi shi", "hachioji")


def test_fetch_verifies_pinned_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(B, "RAW", tmp_path)
    blob = io.BytesIO()
    with tarfile.open(fileobj=blob, mode="w:gz") as tar:
        data = b"hello"
        info = tarfile.TarInfo("package/x.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    raw = blob.getvalue()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "pkg.tgz").write_bytes(raw)
    good = {
        "id": "src",
        "url": "https://example.invalid/pkg.tgz",
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    assert B.tar_member(B.fetch(good, pin=False), "package/x.txt") == b"hello"
    with pytest.raises(SystemExit, match="refusing to build"):
        B.fetch({**good, "sha256": "0" * 64}, pin=False)
