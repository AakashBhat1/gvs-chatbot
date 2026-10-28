from __future__ import annotations

from pathlib import Path

import pytest

from app import evaluate, repl
from app.brains import BrainResult, BrainUnavailable, Cascade
from app.brains.factory import UnknownBrain, brain_names, build_cascade
from app.brains.lexicon import LexiconError, load_lexicon, parse_lexicon
from app.extraction import Context, Extraction, sanitize
from app.form import TripForm
from app.paths import default_lexicon_path
from app.text import MessageTooLong, clean, has_devanagari, tokenize
from app.tree import LoadedTree

CTX = Context(TripForm(), None, "collecting")


class Fake:
    def __init__(
        self, name: str, *, confident: bool = True, fail: bool = False, country: str = "japan"
    ) -> None:
        self.name, self.confident, self.fail, self.country, self.calls = (
            name,
            confident,
            fail,
            country,
            0,
        )

    def understand(self, message: str, ctx: Context) -> BrainResult:
        self.calls += 1
        if self.fail:
            raise BrainUnavailable(self.name)
        return BrainResult(Extraction(country=self.country), self.confident, self.name)


# ----------------------------------------------------------------- cascade


def test_cascade_stops_at_first_confident() -> None:
    a, b = Fake("a"), Fake("b")
    assert Cascade([a, b]).understand("x", CTX).brain == "a"
    assert b.calls == 0


def test_cascade_escalates_and_prefers_later_brain() -> None:
    a, b = Fake("a", confident=False, country="uk"), Fake("b", confident=False, country="japan")
    result = Cascade([a, b]).understand("x", CTX)
    assert (result.brain, result.extraction.country) == ("b", "japan")


def test_cascade_falls_back_when_smart_brain_down() -> None:
    rules, haiku = Fake("rules", confident=False), Fake("haiku", fail=True)
    assert Cascade([rules, haiku]).understand("x", CTX).brain == "rules"


def test_cascade_all_down_raises() -> None:
    with pytest.raises(BrainUnavailable):
        Cascade([Fake("a", fail=True)]).understand("x", CTX)
    with pytest.raises(ValueError, match="at least one"):
        Cascade([])


def test_brain_names() -> None:
    assert brain_names("rules") == ["rules"]
    assert brain_names("") == ["rules"]
    with pytest.raises(UnknownBrain, match="available"):
        brain_names("rules,haiku")
    with pytest.raises(UnknownBrain, match="twice"):
        brain_names("rules,rules")


def test_brain_names_from_env(monkeypatch: pytest.MonkeyPatch, real_tree: LoadedTree) -> None:
    monkeypatch.setenv("GVS_BRAINS", " RULES ")
    assert brain_names() == ["rules"]
    lex = load_lexicon(default_lexicon_path(), real_tree)
    assert [b.name for b in build_cascade(lex).brains] == ["rules"]


def test_sanitize_drops_unknown_values(real_tree: LoadedTree) -> None:
    raw = Extraction.model_validate(
        {
            "country": "mars",
            "purpose": "tourism",
            "travellers": [
                {"relation": "uncle", "profession": "retired"},
                {"relation": "mother", "profession": "astronaut", "country": "mars"},
                {"ref": 3, "age": 4},
                {"ref": 0, "age": 4},
            ],
        }
    )
    clean_ex = sanitize(raw, real_tree, n_travellers=1)
    assert clean_ex.country is None and clean_ex.purpose == "tourism"
    assert [(p.ref, p.relation, p.profession, p.country) for p in clean_ex.travellers] == [
        (None, "mother", None, None),
        (0, None, None, None),
    ]


# ----------------------------------------------------------------- lexicon validation


@pytest.mark.parametrize(
    ("toml", "message"),
    [
        ('[countries]\nmars = ["mars"]', "not an id"),
        ('[countries]\njapan = ["japan"]\nuk = ["japan"]', "already used"),
        ("[weird]\nx = 1", "unknown sections"),
        ('[markers]\nmagic = ["x"]', "unknown roles"),
        ("[numbers]\nbig = 500", "0-110"),
        ("[numbers]\ntrue_ = true", "0-110"),
        ('[markers]\nweak_number = ["zzz"]', r"not in \[numbers\]"),
        ('[settings]\nask_age_for = ["uncle"]', "unknown relation"),
        (
            '[settings]\nplural_relations = ["japan"]\n[countries]\njapan = ["japan"]',
            "not a relation alias",
        ),
        ("[settings]\nfoo = 1", "unknown keys"),
        ('[countries]\njapan = "japan"', "list of non-empty strings"),
        ('[countries]\njapan = ["a b c d e f"]', "1-5 words"),
        ('[countries]\njapan = ["japan"]\n[numbers]\njapan = 3', "also used as an alias"),
        ("not = = toml", "line 1"),
    ],
)
def test_lexicon_rejects_bad_files(real_tree: LoadedTree, toml: str, message: str) -> None:
    with pytest.raises(LexiconError, match=message):
        parse_lexicon(toml.encode(), real_tree)


def test_lexicon_limits(real_tree: LoadedTree, tmp_path: Path) -> None:
    with pytest.raises(LexiconError, match="limit"):
        parse_lexicon(b"#" * 300_000, real_tree)
    with pytest.raises(LexiconError, match="cannot read"):
        load_lexicon(tmp_path / "nope.toml", real_tree)


def test_real_lexicon_covers_every_id(real_tree: LoadedTree) -> None:
    lex = load_lexicon(default_lexicon_path(), real_tree)
    covered = {(kind, value) for kind, value in lex.phrases.values()}
    data = real_tree.data
    for kind, table in (
        ("country", data.countries),
        ("purpose", data.purposes),
        ("profession", data.professions),
        ("relation", data.relations),
    ):
        missing = [i for i in table if (kind, i) not in covered]
        assert not missing, f"no alias for {kind}s {missing}"


# ----------------------------------------------------------------- text


def test_clean_normalises_and_limits() -> None:
    assert clean("  Cafe\u0301\u200b  trip \n") == "Café trip"
    assert clean("\u202eevil") == "evil"
    with pytest.raises(MessageTooLong):
        clean("a" * 501)
    assert clean("e\u0301" * 300) == "é" * 300  # limit applies after NFC


def test_tokenize_rewrites() -> None:
    assert tokenize("U.S.A. & U.K. trip") == ["usa", "&", "uk", "trip"]
    assert tokenize("to the US, b.tech, Dr. Rao, I\u2019m") == [
        "to",
        "usa",
        ",",
        "btech",
        ",",
        "dr",
        "rao",
        ",",
        "i'm",
    ]
    assert tokenize("the usa") == ["the", "usa"]
    assert has_devanagari("जापान") and not has_devanagari("japan")


# ----------------------------------------------------------------- eval + repl CLIs


def test_gvs_eval_cli(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert evaluate.main(["--show", "1"]) == 0
    out = capsys.readouterr().out
    assert "exact" in out and "brains: rules" in out
    assert evaluate.main(["--min-exact", "1.01", "--show", "0"]) == 1
    assert (
        evaluate.main(
            [
                "--set",
                str(evaluate.DEFAULT_SET.parent / "holdout.jsonl"),
                "--runs",
                "2",
                "--show",
                "2",
            ]
        )
        == 0
    )
    assert "runs)" in capsys.readouterr().out

    dup = tmp_path / "dup.jsonl"
    line = '{"id": "a", "message": "x", "expect": {"form": {}}}\n'
    dup.write_text(line * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        evaluate.load_cases(dup)


def test_eval_counts_unavailable_brain(real_tree: LoadedTree) -> None:
    case = evaluate.Case.model_validate({"id": "a", "message": "x", "expect": {"form": {}}})
    outcome = evaluate.run_case(Fake("down", fail=True), case, real_tree)
    assert outcome.intent == "unavailable" and not outcome.matches("intent")


def test_repl_session(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    lines = iter(["Japan tourism just me salaried", "/state", "1", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(lines))
    assert repl.main([]) == 0
    out = capsys.readouterr().out
    assert "brains: rules" in out
    assert '"stage": "confirming"' in out
    assert "Here's your document checklist" in out


def test_repl_eof(monkeypatch: pytest.MonkeyPatch) -> None:
    def eof(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert repl.main(["--brains", "rules"]) == 0
