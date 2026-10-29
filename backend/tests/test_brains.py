from __future__ import annotations

from pathlib import Path

import pytest

from app import evaluate, repl
from app.brains import BrainResult, BrainUnavailable, Cascade
from app.brains.combine import fill_blanks, rules_precede
from app.brains.factory import UnknownBrain, brain_names, build_cascade, combiner
from app.brains.lexicon import LexiconError, load_lexicon, parse_lexicon
from app.extraction import Context, Extraction, sanitize
from app.form import Traveller, TripForm
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
    result = Cascade([a, b], combine=fill_blanks).understand("x", CTX)
    assert (result.brain, result.extraction.country) == ("b", "japan")
    # by default the first brain's precise reading (the place) wins
    assert Cascade([a, b]).understand("x", CTX).extraction.country == "uk"


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


def _ex(**kw: object) -> Extraction:
    return Extraction.model_validate(kw)


class Fixed:
    def __init__(self, name: str, ex: Extraction, confident: bool) -> None:
        self.name, self.ex, self.confident = name, ex, confident

    def understand(self, message: str, ctx: Context) -> BrainResult:
        return BrainResult(self.ex, self.confident, self.name)


def test_later_brain_keeps_country_and_purpose_it_left_out() -> None:
    rules = Fixed("rules", _ex(country="canada", purpose="family_visit"), False)
    model = Fixed("local", _ex(travellers=[{"relation": "mother"}]), True)
    ex = Cascade([rules, model], combine=fill_blanks).understand("x", CTX).extraction
    assert (ex.country, ex.purpose) == ("canada", "family_visit")
    assert [p.relation for p in ex.travellers] == ["mother"]


def test_later_brain_values_win_and_choices_block_purpose() -> None:
    rules = Fixed("rules", _ex(country="uk", purpose="tourism"), False)
    model = Fixed("local", _ex(country="japan", purpose_choices=["tourism", "business"]), True)
    ex = Cascade([rules, model], combine=fill_blanks).understand("x", CTX).extraction
    assert (ex.country, ex.purpose) == ("japan", None)


def test_off_topic_or_restart_answers_are_not_filled() -> None:
    rules = Fixed("rules", _ex(country="uk", other_country=None), False)
    for later in (_ex(on_topic=False), _ex(intent="restart")):
        ex = (
            Cascade([rules, Fixed("local", later, True)], combine=fill_blanks)
            .understand("x", CTX)
            .extraction
        )
        assert ex.country is None


def test_other_country_is_kept_too() -> None:
    rules = Fixed("rules", _ex(other_country="sri-lanka"), False)
    ex = (
        Cascade([rules, Fixed("local", _ex(), True)], combine=fill_blanks)
        .understand("x", CTX)
        .extraction
    )
    assert ex.other_country == "sri-lanka"


# ----------------------------------------------------------------- rules precede


def _rel(ex: Extraction) -> list[tuple[int | None, str | None, str | None, int | None]]:
    return [(p.ref, p.relation, p.profession, p.age) for p in ex.travellers]


def test_rules_decide_who_model_decides_details() -> None:
    # "i am a business owner wife is a school teacher": rules saw both, mixed up the jobs
    rules = _ex(
        travellers=[{"relation": "self"}, {"relation": "spouse", "profession": "business_owner"}]
    )
    model = _ex(
        travellers=[
            {"relation": "self", "profession": "business_owner"},
            {"relation": "spouse", "profession": "employed"},
        ]
    )
    assert _rel(rules_precede(rules, model)) == [
        (None, "self", "business_owner", None),
        (None, "spouse", "employed", None),
    ]


def test_model_inventions_are_dropped_and_omissions_added_back() -> None:
    # "Going to Canada to visit my son, just me": the son is a host; the model forgot "me"
    rules = _ex(country="canada", purpose="family_visit", travellers=[{"relation": "self"}])
    model = _ex(country="canada", purpose="family_visit", travellers=[{"relation": "son"}])
    assert _rel(rules_precede(rules, model)) == [(None, "self", None, None)]


def test_model_details_fill_and_rules_details_fill_back() -> None:
    rules = _ex(travellers=[{"relation": "daughter", "age": 3}, {"relation": "son"}])
    model = _ex(
        travellers=[
            {"relation": "daughter", "profession": "infant"},
            {"relation": "son", "profession": "infant"},
        ]
    )
    assert _rel(rules_precede(rules, model)) == [
        (None, "daughter", "infant", 3),
        (None, "son", "infant", None),
    ]


def test_rules_place_purpose_count_and_intent_win() -> None:
    rules = _ex(country="uk", purpose="business", traveller_count=3, intent="correct")
    model = _ex(country="japan", purpose="tourism", traveller_count=4, intent="confirm")
    ex = rules_precede(rules, model)
    assert (ex.country, ex.purpose, ex.traveller_count, ex.intent) == (
        "uk",
        "business",
        3,
        "correct",
    )


def test_model_fills_what_rules_found_nothing_for() -> None:
    rules = _ex()
    model = _ex(country="japan", purpose="tourism", travellers=[{"relation": "self"}])
    assert rules_precede(rules, model) == model


def test_refs_merge_and_rules_only_refs_kept() -> None:
    rules = _ex(travellers=[{"ref": 1, "remove": True}, {"ref": 0, "age": 40}])
    model = _ex(travellers=[{"ref": 0, "profession": "employed"}])
    assert _rel(rules_precede(rules, model)) == [(1, None, None, None), (0, None, "employed", 40)]
    assert rules_precede(rules, model).travellers[0].remove


def test_a_ref_and_a_relation_for_the_same_person_are_one_person() -> None:
    """Session 5: rules read 'my wife' by relation, the model updated traveller #1 by ref."""
    form = TripForm(
        travellers=(
            Traveller(relation="self"),
            Traveller(relation="spouse"),
            Traveller(relation="son"),
        )
    )
    rules = _ex(
        travellers=[{"relation": "self", "profession": "business_owner"}, {"relation": "spouse"}]
    )
    model = _ex(
        travellers=[
            {"ref": 0, "profession": "business_owner"},
            {"ref": 1, "profession": "self_employed_professional"},
        ]
    )
    assert _rel(rules_precede(rules, model, form)) == [
        (0, None, "business_owner", None),
        (1, None, "self_employed_professional", None),
    ]


def test_off_topic_is_the_models_call_unless_rules_saw_a_trip() -> None:
    question = _ex(country="japan")  # "what's the best time to visit Japan"
    assert rules_precede(question, _ex(on_topic=False)) == _ex(on_topic=False)
    trip = _ex(country="japan", travellers=[{"relation": "self"}])
    assert rules_precede(trip, _ex(on_topic=False)).on_topic


def test_restart_and_group_rules() -> None:
    assert rules_precede(_ex(), _ex(intent="restart")).intent == "restart"
    assert rules_precede(_ex(intent="restart"), _ex(country="uk")).intent == "restart"
    listed = rules_precede(
        _ex(travellers=[{"relation": "friend_colleague"}]), _ex(group_of="friend_colleague")
    )
    assert listed.group_of is None
    assert rules_precede(_ex(group_of="child"), _ex()).group_of == "child"


def test_choices_block_the_models_purpose_and_other_country_blocks_country() -> None:
    rules = _ex(purpose_choices=["tourism", "business"], other_country="sri-lanka")
    ex = rules_precede(rules, _ex(purpose="tourism", country="japan"))
    assert (ex.purpose, ex.country, ex.other_country) == (None, None, "sri-lanka")


def test_combine_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    assert combiner("rules") is rules_precede and combiner("model") is fill_blanks
    monkeypatch.setenv("GVS_COMBINE", "model")
    assert combiner() is fill_blanks
    with pytest.raises(UnknownBrain):
        combiner("vote")
