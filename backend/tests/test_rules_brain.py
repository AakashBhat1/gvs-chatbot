from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import templates as T
from app.brains.lexicon import Lexicon, load_lexicon
from app.brains.rules import RulesBrain
from app.chat import ChatEngine, ChatState
from app.evaluate import load_cases, run_case, score
from app.extraction import Asked, Context, Extraction
from app.form import Traveller, TripForm
from app.paths import default_lexicon_path
from app.tree import LoadedTree

EVALS = Path(__file__).resolve().parents[1] / "evals"


@pytest.fixture(scope="module")
def lexicon(real_tree_module: LoadedTree) -> Lexicon:
    return load_lexicon(default_lexicon_path(), real_tree_module)


@pytest.fixture(scope="module")
def brain(lexicon: Lexicon) -> RulesBrain:
    return RulesBrain(lexicon)


def ex(brain: RulesBrain, msg: str, ctx: Context | None = None) -> Extraction:
    return brain.understand(msg, ctx or Context(TripForm(), None, "collecting")).extraction


def people(e: Extraction) -> list[tuple[str | None, str | None, int | None]]:
    return [(p.relation, p.profession, p.age) for p in e.travellers]


# ----------------------------------------------------------------- scored sets (regression gates)


def test_dev_set_gate(brain: RulesBrain, real_tree_module: LoadedTree) -> None:
    cases = load_cases(EVALS / "understanding.jsonl")
    assert len(cases) >= 60
    s = score([run_case(brain, c, real_tree_module) for c in cases])
    assert s["exact"] >= 0.95, s
    assert s["intent"] == 1.0


def test_holdout_floor(brain: RulesBrain, real_tree_module: LoadedTree) -> None:
    """A floor against regressions, not a target: this set is never tuned against."""
    s = score([run_case(brain, c, real_tree_module) for c in load_cases(EVALS / "holdout.jsonl")])
    assert s["exact"] >= 0.60, s
    assert s["intent"] >= 0.95, s


# ----------------------------------------------------------------- attack suite (hard gate)


ATTACKS = [
    json.loads(line)["message"]
    for line in (EVALS / "attacks.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("//")
]
SAFE_PREFIXES = (T.OFF_TOPIC_START, T.OFF_TOPIC, T.OPENING, T.NOT_UNDERSTOOD)
PARTIAL = TripForm(country="japan", purpose="tourism", travellers=(Traveller(relation="self"),))


def test_attack_suite_size() -> None:
    assert len(ATTACKS) == 40


@pytest.mark.parametrize("message", ATTACKS)
@pytest.mark.parametrize("start", ["empty", "partial"])
def test_attack_never_changes_form(
    message: str, start: str, real_tree_module: LoadedTree, brain: RulesBrain, lexicon: Lexicon
) -> None:
    engine = ChatEngine(lambda: real_tree_module, brain, ask_age_for=lexicon.ask_age_for)
    state, _ = engine.start()
    if start == "partial":
        state = state.model_copy(
            update={"form": PARTIAL, "asked_field": "profession", "asked_travellers": (0,)}
        )
    new_state, reply = engine.handle(state, text=message)
    assert new_state.form == state.form
    assert reply.checklist is None
    assert reply.text.startswith(SAFE_PREFIXES)


# ----------------------------------------------------------------- specific behaviours


def test_host_is_not_a_traveller(brain: RulesBrain) -> None:
    e = ex(brain, "going to Canada to visit my son")
    assert (e.country, e.purpose, e.travellers) == ("canada", "family_visit", ())
    e = ex(brain, "bhai se milne Dubai ja rahe hain mummy papa")
    assert {p.relation for p in e.travellers} == {"mother", "father"}


def test_negated_values_are_ignored(brain: RulesBrain) -> None:
    ctx = Context(TripForm(country="uk", purpose="tourism"), Asked("confirm"), "confirming")
    e = ex(brain, "business, not tourism", ctx)
    assert e.purpose == "business"
    assert e.intent == "correct"


def test_weak_hindi_numbers(brain: RulesBrain) -> None:
    assert ex(brain, "do log Japan ja rahe hain").traveller_count == 2
    assert ex(brain, "my teen son").traveller_count is None
    e = ex(brain, "what do you do")
    assert not e.has_details


def test_hindi_me_after_country_means_in(brain: RulesBrain) -> None:
    assert ex(brain, "Japan me ghumne").travellers == ()
    assert people(ex(brain, "me and my wife")) == [("self", None, None), ("spouse", None, None)]


def test_bare_me_with_unknown_words_dropped(brain: RulesBrain) -> None:
    e = ex(brain, "order me a pizza")
    assert not e.has_details
    assert not e.on_topic
    assert people(ex(brain, "me")) == [("self", None, None)]


def test_off_topic_cue_wins(brain: RulesBrain) -> None:
    e = ex(brain, "tell me a joke about lawyers")
    assert (e.on_topic, e.has_details) == (False, False)


def test_plural_and_counts(brain: RulesBrain) -> None:
    assert people(ex(brain, "3 kids aged 2, 5 and 9")) == [
        ("child", None, 2),
        ("child", None, 5),
        ("child", None, 9),
    ]
    assert ex(brain, "two adults and two kids").traveller_count == 4
    assert ex(brain, "hum dono").traveller_count == 2
    assert ex(brain, "couple").traveller_count == 2
    assert people(ex(brain, "my parents")) == [("mother", None, None), ("father", None, None)]


def test_month_ages_and_bounds(brain: RulesBrain) -> None:
    assert people(ex(brain, "son 18 months old")) == [("son", None, 1)]
    r = brain.understand("son 150", Context(TripForm(), None, "collecting"))
    assert people(r.extraction) == [("son", None, None)]
    assert not r.confident


def test_pronoun_refers_to_last_person(brain: RulesBrain) -> None:
    e = ex(brain, "my wife and I. She is a nurse and I am retired.")
    assert people(e) == [("spouse", "employed", None), ("self", "retired", None)]


def test_retired_wins_over_other_profession(brain: RulesBrain) -> None:
    assert people(ex(brain, "father, army, retired")) == [("father", "retired", None)]


def test_confidence_reflects_unknown_words(brain: RulesBrain) -> None:
    ctx = Context(TripForm(), None, "collecting")
    assert brain.understand("Japan tourism, just me", ctx).confident
    assert not brain.understand("Japan tourism, just me, staying at Hilton", ctx).confident


def test_devanagari_is_understood_or_flagged(brain: RulesBrain) -> None:
    understood = brain.understand("मुझे जापान जाना है", Context(TripForm(), None, "collecting"))
    assert understood.extraction.country == "japan"
    r = brain.understand("क्या हाल है", Context(TripForm(), None, "collecting"))
    assert r.extraction.on_topic
    assert not r.extraction.has_details


def test_two_trip_countries_is_ambiguous(brain: RulesBrain) -> None:
    r = brain.understand("japan or thailand", Context(TripForm(), None, "collecting"))
    assert r.extraction.country == "japan"
    assert not r.confident


def test_remove_needs_a_person(brain: RulesBrain) -> None:
    r = brain.understand("not going", Context(TripForm(), None, "collecting"))
    assert not r.confident


def test_context_answer_only_fills_asked_field(brain: RulesBrain) -> None:
    form = TripForm(travellers=(Traveller(relation="self"), Traveller(relation="son")))
    asked_age = Context(form, Asked("age", (1,)), "collecting")
    assert [(p.ref, p.age) for p in ex(brain, "7", asked_age).travellers] == [(1, 7)]
    asked_prof = Context(form, Asked("profession", (0,)), "collecting")
    assert ex(brain, "7", asked_prof).travellers == ()
    assert [(p.ref, p.profession) for p in ex(brain, "retired", asked_prof).travellers] == [
        (0, "retired")
    ]


def test_restart_beats_everything(brain: RulesBrain) -> None:
    assert ex(brain, "start over, japan").intent == "restart"


def test_state_never_holds_raw_text(real_tree_module: LoadedTree, brain: RulesBrain) -> None:
    engine = ChatEngine(lambda: real_tree_module, brain)
    state, _ = engine.start()
    personal = "my passport number is Z1234567 and name Ravi Kumar"
    state, _ = engine.handle(state, text=f"Japan tourism just me. {personal}")
    dumped = state.model_dump_json()
    for word in ("Z1234567", "Ravi", "Kumar", "passport"):
        assert word.lower() not in dumped.lower()
    assert ChatState.model_validate_json(dumped) == state
