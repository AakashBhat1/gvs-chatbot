from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from app import templates as T
from app.brains import BrainResult, BrainUnavailable, Cascade
from app.brains.lexicon import load_lexicon
from app.brains.rules import RulesBrain
from app.chat import BotReply, ChatEngine, ChatState, merge, prune
from app.extraction import Context, Extraction, TravellerPatch
from app.form import Traveller, TripForm
from app.paths import default_lexicon_path
from app.tree import LoadedTree, load_tree


@pytest.fixture
def engine(real_tree: LoadedTree) -> ChatEngine:
    lex = load_lexicon(default_lexicon_path(), real_tree)
    return ChatEngine(lambda: real_tree, RulesBrain(lex), ask_age_for=lex.ask_age_for)


def talk(engine: ChatEngine, *messages: str) -> tuple[ChatState, BotReply]:
    state, reply = engine.start()
    for m in messages:
        state, reply = engine.handle(state, text=m)
    return state, reply


def choices(reply: BotReply) -> list[str]:
    return [b.choice for b in reply.buttons]


# ----------------------------------------------------------------- full conversations


def test_plan_example_conversation(engine: ChatEngine) -> None:
    state, reply = engine.start()
    assert reply.text == T.OPENING

    state, reply = engine.handle(state, text="family of four going to Japan")
    assert reply.text.startswith("Is the trip for tourism")
    assert choices(reply) == ["purpose:tourism", "purpose:family_visit", "purpose:business"]

    state, reply = engine.handle(state, choice="purpose:tourism")
    assert reply.text == T.ASK_WHO

    state, reply = engine.handle(state, text="me doctor, wife housewife, son is 8 and my mom")
    assert reply.text == "What does your mother do? (e.g. salaried, business, student, retired)"
    assert state.asked_travellers == (3,)

    state, reply = engine.handle(state, text="retired")
    assert state.stage == "confirming"
    assert reply.text.startswith(T.SUMMARY_HEAD)
    assert "• Your son: age 8, School student" in reply.text
    assert choices(reply) == ["confirm", "restart"]

    state, reply = engine.handle(state, text="mom is going to the UK instead")
    assert "United Kingdom, tourism" in reply.text

    state, reply = engine.handle(state, text="yes")
    assert state.stage == "done"
    assert reply.checklist is not None
    groups = reply.checklist.groups
    assert [g.country for g in groups] == ["japan", "uk"]
    assert [p.relation for p in groups[0].travellers] == ["self", "spouse", "son"]
    assert [d.id for d in groups[0].shared] == ["family_covering_letter"]
    assert choices(reply) == ["restart"]


def test_hinglish_conversation(engine: ChatEngine) -> None:
    state, reply = talk(
        engine,
        "Hum 3 log Thailand ghumne ja rahe hain",
        "main engineer hu, wife housewife hai aur beta 5 saal ka",
    )
    assert state.stage == "confirming", reply.text
    assert [t.relation for t in state.form.travellers] == ["self", "spouse", "son"]


def test_asks_ages_for_children_and_groups_questions(engine: ChatEngine) -> None:
    state, reply = talk(engine, "Japan tourism, me and my wife with our son and daughter")
    assert reply.text.startswith("What do you and your spouse do?")
    state, reply = engine.handle(state, text="both salaried")
    assert reply.text == "How old are your son and daughter?"
    state, reply = engine.handle(state, text="9 and 1")
    assert state.stage == "confirming"
    assert [t.age for t in state.form.travellers] == [None, None, 9, 1]


def test_count_gap_asks_for_more(engine: ChatEngine) -> None:
    _, reply = talk(engine, "family of 3 to Japan for tourism, just me so far")
    assert reply.text == "You mentioned 3 travellers and I have 1. Who else is going?"


# ----------------------------------------------------------------- confirm / correct / done


def _to_summary(engine: ChatEngine) -> ChatState:
    state, _ = talk(engine, "Japan tourism, just me, salaried")
    assert state.stage == "confirming"
    return state


def test_no_at_summary_asks_what_to_change(engine: ChatEngine) -> None:
    state, reply = engine.handle(_to_summary(engine), text="no")
    assert reply.text == T.WHAT_CHANGE
    state, reply = engine.handle(state, text="it's for business")
    assert "Conference or business meeting" in reply.text


def test_done_stage(engine: ChatEngine) -> None:
    state, _ = engine.handle(_to_summary(engine), choice="confirm")
    state, reply = engine.handle(state, text="ok")
    assert reply.text == T.ALREADY_DONE
    state, reply = engine.handle(state, text="actually Thailand")
    assert state.stage == "confirming"
    assert "Thailand" in reply.text


def test_unclear_reply_at_summary(engine: ChatEngine) -> None:
    state = _to_summary(engine)
    state, reply = engine.handle(state, text="hmm")
    assert reply.text.startswith(T.NOT_UNDERSTOOD)
    assert state.stage == "confirming"


def test_restart_by_text_and_button(engine: ChatEngine) -> None:
    state = _to_summary(engine)
    state, reply = engine.handle(state, text="start over")
    assert (state.form, reply.text) == (TripForm(), T.OPENING)
    state = _to_summary(engine)
    state, reply = engine.handle(state, choice="restart")
    assert state.form == TripForm()


# ----------------------------------------------------------------- misses, buttons, off-topic


def test_greeting_then_buttons_after_two_misses(engine: ChatEngine) -> None:
    state, reply = talk(engine, "hi")
    assert reply.text == T.OPENING
    state, reply = engine.handle(state, text="hmm")
    assert reply.text.startswith(T.NOT_UNDERSTOOD)
    assert "country:japan" in choices(reply)


def test_off_topic_first_and_later(engine: ChatEngine) -> None:
    _, reply = talk(engine, "write a poem")
    assert reply.text == T.OFF_TOPIC_START
    _, reply = talk(engine, "Japan", "tell me a joke")
    assert reply.text.startswith(f"{T.OFF_TOPIC} Is the trip for tourism")


def test_traveller_and_profession_buttons(engine: ChatEngine) -> None:
    state, reply = talk(engine, "Japan tourism", "xyz", "abc")
    assert reply.text.endswith(T.BUTTONS_HINT)
    assert choices(reply)[0] == "solo"
    state, reply = engine.handle(state, choice="add:self")
    state, reply = engine.handle(state, choice="add:son")
    state, reply = engine.handle(state, choice="add:self")  # duplicate self is ignored
    assert [t.relation for t in state.form.travellers] == ["self", "son"]
    state, reply = engine.handle(state, text="qq")
    state, reply = engine.handle(state, text="ww")
    assert "prof:0:employed" in choices(reply)
    assert "prof:0:infant" not in choices(reply)
    state, reply = engine.handle(state, choice="prof:0:retired")
    state, reply = engine.handle(state, text="??")
    state, reply = engine.handle(state, text="!!")
    assert choices(reply) == [
        "prof:1:infant",
        "prof:1:school_student",
        "prof:1:university_student",
        "prof:1:employed",
    ]
    state, reply = engine.handle(state, choice="prof:1:infant")
    assert state.stage == "confirming"


def test_buttons_for_everything(engine: ChatEngine) -> None:
    state, _ = engine.start()
    for choice in ("country:japan", "purpose:business", "solo", "prof:0:farmer"):
        state, reply = engine.handle(state, choice=choice)
    assert state.stage == "confirming"
    state, reply = engine.handle(state, choice="confirm")
    assert reply.checklist is not None


def test_everyone_button_clears_expected_count(engine: ChatEngine) -> None:
    state, _ = talk(engine, "family of 4 to Japan for tourism, me salaried")
    state, _ = engine.handle(state, choice="everyone")
    assert state.form.expected_count is None
    assert state.stage == "confirming"


@pytest.mark.parametrize(
    "choice",
    [
        "country:mars",
        "prof:7:retired",
        "prof:x:retired",
        "add:uncle",
        "DROP TABLE",
        "confirm",
        "everyone",
        "a:b:c:d",
    ],
)
def test_forged_or_stale_buttons_are_ignored(engine: ChatEngine, choice: str) -> None:
    state, _ = talk(engine, "Japan tourism")
    new, reply = engine.handle(state, choice=choice)
    assert new.form == state.form
    assert reply.checklist is None


def test_button_after_summary_returns_to_collecting(engine: ChatEngine) -> None:
    state, _ = engine.handle(_to_summary(engine), choice="country:uk")
    assert state.form.country == "uk"
    assert state.stage == "confirming"


# ----------------------------------------------------------------- limits & robustness


def test_too_long_and_empty_messages(engine: ChatEngine) -> None:
    state, reply = engine.start()
    state, reply = engine.handle(state, text="x" * 501)
    assert reply.text.startswith("That message is too long")
    state, reply = engine.handle(state, text="   \u200b ")
    assert reply.text == T.ASK_COUNTRY


def test_turn_limit(real_tree: LoadedTree) -> None:
    lex = load_lexicon(default_lexicon_path(), real_tree)
    engine = ChatEngine(lambda: real_tree, RulesBrain(lex), max_turns=2)
    state, _ = talk(engine, "hi", "hi")
    state, reply = engine.handle(state, text="Japan")
    assert reply.text == T.TURN_LIMIT
    assert state.form == TripForm()
    _, reply = engine.handle(state, choice="restart")
    assert reply.text == T.OPENING


def test_more_than_eight_travellers(engine: ChatEngine) -> None:
    state, reply = talk(engine, "Japan tourism", "me, wife, mom, dad, 3 kids, brother, grandma")
    assert len(state.form.travellers) == 8
    assert reply.text.startswith("I can check up to 8 travellers")
    state, reply = talk(engine, "family of 12 going to Japan")
    assert reply.text.startswith("I can check up to 8 travellers")
    assert state.form.expected_count is None


class _Down:
    name = "down"

    def understand(self, message: str, ctx: Context) -> BrainResult:
        raise BrainUnavailable("offline")


def test_all_brains_down_shows_buttons(real_tree: LoadedTree) -> None:
    engine = ChatEngine(lambda: real_tree, Cascade([_Down()]))
    state, _ = engine.start()
    state, reply = engine.handle(state, text="Japan")
    assert reply.text.startswith(T.NOT_UNDERSTOOD)
    assert "country:japan" in choices(reply)


def test_tree_reload_prunes_removed_ids(
    engine: ChatEngine, mini_dict: dict[str, Any], write_tree: Callable[[dict[str, Any]], Path]
) -> None:
    state, _ = talk(engine, "Japan tourism, just me, salaried")
    swapped = load_tree(write_tree(mini_dict))  # knows "jp", not "japan"
    other = ChatEngine(lambda: swapped, engine._brain)
    state, _ = other.handle(state, text="hmm")
    assert state.tree_version == swapped.version
    assert state.form.country is None
    assert state.form.travellers == (Traveller(relation="self", profession="employed"),)


# ----------------------------------------------------------------- merge rules


def _ex(**kw: Any) -> Extraction:
    return Extraction.model_validate(kw)


def test_merge_fills_gaps_but_keeps_values() -> None:
    form = TripForm(country="japan", travellers=(Traveller(relation="son", age=5),))
    new, _ = merge(form, _ex(country="uk", purpose="tourism"), overwrite=False)
    assert (new.country, new.purpose) == ("japan", "tourism")
    new, _ = merge(form, _ex(country="uk"), overwrite=True)
    assert new.country == "uk"


def test_merge_second_son_with_other_age_is_new_person() -> None:
    form = TripForm(travellers=(Traveller(relation="son", age=5),))
    new, _ = merge(form, _ex(travellers=[{"relation": "son", "age": 9}]), overwrite=False)
    assert [t.age for t in new.travellers] == [5, 9]
    new, _ = merge(
        form, _ex(travellers=[{"relation": "son", "profession": "school_student"}]), overwrite=False
    )
    assert len(new.travellers) == 1
    new, _ = merge(form, _ex(travellers=[{"relation": "son", "age": 6}]), overwrite=True)
    assert [t.age for t in new.travellers] == [6]


def test_merge_self_is_unique_and_ref_wins() -> None:
    form = TripForm(
        travellers=(Traveller(relation="self", profession="employed"), Traveller(relation="mother"))
    )
    patches = [{"relation": "self", "profession": "retired"}, {"ref": 1, "profession": "homemaker"}]
    new, _ = merge(form, _ex(travellers=patches), overwrite=False)
    assert [t.profession for t in new.travellers] == ["employed", "homemaker"]


def test_merge_override_equal_to_trip_is_cleared() -> None:
    form = TripForm(country="japan", travellers=(Traveller(relation="mother", country="uk"),))
    new, _ = merge(
        form, _ex(travellers=[{"relation": "mother", "country": "japan"}]), overwrite=True
    )
    assert new.travellers[0].country is None
    new, _ = merge(
        form, _ex(travellers=[{"relation": "mother", "country": "japan"}]), overwrite=False
    )
    assert new.travellers[0].country == "uk"


def test_merge_remove_and_expected_count() -> None:
    form = TripForm(
        travellers=(Traveller(relation="self"), Traveller(relation="mother")), expected_count=4
    )
    new, _ = merge(form, _ex(travellers=[{"relation": "mother", "remove": True}]), overwrite=True)
    assert [t.relation for t in new.travellers] == ["self"]
    new, _ = merge(form, _ex(traveller_count=2), overwrite=False)
    assert new.expected_count == 4  # a smaller count doesn't shrink without a correction
    new, _ = merge(form, _ex(traveller_count=2), overwrite=True)
    assert new.expected_count is None  # 2 <= people already listed
    new, _ = merge(form, _ex(traveller_count=6), overwrite=False)
    assert new.expected_count == 6
    new, _ = merge(form, _ex(travellers=[{"relation": "father", "remove": True}]), overwrite=True)
    assert new == form


def test_merge_ignores_patch_without_person() -> None:
    patch = TravellerPatch(ref=5, profession="retired")
    new, overflow = merge(TripForm(), Extraction(travellers=(patch,)), overwrite=False)
    assert (new, overflow) == (TripForm(), False)


def test_prune_keeps_valid_form(real_tree: LoadedTree) -> None:
    form = TripForm(country="japan", travellers=(Traveller(relation="self"),))
    assert prune(form, real_tree) is form


def test_state_json_round_trip(engine: ChatEngine) -> None:
    state = _to_summary(engine)
    data = json.loads(state.model_dump_json())
    assert set(data) == {
        "form",
        "stage",
        "asked_field",
        "asked_travellers",
        "misses",
        "turns",
        "tree_version",
    }
    assert ChatState.model_validate(data) == state


# ----------------------------------------------------------------- groups and "Same"


def _pending_friends(engine: ChatEngine) -> ChatState:
    state, _ = talk(engine, "uk tourism with my friends")
    assert state.asked_field == "group_size"
    assert state.form.pending_group == "friend_colleague"
    return state


@pytest.mark.parametrize("choice", ["group:0", "group:8", "group:x", "group:1:2", "same:pilot"])
def test_forged_group_and_same_buttons_are_ignored(engine: ChatEngine, choice: str) -> None:
    state = _pending_friends(engine)
    new, _ = engine.handle(state, choice=choice)
    assert new.form == state.form


def test_group_button_without_pending_group_is_ignored(engine: ChatEngine) -> None:
    state, _ = talk(engine, "Japan tourism")
    new, _ = engine.handle(state, choice="group:3")
    assert new.form == state.form


def test_same_button_only_answers_a_profession_question(engine: ChatEngine) -> None:
    state = _pending_friends(engine)
    new, _ = engine.handle(state, choice="same:retired")
    assert new.form == state.form


def test_group_buttons_are_capped_by_seats_left(engine: ChatEngine) -> None:
    state, reply = talk(engine, "uk tourism, me, my wife, 2 sons, 2 daughters and my colleagues")
    assert len(state.form.travellers) == 6
    assert choices(reply) == ["group:1", "group:2"]
    state, _ = engine.handle(state, choice="group:2")
    assert len(state.form.travellers) == 8
    assert state.form.pending_group is None


def test_everyone_button_clears_pending_group(engine: ChatEngine) -> None:
    state = _pending_friends(engine)
    state, _ = engine.handle(state, choice="everyone")
    assert state.form.pending_group is None


def test_merge_group_rules() -> None:
    me = (Traveller(relation="self"),)
    new, _ = merge(TripForm(travellers=me), _ex(group_of="friend_colleague"), overwrite=False)
    assert new.pending_group == "friend_colleague"
    # naming people of another relation keeps the question; naming the group settles it
    spouse = _ex(travellers=[{"relation": "spouse"}])
    assert merge(new, spouse, overwrite=False)[0].pending_group == "friend_colleague"
    friend = _ex(travellers=[{"relation": "friend_colleague"}] * 2)
    assert merge(new, friend, overwrite=False)[0].pending_group is None
    assert merge(new, _ex(traveller_count=1), overwrite=False)[0].pending_group is None
    full = TripForm(travellers=tuple(Traveller(relation="son") for _ in range(8)))
    capped, overflow = merge(full, _ex(group_of="friend_colleague"), overwrite=False)
    assert capped.pending_group is None and overflow


def test_prune_drops_unknown_pending_group(
    engine: ChatEngine, mini_dict: dict[str, Any], write_tree: Callable[[dict[str, Any]], Path]
) -> None:
    form = TripForm(travellers=(Traveller(relation="self"),), pending_group="no_such_relation")
    assert prune(form, load_tree(write_tree(mini_dict))).pending_group is None
