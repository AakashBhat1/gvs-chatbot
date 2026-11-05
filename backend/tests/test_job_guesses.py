"""A job nobody said must not become a job with its own document list."""

from __future__ import annotations

import pytest

from app.brains import Cascade
from app.brains.combine import rules_precede
from app.brains.lexicon import Lexicon, load_lexicon
from app.brains.local import system_prompt
from app.brains.rules import RulesBrain
from app.chat import ChatEngine, ChatState
from app.extraction import Asked, Context, Extraction
from app.form import Traveller, TripForm
from app.paths import default_lexicon_path
from app.tree import LoadedTree

ME = TripForm(country="france", purpose="tourism", travellers=(Traveller(relation="self"),))
GUESSED = (
    "self_employed_professional",
    "government_employee",
    "defence_personnel",
    "company_director",
)


def _ex(**kw: object) -> Extraction:
    return Extraction.model_validate(kw)


@pytest.fixture(scope="module")
def lexicon(real_tree_module: LoadedTree) -> Lexicon:
    return load_lexicon(default_lexicon_path(), real_tree_module)


@pytest.fixture
def brain(lexicon: Lexicon) -> RulesBrain:
    return RulesBrain(lexicon)


# ----------------------------------------------------------------- only the model read it


@pytest.mark.parametrize("job", GUESSED)
def test_a_title_only_the_model_read_is_dropped_at_the_job_question(job: str) -> None:
    model = _ex(travellers=[{"ref": 0, "profession": job}])
    merged = rules_precede(_ex(), model, ME, Asked("profession", (0,)))
    assert all(p.profession is None for p in merged.travellers)


@pytest.mark.parametrize("job", GUESSED)
def test_a_title_only_the_model_read_is_dropped_for_someone_the_rules_named(job: str) -> None:
    rules = _ex(travellers=[{"relation": "self"}])
    model = _ex(travellers=[{"relation": "self", "profession": job}])
    assert rules_precede(rules, model).travellers[0].profession is None


@pytest.mark.parametrize("job", GUESSED)
def test_a_title_only_the_model_read_is_dropped_for_someone_new(job: str) -> None:
    model = _ex(travellers=[{"relation": "self", "profession": job}])
    merged = rules_precede(_ex(), model)
    assert [(p.relation, p.profession) for p in merged.travellers] == [("self", None)]


def test_other_jobs_only_the_model_read_are_kept() -> None:
    model = _ex(travellers=[{"ref": 0, "profession": "employed"}])
    merged = rules_precede(_ex(), model, ME, Asked("profession", (0,)))
    assert merged.travellers[0].profession == "employed"


def test_a_title_the_rules_read_is_kept() -> None:
    rules = _ex(travellers=[{"relation": "self", "profession": "government_employee"}])
    model = _ex(travellers=[{"relation": "self", "profession": "government_employee"}])
    assert rules_precede(rules, model).travellers[0].profession == "government_employee"


def test_a_model_may_move_a_title_the_rules_read_to_the_right_person() -> None:
    rules = _ex(
        travellers=[
            {"relation": "self"},
            {"relation": "spouse", "profession": "self_employed_professional"},
        ]
    )
    model = _ex(
        travellers=[
            {"relation": "self", "profession": "self_employed_professional"},
            {"relation": "spouse", "profession": "employed"},
        ]
    )
    jobs = [p.profession for p in rules_precede(rules, model).travellers]
    assert jobs == ["self_employed_professional", "employed"]


def test_a_model_repeating_a_title_already_on_the_form_does_not_erase_it() -> None:
    form = TripForm(travellers=(Traveller(relation="self", profession="defence_personnel"),))
    rules = _ex(travellers=[{"relation": "self"}])
    model = _ex(travellers=[{"ref": 0, "profession": "defence_personnel"}])
    assert rules_precede(rules, model, form).travellers[0].profession == "defence_personnel"


def test_the_prompt_tells_the_model_not_to_guess_a_job(real_tree: LoadedTree) -> None:
    prompt = system_prompt(real_tree)
    assert 'If the job is not clearly one of those, use "other"' in prompt
    assert 'Never "other"' not in prompt


# ----------------------------------------------------------------- words the rules now know


@pytest.mark.parametrize(
    "answer",
    ["gamer", "streamer", "youtuber", "influencer", "content creator", "gig worker"],
)
def test_online_and_gig_work_is_freelance(answer: str, brain: RulesBrain) -> None:
    ctx = Context(ME, Asked("profession", (0,)), "collecting")
    assert brain.understand(answer, ctx).extraction.travellers[0].profession == "freelancer"


@pytest.mark.parametrize(
    "answer",
    [
        "I run a bakery",
        "restaurant",
        "I own a cafe",
        "my own salon",
        "I own a garage",
        "my wife owns a boutique",
        "I have a shop",
    ],
)
def test_owning_a_small_business_is_business_owner(answer: str, brain: RulesBrain) -> None:
    ctx = Context(ME, Asked("profession", (0,)), "collecting")
    jobs = {p.profession for p in brain.understand(answer, ctx).extraction.travellers}
    assert jobs & {"business_owner"}


# ----------------------------------------------------------------- the buttons


def _state(tree: LoadedTree, **kw: object) -> ChatState:
    return ChatState(tree_version=tree.version, **kw)  # type: ignore[arg-type]


def test_a_job_the_bot_cannot_place_gets_buttons_on_the_first_miss(
    brain: RulesBrain, real_tree: LoadedTree
) -> None:
    engine = ChatEngine(lambda: real_tree, Cascade([brain]))
    state = _state(real_tree, form=ME, asked_field="profession", asked_travellers=(0,))
    new, reply = engine.handle(state, text="blorptician")
    assert new.form == ME
    assert "Other" in " ".join(b.label for b in reply.buttons)


def test_the_first_miss_at_the_age_question_still_has_no_buttons(
    brain: RulesBrain, real_tree: LoadedTree
) -> None:
    form = TripForm(
        country="france",
        purpose="tourism",
        travellers=(Traveller(relation="self", profession="employed"), Traveller(relation="son")),
    )
    engine = ChatEngine(lambda: real_tree, Cascade([brain]))
    state = _state(real_tree, form=form, asked_field="age", asked_travellers=(1,))
    _, reply = engine.handle(state, text="blorptician")
    assert reply.buttons == ()
