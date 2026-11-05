"""How a later brain's answer is merged with the rules brain's.

The cascade only asks a model when the rules are unsure, so both answers are
available. A model cannot say reliably when *it* is unsure, so the rules'
reading is the anchor:

``rules_precede`` (default)
    The rules win wherever they read something precisely: the destination,
    the purpose, a stated count, a yes/no/correction, and *which people* the
    message names. The model decides what the rules are bad at: who has which
    job or age, and anything the rules found nothing for. A person the model
    adds that the rules did not see (the son being visited, a spouse invented
    for "two adults") is dropped; a person the rules saw that the model left
    out ("just me") is added back.

    A purpose only the model saw is dropped too, unless the bot just asked for
    the purpose: models guess one from a job ("I own a restaurant" becomes a
    business trip), a wrong one changes the documents, and asking costs one
    question.

    The model may move a job the rules read to the right person, but may not
    swap it for another: if that job appears nowhere in the model's answer, it
    only recategorised the word ("works in a bank" became government employee).
    A job with its own documents and a clear word for it, like government employee
    or doctor, is dropped when only the model read it, so the bot asks instead.

    A model's answer can be steered by the message, so what it may do alone is
    limited. Only the rules can confirm or remove someone. For a person the rules
    did not read, a model may only fill the empty job or age the bot just asked
    about. It may add someone new, but may not overwrite anything already on the
    form, including the country.

``fill_blanks``
    The model's answer is used as it is; the rules only fill in a missing
    country or purpose. This was the behaviour before ``rules_precede``.

Both are pure functions of two :class:`Extraction` values, so they work the
same for any later brain (local model now, Haiku later).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final

from app.extraction import Asked, Extraction, TravellerPatch
from app.form import MAX_TRAVELLERS, TripForm

# (rules' answer, later brain's answer, the form so far, what the bot last asked)
Combine = Callable[[Extraction, Extraction, TripForm, Asked | None], Extraction]

_DETAILS: Final = ("profession", "age", "country", "purpose")
_ANSWERABLE: Final = ("profession", "age")
# Each has its own documents and its real words are in the lexicon, so if the rules did
# not read one, a model guessed it ("i play games for a living" became a doctor or lawyer).
_NEEDS_A_REAL_WORD: Final = frozenset(
    {
        "self_employed_professional",
        "government_employee",
        "defence_personnel",
        "company_director",
    }
)
BABY: Final = "infant"


def fill_blanks(
    rules: Extraction,
    model: Extraction,
    form: TripForm | None = None,
    asked: Asked | None = None,
) -> Extraction:
    if not model.on_topic or model.intent == "restart":
        return model
    update: dict[str, object] = {}
    if model.country is None and model.other_country is None:
        if rules.country is not None:
            update["country"] = rules.country
        elif rules.other_country is not None:
            update["other_country"] = rules.other_country
    if model.purpose is None and not model.purpose_choices and rules.purpose is not None:
        update["purpose"] = rules.purpose
    return model.model_copy(update=update) if update else model


def _fill(patch: TravellerPatch, other: TravellerPatch) -> TravellerPatch:
    """``patch`` with any detail it lacks taken from ``other``."""
    missing = {f: getattr(other, f) for f in _DETAILS if getattr(patch, f) is None}
    update = {f: v for f, v in missing.items() if v is not None}
    return patch.model_copy(update=update) if update else patch


def _without_purpose(patch: TravellerPatch) -> TravellerPatch:
    return patch.model_copy(update={"purpose": None}) if patch.purpose is not None else patch


def _keep_rules_reading(
    merged: TravellerPatch, mine: TravellerPatch, model_jobs: set[str | None]
) -> TravellerPatch:
    """Undo a model's recategorised job, and an age it read from "5 months" on a baby."""
    update: dict[str, object] = {}
    if mine.profession is not None and mine.profession not in model_jobs:
        update["profession"] = mine.profession
    if mine.profession == BABY and mine.age is None and merged.age is not None:
        update["age"] = None
    return merged.model_copy(update=update) if update else merged


def _still_unanswered(form: TripForm, asked: Asked | None) -> list[int]:
    """The people the bot just asked about whose job or age is still empty."""
    if asked is None or asked.field not in _ANSWERABLE:
        return []
    people = form.travellers
    return [
        i for i in asked.travellers if i < len(people) and getattr(people[i], asked.field) is None
    ]


def _model_alone(
    patches: Sequence[TravellerPatch], form: TripForm, asked: Asked | None
) -> list[TravellerPatch]:
    """What a model may do for people the rules did not read: see the module docstring.

    A patch is about someone on the form by ref or by relation: ``merge`` finds people
    by either, so checking refs alone would let ``{"relation": "self"}`` overwrite the writer.
    """
    people = form.travellers
    known = {t.relation for t in people}
    open_refs = _still_unanswered(form, asked)
    out: list[TravellerPatch] = []
    for patch in patches:
        on_form = patch.ref is not None and patch.ref < len(people)
        if not on_form and patch.relation not in known:
            if patch.relation is not None and not patch.remove:
                out.append(patch)  # someone new
            continue
        who = next(
            (
                i
                for i in open_refs
                if i == patch.ref or (not on_form and people[i].relation == patch.relation)
            ),
            None,
        )
        value = getattr(patch, asked.field) if who is not None and asked is not None else None
        if who is None or asked is None or value is None:
            continue
        open_refs = [i for i in open_refs if i != who]
        out.append(TravellerPatch.model_validate({"ref": who, asked.field: value}))
    return out


def _without_guessed_titles(
    people: Sequence[TravellerPatch], rules: Extraction, form: TripForm
) -> tuple[TravellerPatch, ...]:
    """Drop a job that needs a real word when neither the rules nor the form have it."""
    read = {p.profession for p in rules.travellers} | {t.profession for t in form.travellers}
    return tuple(
        p.model_copy(update={"profession": None})
        if p.profession in _NEEDS_A_REAL_WORD and p.profession not in read
        else p
        for p in people
    )


def _people(
    rules: Extraction, model: Extraction, form: TripForm, asked: Asked | None
) -> tuple[TravellerPatch, ...]:
    """The rules decide who; the model decides their details.

    Each person the rules read is paired with the model's entry for the same
    person, by ref or by relation, so "my wife" (rules, by relation) and the
    model's update to traveller #1 (by ref) are one person, not two. A model
    entry with no rules counterpart is kept only if it updates someone already
    in the form; a new person only the model saw is dropped.
    """
    if asked is None or asked.field != "purpose":
        model = model.model_copy(
            update={"travellers": tuple(_without_purpose(p) for p in model.travellers)}
        )
    if not rules.travellers:  # the rules saw nobody: nothing to check against
        return tuple(_model_alone(model.travellers, form, asked)[:MAX_TRAVELLERS])

    def who(p: TravellerPatch) -> str | None:
        if p.ref is not None and p.ref < len(form.travellers):
            return form.travellers[p.ref].relation
        return p.relation

    left = list(model.travellers)
    model_jobs = {p.profession for p in model.travellers}
    out: list[TravellerPatch] = []
    for mine in rules.travellers:
        theirs = next((p for p in left if mine.ref is not None and p.ref == mine.ref), None)
        theirs = theirs or next(
            (
                p
                for p in left
                if who(p) == who(mine) and (p.ref is None or mine.ref is None or p.ref == mine.ref)
            ),
            None,
        )
        if theirs is None:
            out.append(mine)  # the model missed them ("just me"): keep the rules' reading
            continue
        left.remove(theirs)
        ref = theirs.ref if theirs.ref is not None else mine.ref
        merged = _fill(theirs, mine).model_copy(update={"ref": ref, "remove": mine.remove})
        out.append(_keep_rules_reading(merged, mine, model_jobs))
    out += _model_alone([p for p in left if p.ref is not None], form, asked)
    return tuple(out[:MAX_TRAVELLERS])  # the rules' people come first


def _kinds(ex: Extraction) -> int:
    """How many kinds of trip detail a reading found (place, purpose, people, count)."""
    place = ex.country or ex.other_country
    return sum(bool(x) for x in (place, ex.purpose, ex.travellers, ex.traveller_count))


def rules_precede(
    rules: Extraction,
    model: Extraction,
    form: TripForm | None = None,
    asked: Asked | None = None,
) -> Extraction:
    """See the module docstring. Off-topic is the model's call (it reads questions
    better), unless the rules found at least two kinds of trip detail."""
    if model.intent == "restart" or rules.intent == "restart":
        return model if model.intent == "restart" else rules
    on_topic = model.on_topic or _kinds(rules) >= 2
    if not on_topic:
        # "what's the best time to visit Japan": a question, not a trip; keep nothing
        return Extraction(on_topic=False, asks_office=rules.asks_office or model.asks_office)
    so_far = form or TripForm()
    people = _without_guessed_titles(_people(rules, model, so_far, asked), rules, so_far)
    group = rules.group_of or model.group_of
    if group is not None and any(p.relation == group for p in people):
        group = None
    asked_purpose = asked is not None and asked.field == "purpose"
    can_use_model_purpose = asked_purpose and not rules.purpose_choices
    model_country = None if rules.other_country or so_far.country else model.country
    model_intent = "give_details" if model.intent == "confirm" else model.intent
    return Extraction(
        on_topic=True,
        intent=rules.intent if rules.intent != "give_details" else model_intent,
        country=rules.country or model_country,
        purpose=rules.purpose or (model.purpose if can_use_model_purpose else None),
        traveller_count=rules.traveller_count or model.traveller_count,
        travellers=people,
        other_country=rules.other_country or (None if rules.country else model.other_country),
        others_coming=rules.others_coming or model.others_coming,
        group_of=group,
        purpose_choices=rules.purpose_choices or model.purpose_choices,
        asks_office=rules.asks_office or model.asks_office,
        asks_help=rules.asks_help or model.asks_help,
    )


COMBINERS: Final[dict[str, Combine]] = {"rules": rules_precede, "model": fill_blanks}
