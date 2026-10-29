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

``fill_blanks``
    The model's answer is used as it is; the rules only fill in a missing
    country or purpose. This was the behaviour before ``rules_precede``.

Both are pure functions of two :class:`Extraction` values, so they work the
same for any later brain (local model now, Haiku later).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Final

from app.extraction import Extraction, TravellerPatch

Combine = Callable[[Extraction, Extraction], Extraction]

_DETAILS: Final = ("profession", "age", "country", "purpose")


def fill_blanks(rules: Extraction, model: Extraction) -> Extraction:
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


def _people(rules: Extraction, model: Extraction) -> tuple[TravellerPatch, ...]:
    if not rules.travellers:
        return model.travellers  # the rules saw nobody: nothing to check against
    # Updates to people already in the form, by ref: the model's reading, rules filling gaps.
    rules_refs = {p.ref: p for p in rules.travellers if p.ref is not None}
    model_refs = {p.ref: p for p in model.travellers if p.ref is not None}
    out = [_fill(p, rules_refs[r]) if r in rules_refs else p for r, p in model_refs.items()]
    out += [p for r, p in rules_refs.items() if r not in model_refs]
    # New people: the rules decide who, the model decides their details.
    rules_new = [p for p in rules.travellers if p.ref is None]
    model_new: dict[str, list[TravellerPatch]] = {}
    for p in model.travellers:
        if p.ref is None and p.relation is not None:
            model_new.setdefault(p.relation, []).append(p)
    for relation, n in Counter(p.relation for p in rules_new if p.relation).items():
        theirs = model_new.get(relation, [])[:n]
        mine = [p for p in rules_new if p.relation == relation]
        out += [_fill(m, r) for m, r in zip(theirs, mine, strict=False)]
        out += mine[len(theirs) :]
    return tuple(out)


def _kinds(ex: Extraction) -> int:
    """How many kinds of trip detail a reading found (place, purpose, people, count)."""
    place = ex.country or ex.other_country
    return sum(bool(x) for x in (place, ex.purpose, ex.travellers, ex.traveller_count))


def rules_precede(rules: Extraction, model: Extraction) -> Extraction:
    """See the module docstring. Off-topic is the model's call (it reads questions
    better), unless the rules found at least two kinds of trip detail."""
    if model.intent == "restart" or rules.intent == "restart":
        return model if model.intent == "restart" else rules
    on_topic = model.on_topic or _kinds(rules) >= 2
    if not on_topic:
        # "what's the best time to visit Japan": a question, not a trip; keep nothing
        return Extraction(on_topic=False, asks_office=rules.asks_office or model.asks_office)
    people = _people(rules, model)
    group = rules.group_of or model.group_of
    if group is not None and any(p.relation == group for p in people):
        group = None
    return Extraction(
        on_topic=True,
        intent=rules.intent if rules.intent != "give_details" else model.intent,
        country=rules.country or (None if rules.other_country else model.country),
        purpose=rules.purpose or (None if rules.purpose_choices else model.purpose),
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
