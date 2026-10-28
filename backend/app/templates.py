"""Every sentence the bot can say.

Replies are built only from these templates and labels in ``visa-tree.json``.
No brain ever writes user-visible text, so a prompt injection cannot make the
bot say anything new. Button ``choice`` strings are parsed by ``chat.py``.
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict

from app.form import Traveller, TripForm
from app.tree import LoadedTree

OPENING: Final = "Tell me about your trip: where you're going, why, and who's travelling."
OFF_TOPIC: Final = "I can only help with visa documents."
OFF_TOPIC_START: Final = (
    "I can only help with visa documents. Where are you travelling, and who's going?"
)
NOT_UNDERSTOOD: Final = "Sorry, I didn't catch that."
TOO_LONG: Final = "That message is too long. Please keep it under {limit} characters."
TOO_MANY: Final = "I can check up to {limit} travellers in one chat."
TURN_LIMIT: Final = "This chat has reached its limit. Tap Start over to begin a new one."
ASK_COUNTRY: Final = "Which country are you travelling to?"
ASK_PURPOSE: Final = (
    "Is the trip for tourism, visiting family or friends, or a conference or business meeting?"
)
ASK_WHO: Final = "Who's travelling? For example: just me, or me, my wife and son (8)."
ASK_WHO_MORE: Final = "You mentioned {expected} travellers and I have {have}. Who else is going?"
PROFESSION_HINT: Final = "(e.g. salaried, business, student, retired)"
WHAT_CHANGE: Final = 'What should I change? For example: "mom is going to the UK".'
SUMMARY_HEAD: Final = "Here's the summary:"
SUMMARY_ASK: Final = "Is this correct?"
RESULT: Final = (
    "Here's your document checklist. Please confirm with the embassy or VFS before applying."
)
ALREADY_DONE: Final = "Your checklist is above. Tap Start over to check another trip."
BUTTONS_HINT: Final = "Tap one of the options below, or type your answer."

CHILD_OPTIONS: Final = (
    ("Under 3", "infant"),
    ("School student", "school_student"),
    ("College student", "university_student"),
    ("Working", "employed"),
)


class Button(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    choice: str


START_OVER: Final = Button(label="Start over", choice="restart")
CONFIRM: Final = Button(label="Yes, show my checklist", choice="confirm")


# --------------------------------------------------------------------------- names


def person_name(form: TripForm, index: int, tree: LoadedTree) -> str:
    """'you', 'your mother', 'your son (8)', 'your son #2'."""
    t = form.travellers[index]
    if t.relation == "self":
        return "you"
    label = tree.data.relations[t.relation].label.lower()
    same = [i for i, o in enumerate(form.travellers) if o.relation == t.relation]
    if len(same) > 1:
        label += f" ({t.age})" if t.age is not None else f" #{same.index(index) + 1}"
    return f"your {label}"


def _join(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _people(form: TripForm, indices: tuple[int, ...], tree: LoadedTree) -> tuple[str, bool]:
    """Join names, dropping repeated 'your': 'you and your mother and father'."""
    names = [person_name(form, i, tree) for i in indices]
    you = "you" in names
    others = [n.removeprefix("your ") for n in names if n != "you"]
    parts = (["you"] if you else []) + ([f"your {_join(others)}"] if others else [])
    return " and ".join(parts), len(names) > 1 or you


def ask_profession(form: TripForm, indices: tuple[int, ...], tree: LoadedTree) -> str:
    who, plural = _people(form, indices, tree)
    verb = "do" if plural else "does"
    return f"What {verb} {who} do? {PROFESSION_HINT}"


def ask_age(form: TripForm, indices: tuple[int, ...], tree: LoadedTree) -> str:
    who, plural = _people(form, indices, tree)
    return f"How old {'are' if plural else 'is'} {who}?"


def ask_more(form: TripForm) -> str:
    expected = form.expected_count or 0
    return ASK_WHO_MORE.format(expected=expected, have=len(form.travellers))


# --------------------------------------------------------------------------- summary


def _describe(t: Traveller, form: TripForm, profession: str, tree: LoadedTree) -> str:
    d = tree.data
    bits = [d.professions[profession].label]
    if t.age is not None:
        bits.insert(0, f"age {t.age}")
    country = t.country or form.country
    purpose = t.purpose or form.purpose
    if (country, purpose) != (form.country, form.purpose) and country and purpose:
        bits.append(f"{d.countries[country].name}, {d.purposes[purpose].label.lower()}")
    return ", ".join(bits)


def summary(form: TripForm, professions: list[str], tree: LoadedTree) -> str:
    d = tree.data
    if form.country is None or form.purpose is None:
        raise ValueError("summary needs a complete form")
    lines = [SUMMARY_HEAD, f"• {d.countries[form.country].name}, {d.purposes[form.purpose].label}"]
    for i, t in enumerate(form.travellers):
        name = person_name(form, i, tree)
        lines.append(f"• {name[0].upper()}{name[1:]}: {_describe(t, form, professions[i], tree)}")
    lines.append(SUMMARY_ASK)
    return "\n".join(lines)


# --------------------------------------------------------------------------- buttons


def country_buttons(tree: LoadedTree) -> tuple[Button, ...]:
    return tuple(
        Button(label=c.name, choice=f"country:{cid}") for cid, c in tree.data.countries.items()
    )


def purpose_buttons(tree: LoadedTree) -> tuple[Button, ...]:
    return tuple(
        Button(label=p.label, choice=f"purpose:{pid}") for pid, p in tree.data.purposes.items()
    )


def who_buttons(form: TripForm, tree: LoadedTree) -> tuple[Button, ...]:
    buttons = [] if form.travellers else [Button(label="Just me", choice="solo")]
    buttons += [
        Button(label=f"+ {r.label}", choice=f"add:{rid}")
        for rid, r in tree.data.relations.items()
        if rid != "self" or not any(t.relation == "self" for t in form.travellers)
    ]
    if form.travellers:
        buttons.append(Button(label="That's everyone", choice="everyone"))
    return tuple(buttons)


def profession_buttons(index: int, tree: LoadedTree) -> tuple[Button, ...]:
    return tuple(
        Button(label=p.label, choice=f"prof:{index}:{pid}")
        for pid, p in tree.data.professions.items()
        if pid not in ("infant", "school_student")
    )


def child_buttons(index: int, tree: LoadedTree) -> tuple[Button, ...]:
    return tuple(
        Button(label=label, choice=f"prof:{index}:{pid}")
        for label, pid in CHILD_OPTIONS
        if pid in tree.data.professions
    )
