"""Every sentence the bot can say.

Replies are built only from these templates and labels in ``visa-tree.json``.
No brain ever writes user-visible text, so a prompt injection cannot make the
bot say anything new. Button ``choice`` strings are parsed by ``chat.py``.
"""

from __future__ import annotations

from collections import Counter
from typing import Final

from pydantic import BaseModel, ConfigDict

from app.form import Traveller, TripForm
from app.resolver import Checklist
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
ASK_WHO_WITH: Final = "Who else is travelling with you? For example: my wife and son (8)."
ASK_MAIN_PURPOSE: Final = "A visa is applied for one main purpose. Which is it: {options}?"
HELP: Final = "I can make a document checklist for {countries}."
HELP_MORE: Final = " Other countries are on our website: {site}/visa"
UNSUPPORTED: Final = "I don't have a checklist for {name} yet."
UNSUPPORTED_LINK: Final = (
    " Its visa page is {site}/visa/{slug}, or contact our office: {site}/contact."
)
UNSUPPORTED_NO_LINK: Final = " Please contact our office for {name}."
OFFICE: Final = "Fees, processing times and appointments are handled by our office"
OFFICE_LINK: Final = ": {site}/contact. I can still list the documents you need."
OFFICE_NO_LINK: Final = ". I can still list the documents you need."
COUNTRY_PAGE: Final = "More about {name}: {site}/visa/{slug}"
PROFESSION_HINT: Final = "(e.g. salaried, business, student, retired)"
WHAT_CHANGE: Final = 'What should I change? For example: "mom is going to the UK".'
SUMMARY_HEAD: Final = "Here's the summary:"
SUMMARY_ASK: Final = "Is this correct?"
RESULT: Final = (
    "Here's your document checklist. Please confirm with the embassy or VFS before applying."
)
ALREADY_DONE: Final = "Your checklist is above. Tap Start over to check another trip."
BUTTONS_HINT: Final = "Tap one of the options below, or type your answer."
HINDI_SCRIPT: Final = (
    "For now I can read English and Hinglish (Hindi in English letters), "
    'like "Japan ghumne jaana hai".'
)

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


PLURALS: Final = {"child": "children"}


def _people(form: TripForm, indices: tuple[int, ...], tree: LoadedTree) -> tuple[str, bool]:
    """'you and your mother and father', 'your 2 children'. Returns (phrase, is_plural)."""
    people = [form.travellers[i] for i in indices]
    you = any(t.relation == "self" for t in people)
    counts = Counter(t.relation for t in people if t.relation != "self")
    others: list[str] = []
    for i, t in zip(indices, people, strict=True):
        if t.relation == "self" or t.relation not in counts:
            continue
        n = counts.pop(t.relation)  # each relation is named once, at its first position
        if n > 1:
            label = tree.data.relations[t.relation].label.lower()
            others.append(f"{n} {PLURALS.get(t.relation, label + 's')}")
        else:
            others.append(person_name(form, i, tree).removeprefix("your "))
    parts = (["you"] if you else []) + ([f"your {_join(others)}"] if others else [])
    return " and ".join(parts), len(indices) > 1 or you


def ask_profession(form: TripForm, indices: tuple[int, ...], tree: LoadedTree) -> str:
    who, plural = _people(form, indices, tree)
    verb = "do" if plural else "does"
    return f"What {verb} {who} do? {PROFESSION_HINT}"


def ask_age(form: TripForm, indices: tuple[int, ...], tree: LoadedTree) -> str:
    who, plural = _people(form, indices, tree)
    return f"How old {'are' if plural else 'is'} {who}?"


def ask_more(form: TripForm) -> str:
    if form.expected_count is None:
        return ASK_WHO_WITH
    return ASK_WHO_MORE.format(expected=form.expected_count, have=len(form.travellers))


def ask_main_purpose(choices: tuple[str, ...], tree: LoadedTree) -> str:
    labels = [tree.data.purposes[c].label for c in choices]
    return ASK_MAIN_PURPOSE.format(options=", or ".join(labels))


def help_text(tree: LoadedTree) -> str:
    text = HELP.format(countries=_join([c.name for c in tree.data.countries.values()]))
    site = tree.data.meta.site_url
    return text + (HELP_MORE.format(site=site) if site else "")


def unsupported(name: str, slug: str, site: str | None) -> str:
    tail = (
        UNSUPPORTED_LINK.format(site=site, slug=slug)
        if site
        else UNSUPPORTED_NO_LINK.format(name=name)
    )
    return UNSUPPORTED.format(name=name) + tail


def office(site: str | None) -> str:
    return OFFICE + (OFFICE_LINK.format(site=site) if site else OFFICE_NO_LINK)


def result(checklist: Checklist, site: str | None) -> str:
    lines = [RESULT]
    if site:
        lines += [
            COUNTRY_PAGE.format(name=g.country_name, site=site, slug=g.country)
            for g in checklist.groups
        ]
    return "\n".join(lines)


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
        described = _describe(t, form, professions[i], tree)
        if t.age is not None and f"({t.age})" in name:  # "your child (7)": don't repeat the age
            described = described.removeprefix(f"age {t.age}, ")
        lines.append(f"• {name[0].upper()}{name[1:]}: {described}")
    lines.append(SUMMARY_ASK)
    return "\n".join(lines)


# --------------------------------------------------------------------------- buttons


def country_buttons(tree: LoadedTree) -> tuple[Button, ...]:
    return tuple(
        Button(label=c.name, choice=f"country:{cid}") for cid, c in tree.data.countries.items()
    )


def purpose_buttons(tree: LoadedTree, only: tuple[str, ...] = ()) -> tuple[Button, ...]:
    return tuple(
        Button(label=p.label, choice=f"purpose:{pid}")
        for pid, p in tree.data.purposes.items()
        if not only or pid in only
    )


def who_buttons(form: TripForm, tree: LoadedTree) -> tuple[Button, ...]:
    buttons = [] if form.travellers else [Button(label="Just me", choice="solo")]
    buttons += [
        Button(label="+ Me" if rid == "self" else f"+ {r.label}", choice=f"add:{rid}")
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
