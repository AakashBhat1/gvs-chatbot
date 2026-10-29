"""What a brain may return, and the wall that checks it.

Every brain (rules, local model, Haiku) produces an :class:`Extraction`. Its
shape is fixed and every value is an id or a bounded number, so whatever a brain
returns, :func:`sanitize` drops anything not in ``visa-tree.json`` before the
form is touched. Nothing a brain returns is ever shown to the user as text.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from app.form import MAX_TRAVELLERS, Age, TripForm
from app.tree import Id, LoadedTree

Intent = Literal["give_details", "confirm", "correct", "restart"]
Stage = Literal["collecting", "confirming", "done"]
AskedField = Literal[
    "open", "country", "purpose", "travellers", "group_size", "profession", "age", "confirm"
]

MAX_COUNT = 20  # "family of 12" must be reportable, so the bot can say it handles 8 per chat
Index = Annotated[StrictInt, Field(ge=0, lt=MAX_TRAVELLERS)]
Count = Annotated[StrictInt, Field(ge=1, le=MAX_COUNT)]


class TravellerPatch(BaseModel):
    """One traveller mentioned in a message: a new person, or ``ref`` to an existing one."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ref: Index | None = None
    relation: Id | None = None
    profession: Id | None = None
    age: Age | None = None
    country: Id | None = None
    purpose: Id | None = None
    remove: StrictBool = False


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    on_topic: StrictBool = True
    intent: Intent = "give_details"
    country: Id | None = None
    purpose: Id | None = None
    traveller_count: Count | None = None
    travellers: tuple[TravellerPatch, ...] = Field(default=(), max_length=MAX_TRAVELLERS)
    # A country the site lists but the bot has no checklist for ("Sri Lanka").
    other_country: Id | None = None
    # "with my family": more people are coming who weren't named yet.
    others_coming: StrictBool = False
    # "with my friends": people of this relation are coming, but not how many.
    group_of: Id | None = None
    # "tourism and a conference": the user must pick one main purpose.
    purpose_choices: tuple[Id, ...] = Field(default=(), max_length=3)
    # Asked about fees, timelines, appointments: only the office can answer.
    asks_office: StrictBool = False
    # Asked which countries are covered / what the bot can do.
    asks_help: StrictBool = False

    @property
    def has_details(self) -> bool:
        return bool(
            self.country
            or self.purpose
            or self.traveller_count
            or self.travellers
            or self.others_coming
            or self.group_of
            or self.purpose_choices
        )


@dataclass(frozen=True, slots=True)
class Asked:
    """The question the bot last asked, so a bare answer ("retired", "8") lands correctly."""

    field: AskedField
    travellers: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Context:
    form: TripForm
    asked: Asked | None
    stage: Stage


def sanitize(ex: Extraction, tree: LoadedTree, n_travellers: int) -> Extraction:
    """Drop every value that is not a known id or a valid reference."""
    d = tree.data

    def keep(value: str | None, table: Mapping[str, object]) -> str | None:
        return value if value is not None and value in table else None

    patches: list[TravellerPatch] = []
    for p in ex.travellers:
        ref = p.ref if p.ref is not None and p.ref < n_travellers else None
        relation = keep(p.relation, d.relations)
        if ref is None and relation is None:
            continue  # nobody to attach the details to
        patches.append(
            TravellerPatch(
                ref=ref,
                relation=relation,
                profession=keep(p.profession, d.professions),
                age=p.age,
                country=keep(p.country, d.countries),
                purpose=keep(p.purpose, d.purposes),
                remove=p.remove,
            )
        )
    return Extraction(
        on_topic=ex.on_topic,
        intent=ex.intent,
        country=keep(ex.country, d.countries),
        purpose=keep(ex.purpose, d.purposes),
        traveller_count=ex.traveller_count,
        travellers=tuple(patches),
        other_country=keep(ex.other_country, d.other_countries),
        others_coming=ex.others_coming,
        group_of=keep(ex.group_of, d.relations) if ex.group_of != "self" else None,
        purpose_choices=tuple(p for p in dict.fromkeys(ex.purpose_choices) if p in d.purposes),
        asks_office=ex.asks_office,
        asks_help=ex.asks_help,
    )
