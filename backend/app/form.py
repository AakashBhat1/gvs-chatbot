"""The per-chat trip form.

This is server-side state. It never holds names, passport numbers or dates of
birth — only ids from ``visa-tree.json`` and a bounded age. Ids are checked for
*shape* here and for *existence* against the live tree by :func:`unknown_ids`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from app.tree import Id, LoadedTree

MAX_TRAVELLERS: Final = 8

Age = Annotated[StrictInt, Field(ge=0, le=110)]


class Traveller(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relation: Id
    profession: Id | None = None
    age: Age | None = None
    country: Id | None = None  # only when different from the lead's trip
    purpose: Id | None = None  # only when different from the lead's trip


class TripForm(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    country: Id | None = None
    purpose: Id | None = None
    travellers: tuple[Traveller, ...] = Field(default=(), max_length=MAX_TRAVELLERS)
    # "family of four" before all four are named; never below len(travellers).
    expected_count: Annotated[StrictInt, Field(ge=1, le=MAX_TRAVELLERS)] | None = None

    def country_of(self, t: Traveller) -> str | None:
        return t.country or self.country

    def purpose_of(self, t: Traveller) -> str | None:
        return t.purpose or self.purpose


def unknown_ids(form: TripForm, tree: LoadedTree) -> list[str]:
    """Return a description of every id in ``form`` that the tree does not define."""
    data = tree.data
    problems: list[str] = []

    def check(value: str | None, table: Mapping[str, object], where: str) -> None:
        if value is not None and value not in table:
            problems.append(f"{where}: unknown id {value!r}")

    check(form.country, data.countries, "country")
    check(form.purpose, data.purposes, "purpose")
    for i, t in enumerate(form.travellers):
        check(t.relation, data.relations, f"travellers[{i}].relation")
        check(t.profession, data.professions, f"travellers[{i}].profession")
        check(t.country, data.countries, f"travellers[{i}].country")
        check(t.purpose, data.purposes, f"travellers[{i}].purpose")
    return problems
