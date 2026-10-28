"""Turn a complete :class:`TripForm` into per-person document checklists.

Pure functions over an immutable :class:`LoadedTree` — no I/O, no model calls.

Ordering and de-duplication
    Each person's list is built from matching path rules (least specific first,
    then file order) followed by their relation rules (file order). A document
    id appears at most once per person, at its first position.

Shared (``@family``) items
    Travellers are grouped by their effective (country, purpose). A group of
    two or more gets the ``@family`` documents once, and those documents are
    then removed from each member's personal list.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.form import Traveller, TripForm, unknown_ids
from app.tree import WILDCARD, LoadedTree

MissingField = Literal["country", "purpose", "travellers", "profession"]


@dataclass(frozen=True, slots=True)
class Missing:
    field: MissingField
    traveller: int | None = None  # index into form.travellers for per-person gaps


class FormError(ValueError):
    pass


class UnknownIdError(FormError):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


class IncompleteFormError(FormError):
    def __init__(self, missing: list[Missing]) -> None:
        super().__init__(f"form incomplete: {[m.field for m in missing]}")
        self.missing = missing


# --------------------------------------------------------------------------- output


class _Out(BaseModel):
    model_config = ConfigDict(frozen=True)


class DocItem(_Out):
    id: str
    label: str
    note: str | None = None


class PersonChecklist(_Out):
    index: int
    relation: str
    relation_label: str
    profession: str
    profession_label: str
    age: int | None
    documents: tuple[DocItem, ...]


class GroupChecklist(_Out):
    country: str
    country_name: str
    purpose: str
    purpose_label: str
    last_reviewed: date | None
    shared: tuple[DocItem, ...]
    travellers: tuple[PersonChecklist, ...]


class Checklist(_Out):
    tree_version: str
    tree_status: str
    groups: tuple[GroupChecklist, ...]


# --------------------------------------------------------------------------- logic


def effective_profession(t: Traveller, country: str | None, tree: LoadedTree) -> str | None:
    """Children are classified by age in code, never by the model.

    If the age falls in an age band, the band's profession wins over anything
    reported; otherwise the reported profession is used.
    """
    if t.age is not None:
        for band in tree.age_bands_for(country or WILDCARD):
            if t.age <= band.max_age:
                return band.profession
    return t.profession


def missing_fields(form: TripForm, tree: LoadedTree) -> list[Missing]:
    """Everything still needed, in the order the bot should ask for it."""
    missing: list[Missing] = []
    if form.country is None:
        missing.append(Missing("country"))
    if form.purpose is None:
        missing.append(Missing("purpose"))
    if not form.travellers or len(form.travellers) < (form.expected_count or 0):
        missing.append(Missing("travellers"))
    missing += [
        Missing("profession", i)
        for i, t in enumerate(form.travellers)
        if effective_profession(t, form.country_of(t), tree) is None
    ]
    return missing


def _docs_for(
    tree: LoadedTree, country: str, purpose: str, profession: str, relation: str
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    def add(ids: Iterable[str]) -> None:
        for doc_id in ids:
            if doc_id not in seen:
                seen.add(doc_id)
                ordered.append(doc_id)

    for rule in tree.path_rules:
        if rule.matches(country, purpose, profession):
            add(rule.documents)
    for rel, docs in tree.relation_rules:
        if rel == relation:
            add(docs)
    return ordered


def resolve(form: TripForm, tree: LoadedTree) -> Checklist:
    """Build the checklist. Raises :class:`UnknownIdError` or :class:`IncompleteFormError`."""
    if problems := unknown_ids(form, tree):
        raise UnknownIdError(problems)
    if missing := missing_fields(form, tree):
        raise IncompleteFormError(missing)

    data = tree.data

    def item(doc_id: str) -> DocItem:
        doc = data.documents[doc_id]
        return DocItem(id=doc_id, label=doc.label, note=doc.note)

    # Group by effective (country, purpose), keeping the lead's group first.
    groups: dict[tuple[str, str], list[tuple[int, Traveller, str]]] = {}
    for i, t in enumerate(form.travellers):
        country, purpose = form.country_of(t), form.purpose_of(t)
        profession = effective_profession(t, country, tree)
        if country is None or purpose is None or profession is None:  # pragma: no cover
            raise AssertionError("unreachable: missing_fields() returned no gaps")
        groups.setdefault((country, purpose), []).append((i, t, profession))

    out: list[GroupChecklist] = []
    for (country, purpose), members in groups.items():
        shared_ids = tree.family_documents if len(members) > 1 else ()
        shared = set(shared_ids)
        people = tuple(
            PersonChecklist(
                index=i,
                relation=t.relation,
                relation_label=data.relations[t.relation].label,
                profession=profession,
                profession_label=data.professions[profession].label,
                age=t.age,
                documents=tuple(
                    item(d)
                    for d in _docs_for(tree, country, purpose, profession, t.relation)
                    if d not in shared
                ),
            )
            for i, t, profession in members
        )
        out.append(
            GroupChecklist(
                country=country,
                country_name=data.countries[country].name,
                purpose=purpose,
                purpose_label=data.purposes[purpose].label,
                last_reviewed=data.countries[country].last_reviewed,
                shared=tuple(item(d) for d in shared_ids),
                travellers=people,
            )
        )
    return Checklist(tree_version=tree.version, tree_status=data.meta.status, groups=tuple(out))
