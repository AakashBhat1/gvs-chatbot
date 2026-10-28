from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from app.form import Traveller, TripForm
from app.resolver import (
    DocItem,
    IncompleteFormError,
    Missing,
    UnknownIdError,
    effective_profession,
    missing_fields,
    resolve,
)
from app.tree import LoadedTree, load_tree


def ids(docs: tuple[DocItem, ...]) -> list[str]:
    return [d.id for d in docs]


def trip(*travellers: Traveller, country: str = "jp", purpose: str = "tourism") -> TripForm:
    return TripForm(country=country, purpose=purpose, travellers=travellers)


SELF_EMPLOYED = Traveller(relation="self", profession="employed")


# ----------------------------------------------------------------- ordering & matching


def test_solo_traveller_gets_general_then_specific_in_order(mini_tree: LoadedTree) -> None:
    out = resolve(trip(SELF_EMPLOYED), mini_tree)
    (group,) = out.groups
    (person,) = group.travellers
    # "*" (least specific) before "jp" even though "jp" comes first in the file.
    assert ids(person.documents) == ["passport", "photos", "jp_form", "salary"]
    assert group.shared == ()
    assert group.country_name == "Japan"
    assert out.tree_version == mini_tree.version


def test_purpose_rule_applies_only_to_that_purpose(mini_tree: LoadedTree) -> None:
    out = resolve(trip(SELF_EMPLOYED, purpose="business"), mini_tree)
    assert "biz_invite" in ids(out.groups[0].travellers[0].documents)
    out = resolve(trip(SELF_EMPLOYED, purpose="tourism"), mini_tree)
    assert "biz_invite" not in ids(out.groups[0].travellers[0].documents)


def test_document_listed_once_per_person(
    mini_dict: dict[str, Any], write_tree: Callable[[dict[str, Any]], Path]
) -> None:
    mini_dict["rules"]["jp/tourism"] = ["passport", "salary"]  # overlaps "*" and employed
    tree = load_tree(write_tree(mini_dict))
    docs = ids(resolve(trip(SELF_EMPLOYED), tree).groups[0].travellers[0].documents)
    assert docs.count("passport") == 1
    assert docs.count("salary") == 1


# ----------------------------------------------------------------- family & groups


def test_family_items_shared_once_and_removed_from_people(mini_tree: LoadedTree) -> None:
    form = trip(SELF_EMPLOYED, Traveller(relation="mother", profession="lawyer"))
    (group,) = resolve(form, mini_tree).groups
    assert ids(group.shared) == ["cover"]
    for person in group.travellers:
        assert "cover" not in ids(person.documents)
    assert ids(group.travellers[1].documents) == ["passport", "photos", "jp_form", "bar_id"]


def test_member_going_elsewhere_forms_own_group(mini_tree: LoadedTree) -> None:
    form = trip(
        SELF_EMPLOYED,
        Traveller(relation="son", age=10),
        Traveller(relation="mother", profession="lawyer", country="uk"),
    )
    jp, uk = resolve(form, mini_tree).groups
    assert (jp.country, uk.country) == ("jp", "uk")
    assert [p.index for p in jp.travellers] == [0, 1]
    assert [p.index for p in uk.travellers] == [2]
    assert ids(jp.shared) == ["cover"]
    assert uk.shared == ()  # a group of one has no shared covering letter
    assert "jp_form" not in ids(uk.travellers[0].documents)


# ----------------------------------------------------------------- age handling


@pytest.mark.parametrize(
    ("age", "country", "expected"),
    [
        (0, "jp", "infant"),
        (2, "jp", "infant"),
        (3, "jp", "school_student"),
        (17, "jp", "school_student"),
        (18, "jp", "employed"),  # adult: reported profession is used
        (16, "uk", "employed"),  # uk override: minors only up to 15
        (15, "uk", "school_student"),
    ],
)
def test_age_bands(mini_tree: LoadedTree, age: int, country: str, expected: str) -> None:
    t = Traveller(relation="son", profession="employed", age=age)
    assert effective_profession(t, country, mini_tree) == expected


def test_child_gets_relation_group_docs(mini_tree: LoadedTree) -> None:
    form = trip(SELF_EMPLOYED, Traveller(relation="daughter", age=1))
    child = resolve(form, mini_tree).groups[0].travellers[1]
    assert child.profession == "infant"
    assert child.age == 1
    assert "birth_cert" in ids(child.documents)
    adult = resolve(trip(SELF_EMPLOYED), mini_tree).groups[0].travellers[0]
    assert "birth_cert" not in ids(adult.documents)


# ----------------------------------------------------------------- incomplete / invalid


def test_missing_fields_in_ask_order(mini_tree: LoadedTree) -> None:
    assert missing_fields(TripForm(), mini_tree) == [
        Missing("country"),
        Missing("purpose"),
        Missing("travellers"),
    ]
    form = trip(
        Traveller(relation="self"),
        Traveller(relation="son", age=3),  # age fills the gap
        Traveller(relation="mother", age=60),  # adult age does not
    )
    assert missing_fields(form, mini_tree) == [Missing("profession", 0), Missing("profession", 2)]


def test_incomplete_form_raises_with_gaps(mini_tree: LoadedTree) -> None:
    with pytest.raises(IncompleteFormError) as exc:
        resolve(TripForm(country="jp"), mini_tree)
    assert [m.field for m in exc.value.missing] == ["purpose", "travellers"]


def test_unknown_ids_rejected_before_resolving(mini_tree: LoadedTree) -> None:
    form = trip(Traveller(relation="uncle", profession="astronaut"), country="mars")
    with pytest.raises(UnknownIdError) as exc:
        resolve(form, mini_tree)
    assert len(exc.value.problems) == 3


def test_form_rejects_more_than_eight_travellers() -> None:
    with pytest.raises(ValueError, match="at most 8"):
        TripForm(travellers=tuple(SELF_EMPLOYED for _ in range(9)))


@pytest.mark.parametrize("bad", [{"age": -1}, {"age": 111}, {"age": "5"}, {"nickname": "x"}])
def test_traveller_rejects_bad_values(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Traveller.model_validate({"relation": "self", **bad})


# ----------------------------------------------------------------- real data file


def test_real_tree_family_of_four(real_tree: LoadedTree) -> None:
    form = TripForm(
        country="japan",
        purpose="tourism",
        travellers=(
            Traveller(relation="self", profession="employed"),
            Traveller(relation="spouse", profession="homemaker"),
            Traveller(relation="son", age=8),
            Traveller(relation="daughter", age=1),
        ),
    )
    (group,) = resolve(form, real_tree).groups
    assert ids(group.shared) == ["family_covering_letter"]
    professions = [p.profession for p in group.travellers]
    assert professions == ["employed", "homemaker", "school_student", "infant"]
    for person in group.travellers:
        docs = ids(person.documents)
        assert docs[:4] == ["passport", "photos", "application_form", "itinerary"]
        assert len(docs) == len(set(docs))
