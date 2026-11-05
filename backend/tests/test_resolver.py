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


# ----------------------------------------------------------------- own checklists


@pytest.fixture
def own_tree(mini_dict: dict[str, Any], write_tree: Callable[[dict[str, Any]], Path]) -> LoadedTree:
    """Japan owns a checklist that the UK shares; a third country stays generic."""
    mini_dict["countries"]["jp"]["fee"] = "Fee: 10"
    mini_dict["countries"]["jp"]["notes"] = ["Originals only"]
    mini_dict["countries"]["us"] = {"name": "USA"}
    mini_dict["own_checklists"] = {"jp": ["uk"]}
    mini_dict["documents"]["jp_salary"] = {"label": "Japan salary proof", "note": "Attested"}
    mini_dict["rules"]["jp/*/employed"] = ["jp_salary"]
    return load_tree(write_tree(mini_dict))


@pytest.mark.parametrize("country", ["jp", "uk"])
def test_own_checklist_replaces_generic_rules(own_tree: LoadedTree, country: str) -> None:
    (group,) = resolve(trip(SELF_EMPLOYED, country=country), own_tree).groups
    assert ids(group.travellers[0].documents) == ["jp_form", "jp_salary"]
    assert group.notes == ("Fee: 10", "Originals only")
    assert own_tree.fee_for(country) == "Fee: 10"
    assert group.travellers[0].documents[1].note == "Attested"


def test_own_checklist_has_no_relation_or_family_documents(own_tree: LoadedTree) -> None:
    form = trip(SELF_EMPLOYED, Traveller(relation="son", age=10))
    (group,) = resolve(form, own_tree).groups
    assert group.shared == ()
    assert ids(group.travellers[1].documents) == ["jp_form"]


def test_other_countries_keep_generic_rules(own_tree: LoadedTree) -> None:
    form = trip(SELF_EMPLOYED, Traveller(relation="son", age=10), country="us")
    (group,) = resolve(form, own_tree).groups
    assert ids(group.shared) == ["cover"]
    assert ids(group.travellers[0].documents) == ["passport", "photos", "salary"]
    assert group.notes == ()
    assert own_tree.fee_for("us") is None


# ----------------------------------------------------------------- Schengen list


COMMON = [
    "sch_passport",
    "sch_photos",
    "application_form",
    "sch_insurance",
    "sch_flight",
    "sch_accommodation",
    "sch_aadhaar",
    "sch_passport_copy",
    "sch_lost_passport",
]
FINANCES = ["sch_bank_statements", "sch_investments", "sch_itr"]
OWN_FINANCES = ["sch_own_finances_if_any", "sch_investments"]
SPONSOR = ["sch_sponsor_letter", "sch_sponsor_occupation"]
DEPENDENT = [
    *SPONSOR,
    "sch_sponsor_bank_fd",
    "sch_sponsor_itr",
    "sch_sponsor_id",
    "sch_relationship_proof",
]
STUDENT = [
    "sch_birth_certificate",
    "sch_bonafide",
    "sch_student_id",
    *SPONSOR,
    "sch_sponsor_bank",
    "sch_sponsor_itr",
    "sch_sponsor_passports",
    "sch_parent_noc",
    "sch_parent_passports",
]
EMPLOYED = [
    *FINANCES,
    "sch_employment_contract",
    "sch_intro_letter",
    "sch_salary_slips",
    "sch_salary_account",
]
BUSINESS = [
    *FINANCES,
    "sch_intro_letter",
    "sch_company_registration",
    "sch_salary_slips_or_certificate",
    "sch_company_bank",
    "sch_company_itr",
]
ASK_OFFICE = ["sch_occupation_ask_office"]
SCHENGEN_SAMPLE = ["schengen", "france", "switzerland", "austria", "netherlands", "czech-republic"]


@pytest.mark.parametrize(
    ("profession", "extra"),
    [
        ("employed", EMPLOYED),
        ("government_employee", EMPLOYED),
        ("defence_personnel", EMPLOYED),
        ("business_owner", BUSINESS),
        ("company_director", BUSINESS),
        (
            "self_employed_professional",
            [
                *FINANCES,
                "sch_professional_registration",
                "sch_professional_letter",
                "sch_entity_registration",
                "sch_degree",
            ],
        ),
        ("farmer", [*FINANCES, "sch_land_documents", "sch_j_form"]),
        (
            "seafarer",
            [
                *FINANCES,
                "sch_offshore_contract",
                "sch_intro_letter",
                "sch_salary_slips_or_invoices",
            ],
        ),
        (
            "retired",
            [
                *FINANCES,
                "sch_pension_order",
                "sch_pension_slips",
                "sch_pension_bank",
                "sch_property_or_business_income",
            ],
        ),
        ("homemaker", [*OWN_FINANCES, *DEPENDENT]),
        ("unemployed", [*OWN_FINANCES, *DEPENDENT]),
        ("infant", ["sch_birth_certificate", *DEPENDENT, "sch_parent_noc"]),
        ("university_student", [*OWN_FINANCES, *STUDENT]),
        ("school_student", STUDENT),
        ("freelancer", [*FINANCES, *ASK_OFFICE]),
        ("landlord_investor", [*FINANCES, "sch_property_or_business_income", *ASK_OFFICE]),
        ("other", [*FINANCES, *ASK_OFFICE]),
    ],
)
def test_schengen_lists_by_profession(
    real_tree: LoadedTree, profession: str, extra: list[str]
) -> None:
    form = trip(Traveller(relation="self", profession=profession), country="schengen")
    person = resolve(form, real_tree).groups[0].travellers[0]
    assert ids(person.documents) == [*COMMON, *extra]


@pytest.mark.parametrize("country", SCHENGEN_SAMPLE)
def test_schengen_tourism_only_placeholder_is_the_application_form(
    real_tree: LoadedTree, country: str
) -> None:
    form = trip(
        SELF_EMPLOYED,
        Traveller(relation="spouse", profession="homemaker"),
        Traveller(relation="son", age=8),
        Traveller(relation="daughter", age=1),
        country=country,
    )
    (group,) = resolve(form, real_tree).groups
    assert group.shared == ()
    assert any("EUR 90" in note for note in group.notes)
    for person in group.travellers:
        placeholders = [d.id for d in person.documents if "[PLACEHOLDER]" in d.label]
        assert placeholders == ["application_form"]


def test_schengen_children_are_not_asked_for_their_own_finances(real_tree: LoadedTree) -> None:
    form = trip(
        SELF_EMPLOYED,
        Traveller(relation="son", age=8),
        Traveller(relation="daughter", age=1),
        country="france",
    )
    (_, son, daughter) = resolve(form, real_tree).groups[0].travellers
    for child in (son, daughter):
        docs = ids(child.documents)
        assert not set(docs) & {*FINANCES, "sch_own_finances_if_any"}
        assert {"sch_birth_certificate", "sch_parent_noc"} <= set(docs)


@pytest.mark.parametrize(
    ("purpose", "kept"),
    [
        ("family_visit", ["invitation_letter", "host_id_proof"]),
        ("business", ["business_invitation", "company_covering_letter"]),
    ],
)
def test_schengen_other_purposes_keep_their_placeholders(
    real_tree: LoadedTree, purpose: str, kept: list[str]
) -> None:
    form = trip(SELF_EMPLOYED, country="schengen", purpose=purpose)
    docs = ids(resolve(form, real_tree).groups[0].travellers[0].documents)
    assert docs[: len(COMMON)] == COMMON
    assert docs[-2:] == kept


def test_other_real_countries_do_not_get_the_schengen_list(real_tree: LoadedTree) -> None:
    form = trip(SELF_EMPLOYED, country="japan")
    (group,) = resolve(form, real_tree).groups
    assert group.notes == ()
    assert not [d for d in ids(group.travellers[0].documents) if d.startswith("sch_")]


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
