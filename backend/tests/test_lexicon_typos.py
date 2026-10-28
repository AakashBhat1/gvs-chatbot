from __future__ import annotations

import json
from typing import Any

import pytest

from app.brains.lexicon import Lexicon, edit_distance, load_lexicon, parse_lexicon
from app.evaluate import Case, Expect, Outcome
from app.form import Traveller, TripForm
from app.paths import default_lexicon_path
from app.tree import LoadedTree, TreeError, parse_tree


@pytest.fixture
def lex(real_tree: LoadedTree) -> Lexicon:
    return load_lexicon(default_lexicon_path(), real_tree)


@pytest.mark.parametrize(
    ("a", "b", "limit", "expected"),
    [
        ("japan", "japan", 1, 0),
        ("japn", "japan", 1, 1),
        ("thialand", "thailand", 1, 1),  # swapped letters count as one typo
        ("abc", "xyz", 1, 2),  # capped at limit + 1
        ("a", "abcdef", 2, 3),
    ],
)
def test_edit_distance(a: str, b: str, limit: int, expected: int) -> None:
    assert edit_distance(a, b, limit) == expected


@pytest.mark.parametrize(
    ("word", "hit"),
    [
        ("japn", ("country", "japan")),
        ("austrailia", ("country", "australia")),
        ("newzealand", ("country", "new-zealand")),
        ("unitedstates", ("country", "us")),
        ("tourisum", ("purpose", "tourism")),
        ("retierd", ("profession", "retired")),
        ("bussiness", ("business_word", "business_word")),
        ("srilankaa", ("other_country", "sri-lanka")),
    ],
)
def test_guess_corrects_typos(lex: Lexicon, word: str, hit: tuple[str, str] | None) -> None:
    assert lex.guess(word) == hit


@pytest.mark.parametrize("word", ["male", "mail", "india", "delhi", "mumbai", "abc", "x1y2"])
def test_guess_leaves_real_words_alone(lex: Lexicon, word: str) -> None:
    assert lex.guess(word) is None


def test_guess_refuses_ties(real_tree: LoadedTree) -> None:
    toml = '[countries]\njapan = ["abcdef"]\nuk = ["abcdeg"]\n'
    tie = parse_lexicon(toml.encode(), real_tree)
    assert tie.guess("abcdex") is None  # one typo from two different countries


def test_compact_clash_is_dropped(real_tree: LoadedTree) -> None:
    toml = '[countries]\njapan = ["new zea land"]\nuk = ["newzea land"]\n'
    lex = parse_lexicon(toml.encode(), real_tree)
    assert lex.guess("newzealand") is None


def test_other_country_alias_must_exist(real_tree: LoadedTree) -> None:
    with pytest.raises(ValueError, match="not an id"):
        parse_lexicon(b'[other_countries]\natlantis = ["atlantis"]\n', real_tree)


def test_tree_rejects_overlapping_country_lists(mini_dict: dict[str, Any]) -> None:
    mini_dict["other_countries"] = {"jp": {"name": "Japan again"}}
    with pytest.raises(TreeError, match="already a supported country"):
        parse_tree(json.dumps(mini_dict).encode())


@pytest.mark.parametrize(
    "url", ["http://plain.example", "https://evil.example/path", "javascript:x"]
)
def test_tree_rejects_bad_site_url(mini_dict: dict[str, Any], url: str) -> None:
    mini_dict["meta"]["site_url"] = url
    with pytest.raises(TreeError):
        parse_tree(json.dumps(mini_dict).encode())


def test_eval_ignores_traveller_order() -> None:
    a, b = Traveller(relation="self"), Traveller(relation="spouse")
    case = Case(id="x", message="m", expect=Expect(form=TripForm(travellers=(a, b))))
    outcome = Outcome(case, TripForm(travellers=(b, a)), "give_details", True, True)
    assert outcome.matches("travellers")
