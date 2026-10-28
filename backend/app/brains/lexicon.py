"""Load and validate ``lexicon.toml`` — the vocabulary of the rules brain."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from app.text import tokenize
from app.tree import LoadedTree

MAX_LEXICON_BYTES: Final = 262_144
MAX_PHRASE_TOKENS: Final = 5

VALUE_SECTIONS: Final = ("countries", "purposes", "professions", "relations")
MARKERS: Final = frozenset(
    {
        "we",
        "couple",
        "parents",
        "two_of_us",
        "solo",
        "plural",
        "count_noun",
        "count_of",
        "age_unit",
        "month_unit",
        "age_word",
        "confirm",
        "deny",
        "correct_cue",
        "restart",
        "remove",
        "greeting",
        "business_word",
        "pronoun",
        "conj",
        "weak_number",
        "off_topic",
    }
)
# Section name -> the kind of span it produces.
_KIND: Final = {
    "countries": "country",
    "purposes": "purpose",
    "professions": "profession",
    "relations": "relation",
}


class LexiconError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Lexicon:
    phrases: Mapping[tuple[str, ...], tuple[str, str]]  # tokens -> (kind, value)
    numbers: Mapping[str, int]
    weak_numbers: frozenset[str]
    ask_age_for: frozenset[str]
    plural_relations: frozenset[str]
    max_phrase: int

    def lookup(self, tokens: tuple[str, ...]) -> tuple[str, str] | None:
        return self.phrases.get(tokens)


def _str_list(value: object, where: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise LexiconError(f"{where}: expected a list of non-empty strings")
    return [v.strip().lower() for v in value]


def parse_lexicon(raw: bytes, tree: LoadedTree) -> Lexicon:
    if len(raw) > MAX_LEXICON_BYTES:
        raise LexiconError(f"lexicon is {len(raw)} bytes; limit is {MAX_LEXICON_BYTES}")
    try:
        doc: dict[str, Any] = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise LexiconError(str(exc)) from exc

    known = {"settings", "fillers", "numbers", "markers", *VALUE_SECTIONS}
    if unknown := set(doc) - known:
        raise LexiconError(f"unknown sections: {sorted(unknown)}")

    data = tree.data
    tables: dict[str, Mapping[str, object]] = {
        "countries": data.countries,
        "purposes": data.purposes,
        "professions": data.professions,
        "relations": data.relations,
    }
    errors: list[str] = []
    phrases: dict[tuple[str, ...], tuple[str, str]] = {}
    origin: dict[tuple[str, ...], str] = {}

    def add(alias: str, kind: str, value: str, where: str) -> None:
        tokens = tuple(tokenize(alias))
        if not tokens or len(tokens) > MAX_PHRASE_TOKENS:
            errors.append(f"{where}: alias {alias!r} must be 1-{MAX_PHRASE_TOKENS} words")
            return
        if tokens in phrases:
            errors.append(f"{where}: alias {alias!r} already used in {origin[tokens]}")
            return
        phrases[tokens] = (kind, value)
        origin[tokens] = where

    for section in VALUE_SECTIONS:
        for key, aliases in doc.get(section, {}).items():
            if key not in tables[section]:
                errors.append(f"{section}.{key}: not an id in visa-tree.json")
                continue
            for alias in _str_list(aliases, f"{section}.{key}"):
                add(alias, _KIND[section], key, f"{section}.{key}")

    markers = doc.get("markers", {})
    if unknown_markers := set(markers) - MARKERS:
        errors.append(f"markers: unknown roles {sorted(unknown_markers)}")
    weak = frozenset(_str_list(markers.get("weak_number", []), "markers.weak_number"))
    for role, words in markers.items():
        if role in MARKERS and role != "weak_number":
            for alias in _str_list(words, f"markers.{role}"):
                add(alias, role, role, f"markers.{role}")
    for alias in _str_list(doc.get("fillers", {}).get("words", []), "fillers.words"):
        add(alias, "filler", "filler", "fillers")

    numbers: dict[str, int] = {}
    for word, value in doc.get("numbers", {}).items():
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 110:
            errors.append(f"numbers.{word}: must be a whole number 0-110")
        elif (word,) in phrases:
            errors.append(f"numbers.{word}: also used as an alias in {origin[(word,)]}")
        else:
            numbers[word] = value
    errors += [f"markers.weak_number: {w!r} is not in [numbers]" for w in weak if w not in numbers]

    settings = doc.get("settings", {})
    if unknown_settings := set(settings) - {"ask_age_for", "plural_relations"}:
        errors.append(f"settings: unknown keys {sorted(unknown_settings)}")
    ask_age_for = frozenset(_str_list(settings.get("ask_age_for", []), "settings.ask_age_for"))
    errors += [
        f"settings.ask_age_for: unknown relation {r!r}"
        for r in sorted(ask_age_for - set(data.relations))
    ]
    plural_relations = frozenset(
        _str_list(settings.get("plural_relations", []), "settings.plural_relations")
    )
    errors += [
        f"settings.plural_relations: {w!r} is not a relation alias"
        for w in sorted(plural_relations)
        if phrases.get(tuple(tokenize(w)), ("", ""))[0] != "relation"
    ]

    if errors:
        raise LexiconError("; ".join(errors))
    return Lexicon(
        phrases=phrases,
        numbers=numbers,
        weak_numbers=weak,
        ask_age_for=ask_age_for,
        plural_relations=plural_relations,
        max_phrase=max(len(t) for t in phrases),
    )


def load_lexicon(path: Path, tree: LoadedTree) -> Lexicon:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise LexiconError(f"cannot read {path}: {exc}") from exc
    return parse_lexicon(raw, tree)
