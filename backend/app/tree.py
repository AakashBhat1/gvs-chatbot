"""Load, validate and hot-reload ``visa-tree.json``.

The tree is the *only* source of document lists, so it is validated strictly:
unknown fields, unknown ids, duplicate JSON keys and malformed rule keys are all
rejected at load time. A bad edit on the server therefore never reaches users —
:class:`TreeStore` keeps serving the last good version instead.

Rule key grammar (keys of ``rules``)::

    "*"                       every traveller
    "<country>"               == "<country>/*/*"
    "<country>/<purpose>"     == "<country>/<purpose>/*"
    "<c>/<p>/<profession>"    any segment may be "*"
    "@<relation>"             travellers with that relation (e.g. "@son")
    "@<group>"                travellers whose relation is in relation_groups[group]
    "@family"                 shared items, once per (country, purpose) group of 2+
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StringConstraints, model_validator

log = logging.getLogger(__name__)

MAX_TREE_BYTES: Final = 1_048_576  # 1 MiB — the real file should be well under 200 KB
WILDCARD: Final = "*"
FAMILY: Final = "family"

Id = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]{0,39}$")]
Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
Note = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=400)]


class TreeError(ValueError):
    """Raised when ``visa-tree.json`` is unreadable or invalid."""


# --------------------------------------------------------------------------- schema


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Meta(_Strict):
    schema_version: Literal[1]
    status: Literal["placeholder", "draft", "reviewed"]
    note: str = Field(default="", max_length=500)
    # Base URL for links in replies; country pages are <site_url>/visa/<id>.
    site_url: Annotated[str, StringConstraints(pattern=r"^https://[a-z0-9.-]+$")] | None = None


Iso2 = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]


class Country(_Strict):
    name: Label
    last_reviewed: date | None = None
    iso2: Iso2 | None = None  # ISO 3166 code; links open data (cities, names) to this id


class OtherCountry(_Strict):
    """A country the website covers but the bot has no checklist for yet."""

    name: Label
    iso2: Iso2 | None = None


class Labelled(_Strict):
    label: Label


class AgeBand(_Strict):
    max_age: Annotated[StrictInt, Field(ge=0, le=110)]
    profession: Id


class Document(_Strict):
    label: Label
    note: Note | None = None


@dataclass(frozen=True, slots=True)
class PathKey:
    """A parsed path rule; ``None`` means wildcard."""

    country: str | None
    purpose: str | None
    profession: str | None

    @property
    def specificity(self) -> int:
        return sum(seg is not None for seg in (self.country, self.purpose, self.profession))


@dataclass(frozen=True, slots=True)
class RelationKey:
    target: str  # a relation id, a relation-group id, or FAMILY


RuleKey = PathKey | RelationKey


def parse_rule_key(raw: str) -> RuleKey:
    """Parse a ``rules`` key into a structured form (syntax only, no id checks)."""
    if raw.startswith("@"):
        target = raw[1:]
        if not target:
            raise TreeError(f"rule key {raw!r}: empty relation after '@'")
        return RelationKey(target)
    parts = raw.split("/")
    if not 1 <= len(parts) <= 3 or any(p == "" for p in parts):
        raise TreeError(f"rule key {raw!r}: expected 'country[/purpose[/profession]]'")
    parts += [WILDCARD] * (3 - len(parts))
    country, purpose, profession = (None if p == WILDCARD else p for p in parts)
    return PathKey(country, purpose, profession)


class VisaTree(_Strict):
    meta: Meta
    countries: dict[Id, Country] = Field(min_length=1)
    other_countries: dict[Id, OtherCountry] = Field(default_factory=dict)
    purposes: dict[Id, Labelled] = Field(min_length=1)
    professions: dict[Id, Labelled] = Field(min_length=1)
    relations: dict[Id, Labelled] = Field(min_length=1)
    relation_groups: dict[Id, list[Id]] = Field(default_factory=dict)
    age_bands: dict[str, list[AgeBand]]
    documents: dict[Id, Document]
    rules: dict[str, list[Id]]

    @model_validator(mode="after")
    def _check_references(self) -> VisaTree:
        errors: list[str] = [
            f"other_countries.{c}: already a supported country"
            for c in self.other_countries
            if c in self.countries
        ]
        codes: dict[str, str] = {}
        listed: list[tuple[str, Country | OtherCountry]] = [
            *self.countries.items(),
            *self.other_countries.items(),
        ]
        for cid, c in listed:
            if c.iso2 is not None:
                if c.iso2 in codes:
                    errors.append(f"{cid}: iso2 {c.iso2} already used by {codes[c.iso2]}")
                codes[c.iso2] = cid
        errors += self._check_groups()
        errors += self._check_age_bands()
        errors += self._check_rules()
        if errors:
            raise ValueError("; ".join(errors))
        return self

    def _check_groups(self) -> list[str]:
        errors: list[str] = []
        for gid, members in self.relation_groups.items():
            if gid in self.relations or gid == FAMILY:
                errors.append(f"relation_groups.{gid}: clashes with a relation id or '@family'")
            if not members:
                errors.append(f"relation_groups.{gid}: must not be empty")
            errors += [
                f"relation_groups.{gid}: unknown relation {m!r}"
                for m in members
                if m not in self.relations
            ]
        return errors

    def _check_age_bands(self) -> list[str]:
        errors: list[str] = []
        if WILDCARD not in self.age_bands:
            errors.append("age_bands: a '*' default is required")
        for key, bands in self.age_bands.items():
            if key != WILDCARD and key not in self.countries:
                errors.append(f"age_bands.{key}: unknown country")
            ages = [b.max_age for b in bands]
            if ages != sorted(set(ages)):
                errors.append(f"age_bands.{key}: max_age must be strictly increasing")
            errors += [
                f"age_bands.{key}: unknown profession {b.profession!r}"
                for b in bands
                if b.profession not in self.professions
            ]
        return errors

    def _check_rules(self) -> list[str]:
        errors: list[str] = []
        seen: dict[RuleKey, str] = {}
        relation_targets = {*self.relations, *self.relation_groups, FAMILY}
        for raw, doc_ids in self.rules.items():
            try:
                key = parse_rule_key(raw)
            except TreeError as exc:
                errors.append(str(exc))
                continue
            if key in seen:
                errors.append(f"rules: {raw!r} duplicates {seen[key]!r}")
            seen[key] = raw
            if isinstance(key, RelationKey):
                if key.target not in relation_targets:
                    errors.append(f"rules.{raw}: unknown relation or group {key.target!r}")
            else:
                for value, table, kind in (
                    (key.country, self.countries, "country"),
                    (key.purpose, self.purposes, "purpose"),
                    (key.profession, self.professions, "profession"),
                ):
                    if value is not None and value not in table:
                        errors.append(f"rules.{raw}: unknown {kind} {value!r}")
            if not doc_ids:
                errors.append(f"rules.{raw}: document list is empty")
            if len(doc_ids) != len(set(doc_ids)):
                errors.append(f"rules.{raw}: document listed twice")
            errors += [
                f"rules.{raw}: unknown document {d!r}" for d in doc_ids if d not in self.documents
            ]
        return errors


# --------------------------------------------------------------------------- compiled


@dataclass(frozen=True, slots=True)
class PathRule:
    key: PathKey
    documents: tuple[str, ...]

    def matches(self, country: str, purpose: str, profession: str) -> bool:
        k = self.key
        return (
            (k.country is None or k.country == country)
            and (k.purpose is None or k.purpose == purpose)
            and (k.profession is None or k.profession == profession)
        )


@dataclass(frozen=True, slots=True)
class LoadedTree:
    """A validated tree plus pre-compiled rules. Immutable; safe to share across threads."""

    data: VisaTree
    version: str
    path_rules: tuple[PathRule, ...]  # least specific first, then file order
    relation_rules: tuple[tuple[str, tuple[str, ...]], ...]  # (relation id, docs), file order
    family_documents: tuple[str, ...]

    @classmethod
    def compile(cls, data: VisaTree, version: str) -> LoadedTree:
        path_rules: list[tuple[int, int, PathRule]] = []
        relation_rules: list[tuple[str, tuple[str, ...]]] = []
        family: list[str] = []
        for order, (raw, docs) in enumerate(data.rules.items()):
            key = parse_rule_key(raw)
            if isinstance(key, PathKey):
                path_rules.append((key.specificity, order, PathRule(key, tuple(docs))))
            elif key.target == FAMILY:
                family += [d for d in docs if d not in family]
            elif key.target in data.relation_groups:
                relation_rules += [(rel, tuple(docs)) for rel in data.relation_groups[key.target]]
            else:
                relation_rules.append((key.target, tuple(docs)))
        path_rules.sort(key=lambda item: (item[0], item[1]))
        return cls(
            data=data,
            version=version,
            path_rules=tuple(rule for _, _, rule in path_rules),
            relation_rules=tuple(relation_rules),
            family_documents=tuple(family),
        )

    def age_bands_for(self, country: str) -> list[AgeBand]:
        bands = self.data.age_bands
        return bands.get(country) or bands[WILDCARD]


# --------------------------------------------------------------------------- loading


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise TreeError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def parse_tree(raw: bytes) -> LoadedTree:
    """Validate raw ``visa-tree.json`` bytes. Raises :class:`TreeError`."""
    if len(raw) > MAX_TREE_BYTES:
        raise TreeError(f"tree is {len(raw)} bytes; limit is {MAX_TREE_BYTES}")
    try:
        obj = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
        data = VisaTree.model_validate(obj)
    except TreeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise TreeError(str(exc)) from exc
    version = hashlib.sha256(raw).hexdigest()[:12]
    return LoadedTree.compile(data, version)


def load_tree(path: Path) -> LoadedTree:
    try:
        size = path.stat().st_size
        if size > MAX_TREE_BYTES:
            raise TreeError(f"{path}: {size} bytes; limit is {MAX_TREE_BYTES}")
        raw = path.read_bytes()
    except OSError as exc:
        raise TreeError(f"cannot read {path}: {exc}") from exc
    return parse_tree(raw)


class TreeStore:
    """Serves the current tree and reloads it when the file changes.

    * Start-up fails loudly if the file is invalid.
    * A later invalid edit is logged and ignored: the last good tree stays live.
    * The file is checked at most once per ``check_interval`` seconds, so
      :meth:`get` is cheap on the request path.

    Replace the file atomically (write a temp file, then rename) so a reader
    never sees a half-written tree.
    """

    def __init__(self, path: Path, check_interval: float = 2.0) -> None:
        self._path = path
        self._interval = check_interval
        self._lock = threading.Lock()
        self._current = load_tree(path)
        self._signature = self._stat_signature()
        self._next_check = time.monotonic() + check_interval

    @property
    def path(self) -> Path:
        return self._path

    def get(self) -> LoadedTree:
        if time.monotonic() >= self._next_check and self._lock.acquire(blocking=False):
            try:
                self._maybe_reload()
            finally:
                self._lock.release()
        return self._current

    def _stat_signature(self) -> tuple[int, int] | None:
        try:
            st = self._path.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _maybe_reload(self) -> None:
        self._next_check = time.monotonic() + self._interval
        signature = self._stat_signature()
        if signature is None or signature == self._signature:
            return
        self._signature = signature  # don't re-parse a bad file on every check
        try:
            fresh = load_tree(self._path)
        except TreeError as exc:
            log.error("visa-tree reload rejected, keeping %s: %s", self._current.version, exc)
            return
        if fresh.version != self._current.version:
            log.info("visa-tree reloaded: %s -> %s", self._current.version, fresh.version)
            self._current = fresh
