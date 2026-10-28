from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from app.tree import (
    MAX_TREE_BYTES,
    PathKey,
    RelationKey,
    TreeError,
    TreeStore,
    load_tree,
    parse_rule_key,
    parse_tree,
)

WriteTree = Callable[[dict[str, Any]], Path]


def as_bytes(obj: dict[str, Any]) -> bytes:
    return json.dumps(obj).encode()


# ----------------------------------------------------------------- rule keys


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("*", PathKey(None, None, None)),
        ("jp", PathKey("jp", None, None)),
        ("jp/tourism", PathKey("jp", "tourism", None)),
        ("*/*/employed", PathKey(None, None, "employed")),
        ("jp/*/employed", PathKey("jp", None, "employed")),
        ("@son", RelationKey("son")),
    ],
)
def test_parse_rule_key(raw: str, expected: object) -> None:
    assert parse_rule_key(raw) == expected


@pytest.mark.parametrize("raw", ["", "@", "a/b/c/d", "jp//employed", "/jp"])
def test_parse_rule_key_rejects_malformed(raw: str) -> None:
    with pytest.raises(TreeError):
        parse_rule_key(raw)


# ----------------------------------------------------------------- validation


def test_real_tree_loads(real_tree: Any) -> None:
    assert len(real_tree.data.countries) == 15
    assert len(real_tree.data.purposes) == 3
    assert len(real_tree.data.professions) == 17
    assert len(real_tree.data.relations) == 9


def test_mini_tree_compiles_family_and_groups(mini_tree: Any) -> None:
    assert mini_tree.family_documents == ("cover",)
    assert ("son", ("birth_cert",)) in mini_tree.relation_rules
    assert ("daughter", ("birth_cert",)) in mini_tree.relation_rules
    specificity = [r.key.specificity for r in mini_tree.path_rules]
    assert specificity == sorted(specificity)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda t: t["rules"].update({"jp": ["nope"]}), "unknown document"),
        (lambda t: t["rules"].update({"mars": ["passport"]}), "unknown country"),
        (lambda t: t["rules"].update({"*/holiday": ["passport"]}), "unknown purpose"),
        (lambda t: t["rules"].update({"*/*/pilot": ["passport"]}), "unknown profession"),
        (lambda t: t["rules"].update({"@uncle": ["passport"]}), "unknown relation"),
        (lambda t: t["rules"].update({"jp/*/*": ["passport"]}), "duplicates"),
        (lambda t: t["rules"].update({"uk": []}), "empty"),
        (lambda t: t["rules"].update({"uk": ["passport", "passport"]}), "listed twice"),
        (lambda t: t["rules"].update({"a/b/c/d": ["passport"]}), "expected"),
        (lambda t: t["countries"].update({"Bad Id": {"name": "x"}}), "pattern"),
        (lambda t: t.update({"surprise": 1}), "Extra inputs"),
        (lambda t: t["meta"].update({"schema_version": 2}), "schema_version"),
        (lambda t: t["age_bands"].pop("*"), "'*' default"),
        (lambda t: t["age_bands"].update({"mars": []}), "unknown country"),
        (
            lambda t: t["age_bands"]["*"].reverse(),
            "strictly increasing",
        ),
        (
            lambda t: t["age_bands"]["*"].append({"max_age": 30, "profession": "pilot"}),
            "unknown profession",
        ),
        (lambda t: t["age_bands"]["*"][0].update({"max_age": "2"}), "integer"),
        (lambda t: t["relation_groups"].update({"son": ["son"]}), "clashes"),
        (lambda t: t["relation_groups"].update({"family": ["son"]}), "clashes"),
        (lambda t: t["relation_groups"].update({"kids": []}), "must not be empty"),
        (lambda t: t["relation_groups"].update({"kids": ["nephew"]}), "unknown relation"),
        (lambda t: t["documents"]["passport"].update({"label": "  "}), "at least 1"),
    ],
)
def test_invalid_trees_rejected(
    mini_dict: dict[str, Any], mutate: Callable[[dict[str, Any]], None], message: str
) -> None:
    mutate(mini_dict)
    with pytest.raises(TreeError, match=message):
        parse_tree(as_bytes(mini_dict))


def test_duplicate_json_keys_rejected() -> None:
    raw = b'{"meta": {}, "meta": {}}'
    with pytest.raises(TreeError, match="duplicate JSON key 'meta'"):
        parse_tree(raw)


@pytest.mark.parametrize("raw", [b"not json", b"\xff\xfe", b"[]"])
def test_garbage_rejected(raw: bytes) -> None:
    with pytest.raises(TreeError):
        parse_tree(raw)


def test_oversized_tree_rejected(tmp_path: Path) -> None:
    with pytest.raises(TreeError, match="limit"):
        parse_tree(b" " * (MAX_TREE_BYTES + 1))
    big = tmp_path / "big.json"
    big.write_bytes(b" " * (MAX_TREE_BYTES + 1))
    with pytest.raises(TreeError, match="limit"):
        load_tree(big)


def test_missing_file_is_tree_error(tmp_path: Path) -> None:
    with pytest.raises(TreeError, match="cannot read"):
        load_tree(tmp_path / "absent.json")


def test_version_tracks_content(mini_dict: dict[str, Any]) -> None:
    a = parse_tree(as_bytes(mini_dict)).version
    assert parse_tree(as_bytes(mini_dict)).version == a
    mini_dict["documents"]["passport"]["label"] = "Passport (edited)"
    assert parse_tree(as_bytes(mini_dict)).version != a


# ----------------------------------------------------------------- hot reload


def _bump_mtime(path: Path) -> None:
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


def test_store_reloads_valid_change(mini_dict: dict[str, Any], write_tree: WriteTree) -> None:
    path = write_tree(mini_dict)
    store = TreeStore(path, check_interval=0)
    first = store.get().version

    mini_dict["documents"]["passport"]["label"] = "Passport v2"
    write_tree(mini_dict)
    _bump_mtime(path)
    assert store.get().version != first
    assert store.get().data.documents["passport"].label == "Passport v2"
    assert store.path == path


def test_store_keeps_last_good_tree_on_bad_edit(
    mini_dict: dict[str, Any], write_tree: WriteTree, caplog: pytest.LogCaptureFixture
) -> None:
    path = write_tree(mini_dict)
    store = TreeStore(path, check_interval=0)
    good = store.get()

    path.write_text("{ broken", encoding="utf-8")
    _bump_mtime(path)
    assert store.get() is good
    assert "reload rejected" in caplog.text

    path.unlink()  # file vanishing must not take the service down either
    assert store.get() is good


def test_store_throttles_checks(mini_dict: dict[str, Any], write_tree: WriteTree) -> None:
    path = write_tree(mini_dict)
    store = TreeStore(path, check_interval=3600)
    first = store.get()
    mini_dict["documents"]["passport"]["label"] = "changed"
    write_tree(mini_dict)
    _bump_mtime(path)
    assert store.get() is first  # not re-checked within the interval


def test_store_fails_fast_on_invalid_startup(write_tree: WriteTree) -> None:
    with pytest.raises(TreeError):
        TreeStore(write_tree({"meta": {}}))
