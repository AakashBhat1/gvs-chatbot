from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from app.paths import default_tree_path
from app.tree import LoadedTree, load_tree, parse_tree

# A tiny, fully-specified tree so tests don't depend on placeholder data.
MINI_TREE: dict[str, Any] = {
    "meta": {"schema_version": 1, "status": "reviewed"},
    "countries": {
        "jp": {"name": "Japan", "last_reviewed": "2026-10-01"},
        "uk": {"name": "United Kingdom", "last_reviewed": "2026-10-01"},
    },
    "purposes": {"tourism": {"label": "Tourism"}, "business": {"label": "Business"}},
    "professions": {
        "employed": {"label": "Employed"},
        "lawyer": {"label": "Lawyer"},
        "infant": {"label": "Infant"},
        "school_student": {"label": "School student"},
    },
    "relations": {
        "self": {"label": "You"},
        "mother": {"label": "Mother"},
        "son": {"label": "Son"},
        "daughter": {"label": "Daughter"},
    },
    "relation_groups": {"child": ["son", "daughter"]},
    "age_bands": {
        "*": [
            {"max_age": 2, "profession": "infant"},
            {"max_age": 17, "profession": "school_student"},
        ],
        "uk": [{"max_age": 15, "profession": "school_student"}],
    },
    "documents": {
        "passport": {"label": "Passport"},
        "photos": {"label": "Photos"},
        "jp_form": {"label": "Japan visa form"},
        "salary": {"label": "Salary slips"},
        "bar_id": {"label": "Bar council ID"},
        "birth_cert": {"label": "Birth certificate"},
        "school_letter": {"label": "School letter"},
        "cover": {"label": "Covering letter"},
        "biz_invite": {"label": "Business invitation"},
    },
    "rules": {
        "jp": ["jp_form"],
        "*": ["passport", "photos"],
        "*/*/employed": ["salary"],
        "*/*/lawyer": ["bar_id"],
        "*/*/school_student": ["school_letter"],
        "*/business": ["biz_invite"],
        "@child": ["birth_cert"],
        "@family": ["cover"],
    },
}


@pytest.fixture
def mini_dict() -> dict[str, Any]:
    return copy.deepcopy(MINI_TREE)


@pytest.fixture
def mini_tree() -> LoadedTree:
    return parse_tree(json.dumps(MINI_TREE).encode())


@pytest.fixture
def real_tree() -> LoadedTree:
    return load_tree(default_tree_path())


@pytest.fixture
def write_tree(tmp_path: Path) -> Callable[[dict[str, Any]], Path]:
    def _write(obj: dict[str, Any], name: str = "visa-tree.json") -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(obj), encoding="utf-8")
        return path

    return _write
