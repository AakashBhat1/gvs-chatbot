from __future__ import annotations

import os
from pathlib import Path

_DATA = Path(__file__).resolve().parents[1] / "data"


def default_tree_path() -> Path:
    """``$GVS_TREE_PATH`` if set, else ``backend/data/visa-tree.json``."""
    override = os.environ.get("GVS_TREE_PATH")
    return Path(override) if override else _DATA / "visa-tree.json"


def default_lexicon_path() -> Path:
    """``$GVS_LEXICON_PATH`` if set, else ``backend/data/lexicon.toml``."""
    override = os.environ.get("GVS_LEXICON_PATH")
    return Path(override) if override else _DATA / "lexicon.toml"
