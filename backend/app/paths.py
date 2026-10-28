from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_TREE = Path(__file__).resolve().parents[1] / "data" / "visa-tree.json"


def default_tree_path() -> Path:
    """``$GVS_TREE_PATH`` if set, else ``backend/data/visa-tree.json``."""
    override = os.environ.get("GVS_TREE_PATH")
    return Path(override) if override else _DEFAULT_TREE
