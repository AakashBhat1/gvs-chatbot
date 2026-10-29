"""Build the cascade from a setting like ``GVS_BRAINS=rules,local,haiku``.

Order is cheapest first. ``rules`` is always appended if missing, because it
is the brain that never fails — the bot must work with no API key and no model.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Final

from app.brains import Brain, Cascade
from app.brains.lexicon import Lexicon
from app.brains.local import LocalBrain, load_hints
from app.brains.rules import RulesBrain
from app.paths import default_lexicon_path, default_tree_path
from app.tree import LoadedTree, TreeStore

DEFAULT_BRAINS: Final = "rules"

TreeGetter = Callable[[], LoadedTree]

# Each entry builds one brain. "haiku" registers here when it exists.
_BUILDERS: Final[dict[str, Callable[[Lexicon, TreeGetter], Brain]]] = {
    "rules": lambda lexicon, _tree: RulesBrain(lexicon),
    "local": lambda _lexicon, tree: LocalBrain(tree, hints=load_hints(default_lexicon_path())),
}


class UnknownBrain(ValueError):
    pass


def brain_names(setting: str | None = None, *, with_rules: bool = True) -> list[str]:
    """Parse the setting. ``with_rules=False`` is for scoring one brain on its own."""
    raw = setting if setting is not None else os.environ.get("GVS_BRAINS", DEFAULT_BRAINS)
    names = [n.strip().lower() for n in raw.split(",") if n.strip()]
    if unknown := [n for n in names if n not in _BUILDERS]:
        raise UnknownBrain(f"unknown brain(s) {unknown}; available: {sorted(_BUILDERS)}")
    if len(set(names)) != len(names):
        raise UnknownBrain(f"brain listed twice in {raw!r}")
    if with_rules and "rules" not in names:
        names.insert(0, "rules")
    if not names:
        raise UnknownBrain("no brain given")
    return names


def build_cascade(
    lexicon: Lexicon, names: Sequence[str] | None = None, tree: TreeGetter | None = None
) -> Cascade:
    """``tree`` should be the engine's tree getter, so a reload reaches every brain."""
    store: TreeStore | None = None

    def default_tree() -> LoadedTree:  # opened only if a brain asks for it
        nonlocal store
        store = store or TreeStore(default_tree_path())
        return store.get()

    getter = tree or default_tree
    return Cascade([_BUILDERS[n](lexicon, getter) for n in (names or brain_names())])
