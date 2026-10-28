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
from app.brains.rules import RulesBrain

DEFAULT_BRAINS: Final = "rules"

# Each entry builds one brain. "local" and "haiku" register here when they exist.
_BUILDERS: Final[dict[str, Callable[[Lexicon], Brain]]] = {
    "rules": RulesBrain,
}


class UnknownBrain(ValueError):
    pass


def brain_names(setting: str | None = None) -> list[str]:
    raw = setting if setting is not None else os.environ.get("GVS_BRAINS", DEFAULT_BRAINS)
    names = [n.strip().lower() for n in raw.split(",") if n.strip()]
    if unknown := [n for n in names if n not in _BUILDERS]:
        raise UnknownBrain(f"unknown brain(s) {unknown}; available: {sorted(_BUILDERS)}")
    if len(set(names)) != len(names):
        raise UnknownBrain(f"brain listed twice in {raw!r}")
    if "rules" not in names:
        names.insert(0, "rules")
    return names


def build_cascade(lexicon: Lexicon, names: Sequence[str] | None = None) -> Cascade:
    return Cascade([_BUILDERS[n](lexicon) for n in (names or brain_names())])
