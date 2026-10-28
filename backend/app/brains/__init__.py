"""Brains turn one user message into an :class:`~app.extraction.Extraction`.

They are interchangeable. The :class:`Cascade` tries them cheapest first and
stops at the first *confident* answer; if a brain is down, out of budget or
refuses, it moves on. The rules brain never fails, so the bot always works,
even with no API key and no local model.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.extraction import Context, Extraction

log = logging.getLogger(__name__)


class BrainUnavailable(Exception):
    """The brain could not answer this time (timeout, quota, refusal, missing model)."""


@dataclass(frozen=True, slots=True)
class BrainResult:
    extraction: Extraction
    confident: bool
    brain: str


class Brain(Protocol):
    name: str

    def understand(self, message: str, ctx: Context) -> BrainResult: ...


class Cascade:
    """Try brains in order; return the first confident result.

    If none is confident, the answer from the *last* brain that answered is
    used, since later brains are the more capable ones. At least one brain
    must never raise :class:`BrainUnavailable` (the rules brain).
    """

    name = "cascade"

    def __init__(self, brains: Sequence[Brain]) -> None:
        if not brains:
            raise ValueError("cascade needs at least one brain")
        self.brains = tuple(brains)

    def understand(self, message: str, ctx: Context) -> BrainResult:
        best: BrainResult | None = None
        for brain in self.brains:
            try:
                result = brain.understand(message, ctx)
            except BrainUnavailable as exc:
                log.warning("brain %s unavailable: %s", brain.name, exc)
                continue
            if result.confident:
                return result
            best = result
        if best is None:
            raise BrainUnavailable("no brain could answer")
        return best
