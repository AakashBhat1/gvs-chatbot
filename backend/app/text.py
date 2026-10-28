"""Input cleaning and tokenisation.

The length limit is applied *after* Unicode NFC normalisation, so decomposed
characters can't be used to slip past it, and control/format characters
(zero-width joiners, bidi overrides) are stripped before anything else sees the
text.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

MAX_MESSAGE_CHARS: Final = 500

_DROP_CATEGORIES = frozenset({"Cc", "Cf", "Co", "Cs", "Cn"})
_SPACE = re.compile(r"\s+")
# Abbreviations whose dots would otherwise split them into letters.
_REWRITES: Final = (
    (re.compile(r"\bu\.?s\.?a\.?(?=\W|$)"), "usa"),
    (re.compile(r"\bu\.s\.?(?=\W|$)"), "usa"),
    (re.compile(r"\bu\.k\.?(?=\W|$)"), "uk"),
    (re.compile(r"\bthe us\b"), "usa"),
    (re.compile(r"\bb\.tech\b"), "btech"),
    (re.compile(r"\bdr\.(?=\s|$)"), "dr"),
)
_TOKEN = re.compile(
    r"[0-9]+"  # numbers
    r"|[a-z]+(?:['\u2019\-/][a-z]+)*"  # words, incl. i'm, co-founder, y/o
    r"|[\u0900-\u097f]+"  # Devanagari words (understood by AI brains only)
    r"|[,.;!?&+]"  # separators
)


class MessageTooLong(ValueError):
    pass


def clean(raw: str, limit: int = MAX_MESSAGE_CHARS) -> str:
    """NFC-normalise, strip invisible characters, collapse whitespace, enforce the limit."""
    text = unicodedata.normalize("NFC", raw)
    text = "".join(
        ch if unicodedata.category(ch) not in _DROP_CATEGORIES or ch in "\n\t" else " "
        for ch in text
    )
    text = _SPACE.sub(" ", text).strip()
    if len(text) > limit:
        raise MessageTooLong(f"{len(text)} characters; limit is {limit}")
    return text


def tokenize(text: str) -> list[str]:
    """Lowercase tokens; apostrophes normalised to ASCII."""
    lowered = text.lower().replace("\u2019", "'")
    for pattern, replacement in _REWRITES:
        lowered = pattern.sub(replacement, lowered)
    return _TOKEN.findall(lowered)


def has_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097f" for ch in text)
