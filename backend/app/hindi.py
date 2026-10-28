"""Hindi script -> Hinglish spelling, so Devanagari input reaches the same lexicon.

People type Hindi in English letters with loose, varied spellings ("ghumne",
"ghoomne", "jaana", "jana"). This converts a Devanagari word into the spellings
such a person would type, so "घूमने" -> "ghoomne" and "जापान" -> "jaapaan" /
"japan", which the lexicon and typo matching already understand.

It applies Hindi schwa deletion (the inherent 'a' that isn't pronounced is
dropped): word-finally, and medially in the context vowel-C_C-vowel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

CONSONANTS: Final = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "ळ": "l",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f", "य़": "y",
}  # fmt: skip
NUKTA_FORMS: Final = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f"}
VOWELS: Final = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e",
}  # fmt: skip
MATRAS: Final = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e",
}  # fmt: skip
NASALS: Final = {"ं": "n", "ँ": "", "ः": "h"}  # noqa: RUF001 - Devanagari signs
VIRAMA: Final = "्"
NUKTA: Final = "़"
DIGITS: Final = str.maketrans("०१२३४५६७८९", "0123456789")
_COLLAPSE: Final = (("aa", "a"), ("ee", "i"), ("oo", "u"))


@dataclass(slots=True)
class _Unit:
    consonant: str  # "" for a standalone vowel
    vowel: str  # what follows the consonant ("" after a virama)
    inherent: bool  # vowel is the unwritten 'a' (a deletion candidate)


def _units(word: str) -> list[_Unit | str]:
    """Split into consonant+vowel units; nasals stay as plain strings."""
    out: list[_Unit | str] = []
    chars = list(word)
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch in CONSONANTS:
            base = CONSONANTS[ch]
            if i + 1 < len(chars) and chars[i + 1] == NUKTA:
                base = NUKTA_FORMS.get(ch, base)
                i += 1
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            if nxt in MATRAS:
                out.append(_Unit(base, MATRAS[nxt], inherent=False))
                i += 2
            elif nxt == VIRAMA:
                out.append(_Unit(base, "", inherent=False))
                i += 2
            else:
                out.append(_Unit(base, "a", inherent=True))
                i += 1
        elif ch in VOWELS:
            out.append(_Unit("", VOWELS[ch], inherent=False))
            i += 1
        elif ch in NASALS:
            out.append(NASALS[ch])
            i += 1
        else:
            i += 1  # zero-width joiners, danda, anything else
    return out


def _delete_schwas(units: list[_Unit | str]) -> None:
    """Drop the unpronounced 'a': at the end, and in vowel-C_C-vowel (घूमने -> ghoomne)."""
    cv = [u for u in units if isinstance(u, _Unit)]
    if len(cv) > 1 and cv[-1].inherent and cv[-1].consonant:
        cv[-1].vowel = ""
    for k in range(len(cv) - 2, 0, -1):
        unit, before, after = cv[k], cv[k - 1], cv[k + 1]
        if unit.inherent and unit.consonant and before.vowel and after.consonant and after.vowel:
            unit.vowel = ""


def romanize(word: str) -> list[str]:
    """Hinglish spellings for one Devanagari word, most literal first.

    >>> romanize("घूमने")
    ['ghoomne', 'ghumne']
    """
    units = _units(word)
    if not any(isinstance(u, _Unit) for u in units):
        return []
    _delete_schwas(units)
    literal = "".join(u if isinstance(u, str) else u.consonant + u.vowel for u in units)
    collapsed = literal
    for long, short in _COLLAPSE:
        collapsed = collapsed.replace(long, short)
    return [literal] if collapsed == literal else [literal, collapsed]
