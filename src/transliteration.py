"""
Rule-based Indic script -> Latin transliteration (no external data or libraries).

Source 2/3 records for India contain business names and addresses written in
Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada and
Malayalam (e.g. 'राम मार्केटिंग प्राइवेट लिमिटेड' for 'Ram Marketing Private Limited').
Without transliteration, `clean_text` strips every non-ASCII letter and these
records become empty strings.

All nine scripts share the ISCII-derived Unicode layout: each occupies a
128-code-point block and a given offset inside the block encodes the same
phoneme (e.g. offset 0x15 is KA in every script). One offset table therefore
covers all of them.

The output is a loose, lowercase romanisation tuned for matching rather than
linguistic accuracy: long/short vowels are merged ('aa' -> 'a'), the inherent
vowel is dropped at the end of a word (Hindi-style schwa deletion), and
nukta/length marks are folded into the base letter.
"""

from typing import Dict

INDIC_START = 0x0900
INDIC_END = 0x0D7F

# Independent vowels (offset within block -> latin)
_VOWELS: Dict[int, str] = {
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u",
    0x0B: "ri", 0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai",
    0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li",
    0x72: "i", 0x73: "u",  # Gurmukhi vowel bearers
}

# Consonants (carry an inherent 'a' unless followed by a vowel sign / virama)
_CONSONANTS: Dict[int, str] = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "ny",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh",
    0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x58: "q", 0x59: "kh", 0x5A: "gh", 0x5B: "z", 0x5C: "r", 0x5D: "rh",
    0x5E: "f", 0x5F: "y",
}

# Dependent vowel signs (replace the inherent vowel of the preceding consonant)
_VOWEL_SIGNS: Dict[int, str] = {
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri",
    0x44: "ri", 0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o",
    0x4A: "o", 0x4B: "o", 0x4C: "au", 0x62: "li", 0x63: "li",
}

_VIRAMA = 0x4D
_NUKTA = 0x3C
_NASALS = (0x01, 0x02)  # candrabindu, anusvara
_VISARGA = 0x03
_DANDAS = (0x64, 0x65)
_GURMUKHI_TIPPI = 0x0A70

# Nukta turns a native consonant into a borrowed sound (ज़ -> z, फ़ -> f, ड़ -> r)
_NUKTA_MAP = {"j": "z", "ph": "f", "k": "q", "d": "r", "dh": "rh", "g": "g", "kh": "kh"}

# Script-specific code points outside the shared layout
_SPECIAL: Dict[int, str] = {
    0x09CE: "t",   # Bengali khanda ta
    0x09F0: "r",   # Assamese ra
    0x09F1: "w",   # Assamese wa
    0x0D54: "m", 0x0D55: "y", 0x0D56: "l",  # Malayalam chillu m / y / lll
    0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l", 0x0D7F: "k",  # Malayalam chillus
}

_ZERO_WIDTH = {"‌", "‍"}

# Malayalam writes the loanword 'tt' sound as RRA + virama + RRA (റ്റ); map it to TTA (ട്ട)
_MALAYALAM_TT = ("റ്റ", "ട്ട")


def has_indic(text: str) -> bool:
    """True if the string contains at least one Indic-script code point."""
    return any(INDIC_START <= ord(c) <= INDIC_END for c in text)


def transliterate_indic(text: str) -> str:
    """Transliterate any Indic-script runs in `text` to Latin; other characters pass through."""
    if not text or not has_indic(text):
        return text or ""

    text = text.replace(*_MALAYALAM_TT)
    out = []
    pending = False  # last emitted token is a consonant still owning its inherent 'a'

    def flush_inherent():
        nonlocal pending
        if pending:
            out.append("a")
            pending = False

    for ch in text:
        cp = ord(ch)
        if ch in _ZERO_WIDTH:
            continue
        if not (INDIC_START <= cp <= INDIC_END):
            # Word boundary / non-Indic char: word-final inherent vowel is dropped
            pending = False
            out.append(ch)
            continue

        if cp in _SPECIAL:
            flush_inherent()
            out.append(_SPECIAL[cp])
            continue
        if cp == _GURMUKHI_TIPPI:
            flush_inherent()
            out.append("n")
            continue

        off = cp & 0x7F
        if off in _CONSONANTS:
            flush_inherent()
            out.append(_CONSONANTS[off])
            pending = True
        elif off in _VOWEL_SIGNS:
            out.append(_VOWEL_SIGNS[off])
            pending = False
        elif off == _VIRAMA:
            pending = False
        elif off == _NUKTA:
            if out and out[-1] in _NUKTA_MAP:
                out[-1] = _NUKTA_MAP[out[-1]]
        elif off in _VOWELS:
            flush_inherent()
            out.append(_VOWELS[off])
        elif off in _NASALS:
            flush_inherent()
            out.append("n")
        elif off == _VISARGA:
            flush_inherent()
            out.append("h")
        elif 0x66 <= off <= 0x6F:
            flush_inherent()
            out.append(str(off - 0x66))
        elif off in _DANDAS:
            pending = False
            out.append(" ")
        # Anything else (length marks, avagraha, addak, abbreviation signs) is dropped

    return "".join(out)
