"""
Blocking-specific normalization on top of Stage 1 output (`clean_name`, `clean_address`).

Stage 1 already lowercases, de-accents, transliterates Indic scripts and expands
address abbreviations. For candidate generation we additionally:

- drop legal-form / filler tokens *anywhere* in the name (noise moves them around:
  'Ltd. Ontime Pvt. Services', '[LLC] Coastal Tungsten', 'M/s ...', 'D.B.A.'),
- undo digit-for-letter substitutions inside words ('s0reis' -> 'soreis', '5ervices'),
- drop long numeric ids from names ('(ID: [14307)]', '#98825'),
- build a coarse phonetic skeleton that survives transliteration and vowel noise
  ('kulopal pichinas' ~ 'global business', 'entarpraisas' ~ 'enterprises').
"""

import re
from typing import List

from src.phonetic import phonetic_key  # shared with Stage 1; re-exported here

# Legal forms (US / India / France + common others) and their transliterated spellings
LEGAL_TOKENS = {
    "pvt", "private", "ltd", "limited", "llp", "opc", "llc", "inc", "incorporated",
    "corp", "corporation", "co", "company", "lp", "pllc", "plc", "pc", "pte", "pty",
    "sarl", "sas", "sasu", "sa", "eurl", "snc", "sci", "gmbh",
    # transliterations produced by src/transliteration.py from Indic-script names
    "praivet", "piraivet", "praivat", "pryvet", "limitet", "limitad", "limited",
    "limittad", "limitted", "kampani", "kampni", "kampeni",
}

# Filler tokens that carry no identity signal in names (EN / FR / web noise)
NAME_FILLER_TOKENS = {
    "the", "and", "of", "dba", "aka", "ms", "com", "www", "net", "org", "in", "id",
    "null", "et", "de", "du", "des", "la", "le", "les",
}

NAME_STOP_TOKENS = LEGAL_TOKENS | NAME_FILLER_TOKENS

# Address tokens that are pure formatting noise after Stage 1 expansion ('H.No' -> 'h number')
ADDRESS_STOP_TOKENS = {"null", "number", "no", "h", "door", "nan", "none"}

_LEET_MAP = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})
_HOUSE_NUMBER_RE = re.compile(r"^\d+[a-z]?$")
_ORDINAL_RE = re.compile(r"^\d+(st|nd|rd|th)$")
_ALNUM_SPLIT_RE = re.compile(r"^(\d+)([a-z]+)$")


def fix_leet(token: str) -> str:
    """Map digits inside alphabetic words back to letters ('c0astal' -> 'coastal').

    Leaves numbers, house numbers ('127a') and ordinals ('3rd') untouched.
    """
    if token.isalpha() or token.isdigit():
        return token
    if _HOUSE_NUMBER_RE.match(token) or _ORDINAL_RE.match(token):
        return token
    n_alpha = sum(c.isalpha() for c in token)
    n_digit = len(token) - n_alpha
    if n_alpha >= 2 and n_digit <= 2:
        return token.translate(_LEET_MAP)
    return token


def name_tokens(clean_name: str) -> List[str]:
    """Identity-bearing tokens of a Stage 1 `clean_name`.

    Falls back to all tokens when filtering would leave nothing (e.g. a name that
    is only 'the company').
    """
    if not clean_name:
        return []
    raw = [fix_leet(t) for t in clean_name.split()]
    kept = [
        t for t in raw
        if t not in NAME_STOP_TOKENS and not (t.isdigit() and len(t) >= 4)
    ]
    # 'm s' is what 'M/s' becomes after punctuation stripping
    if len(kept) >= 2 and kept[0] == "m" and kept[1] == "s":
        kept = kept[2:]
    return kept if kept else raw


def address_tokens(clean_address: str) -> List[str]:
    """Tokens of a Stage 1 `clean_address`: leading zeros stripped from numbers ('08919' -> '8919'),
    '127a'-style numbers also emitted as '127'."""
    if not clean_address:
        return []
    out = []
    for t in clean_address.split():
        if t in ADDRESS_STOP_TOKENS:
            continue
        if t[0] == "0" and t.isdigit():
            t = t.lstrip("0") or "0"
        out.append(t)
        m = _ALNUM_SPLIT_RE.match(t)
        if m:
            out.append(m.group(1))
    return out


def phonetic_name(tokens: List[str]) -> str:
    """Space-joined phonetic skeletons of name tokens."""
    return " ".join(k for k in (phonetic_key(t) for t in tokens) if k)
