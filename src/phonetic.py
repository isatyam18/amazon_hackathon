"""
Coarse phonetic skeleton of a token, robust to transliteration and vowel noise.

Used by Stage 1 (matching transliterated region names, e.g. 'maharashtr' ~
'maharashtra') and Stage 2/3 (phonetic name keys and similarities, e.g.
'kulopal pichinas' ~ 'global business').
"""

import re

_DIGRAPHS = (
    ("sch", "s"), ("ph", "f"), ("sh", "s"), ("ch", "s"), ("th", "t"), ("kh", "k"),
    ("gh", "g"), ("dh", "d"), ("bh", "b"), ("jh", "j"), ("ck", "k"), ("qu", "k"),
    ("x", "ks"),
)
_SOFT_C_RE = re.compile(r"c(?=[eiy])")
_PHONETIC_CLASS = str.maketrans({
    "b": "p", "p": "p", "f": "p",
    "v": "v", "w": "v",
    "c": "k", "g": "k", "k": "k", "q": "k",
    "d": "t", "t": "t",
    "s": "s", "z": "s",
})
_VOWELS = set("aeiouyh")


def phonetic_key(token: str) -> str:
    """Coarse consonant skeleton: voicing merged (b/p, d/t, g/k), vowels and 'h' dropped.

    A leading vowel is kept as 'a' so vowel-initial words don't collapse onto
    their consonant-initial neighbours. Digits pass through unchanged.
    """
    if not token:
        return ""
    t = token
    for src, dst in _DIGRAPHS:
        t = t.replace(src, dst)
    t = _SOFT_C_RE.sub("s", t)
    t = t.translate(_PHONETIC_CLASS)
    out = ["a"] if t[0] in _VOWELS else []
    prev = ""
    for c in t:
        if c in _VOWELS:
            prev = ""
            continue
        if c != prev:
            out.append(c)
        prev = c
    return "".join(out)
