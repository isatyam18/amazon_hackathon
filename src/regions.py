"""
Country-aware address normalisation tables (Stage 1).

Country is an open set: every table is keyed by the lower-cased country label,
and a country without an entry falls back to the generic table (address
abbreviations) or is left untouched (regions). Nothing is filtered by country.

1. Street-type / filler abbreviations. The generic table is English/US.
   France overrides the entries whose meaning differs ('st' is Saint, not
   Street; 'r' is rue, 'ste' is Sainte, not Suite).
2. First-level regions (US states, Indian states / UTs) are mapped to one
   canonical code, whichever form a source uses: 'north carolina' / 'nc',
   'maharashtra' / 'mh', and transliterated regional-script spellings
   ('maharashtr', 'karnatak', 'dilli'). For India, transliterations are matched
   through the phonetic skeleton (src/phonetic.py).
"""

from typing import Dict, List, Optional

from src.phonetic import phonetic_key

# --- 1. Street-type / filler abbreviations -----------------------------------------

GENERIC_ADDRESS_EXPANSIONS: Dict[str, str] = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "hwy": "highway", "ct": "court",
    "pl": "place", "pkwy": "parkway", "fl": "floor", "flr": "floor", "ste": "suite",
    "apt": "apartment", "rm": "room", "opp": "opposite", "nr": "near", "bldg": "building",
    "no": "number", "dist": "district",
}

FRENCH_ADDRESS_EXPANSIONS: Dict[str, str] = {
    **GENERIC_ADDRESS_EXPANSIONS,
    "r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "bld": "boulevard",
    "blvd": "boulevard", "st": "saint", "ste": "sainte", "all": "allee", "pl": "place",
    "imp": "impasse", "ch": "chemin", "chem": "chemin", "rte": "route", "fbg": "faubourg",
    "sq": "square", "crs": "cours", "res": "residence", "bat": "batiment",
    "apt": "appartement", "appt": "appartement", "n": "number", "no": "number",
    "num": "number", "numero": "number",
}

ADDRESS_EXPANSIONS_BY_COUNTRY: Dict[str, Dict[str, str]] = {
    "france": FRENCH_ADDRESS_EXPANSIONS,
}


def expand_address_tokens(tokens: List[str], country: Optional[str]) -> List[str]:
    table = ADDRESS_EXPANSIONS_BY_COUNTRY.get((country or "").strip().lower(), GENERIC_ADDRESS_EXPANSIONS)
    return [table.get(t, t) for t in tokens]


# --- 2. First-level regions ------------------------------------------------------------

US_STATES = {
    "al": ["alabama"], "ak": ["alaska"], "az": ["arizona"], "ar": ["arkansas"],
    "ca": ["california"], "co": ["colorado"], "ct": ["connecticut"], "de": ["delaware"],
    "dc": ["district of columbia"], "fl": ["florida"], "ga": ["georgia"], "hi": ["hawaii"],
    "id": ["idaho"], "il": ["illinois"], "in": ["indiana"], "ia": ["iowa"], "ks": ["kansas"],
    "ky": ["kentucky"], "la": ["louisiana"], "me": ["maine"], "md": ["maryland"],
    "ma": ["massachusetts"], "mi": ["michigan"], "mn": ["minnesota"], "ms": ["mississippi"],
    "mo": ["missouri"], "mt": ["montana"], "ne": ["nebraska"], "nv": ["nevada"],
    "nh": ["new hampshire"], "nj": ["new jersey"], "nm": ["new mexico"], "ny": ["new york"],
    "nc": ["north carolina"], "nd": ["north dakota"], "oh": ["ohio"], "ok": ["oklahoma"],
    "or": ["oregon"], "pa": ["pennsylvania"], "ri": ["rhode island"], "sc": ["south carolina"],
    "sd": ["south dakota"], "tn": ["tennessee"], "tx": ["texas"], "ut": ["utah"],
    "vt": ["vermont"], "va": ["virginia"], "wa": ["washington"], "wv": ["west virginia"],
    "wi": ["wisconsin"], "wy": ["wyoming"], "pr": ["puerto rico"],
}

INDIA_STATES = {
    "ap": ["andhra pradesh"], "ar": ["arunachal pradesh"], "as": ["assam"], "br": ["bihar"],
    "cg": ["chhattisgarh", "chattisgarh", "ct"], "ga": ["goa"], "gj": ["gujarat"],
    "hr": ["haryana", "hariyana"], "hp": ["himachal pradesh"], "jh": ["jharkhand"],
    "ka": ["karnataka"], "kl": ["kerala", "keralam"], "mp": ["madhya pradesh"],
    "mh": ["maharashtra"], "mn": ["manipur"], "ml": ["meghalaya"], "mz": ["mizoram"],
    "nl": ["nagaland"], "od": ["odisha", "orissa", "orisha", "or"], "pb": ["punjab", "panjab"],
    "rj": ["rajasthan"], "sk": ["sikkim"], "tn": ["tamil nadu", "tamilnadu", "tamizh nadu"],
    "tg": ["telangana", "ts"], "tr": ["tripura"], "up": ["uttar pradesh"],
    "uk": ["uttarakhand", "uttaranchal", "ut"], "wb": ["west bengal", "paschim banga", "pashchim bang"],
    "dl": ["delhi", "dilli", "nct of delhi"], "jk": ["jammu and kashmir", "jammu kashmir"],
    "la": ["ladakh"], "ch": ["chandigarh"], "py": ["puducherry", "pondicherry"],
    "an": ["andaman and nicobar islands"], "ld": ["lakshadweep"],
    "dn": ["dadra and nagar haveli and daman and diu", "dadra and nagar haveli", "daman and diu"],
}

REGIONS_BY_COUNTRY = {"us": US_STATES, "united states": US_STATES, "india": INDIA_STATES}

# Phonetic matching only where region names also appear transliterated from regional
# scripts. For Latin-only names it causes collisions ('mountain' ~ 'montana').
PHONETIC_REGION_COUNTRIES = {"india"}

_MIN_PHONETIC_LEN = 5  # shorter skeletons collide with ordinary words (manpur ~ manipur)


def _build_lookup(table: Dict[str, List[str]], use_phonetic: bool):
    exact, phonetic = {}, {}
    for code, names in table.items():
        exact[code] = code
        for name in names:
            exact[name] = code
            key = phonetic_key(name.replace(" ", ""))
            if use_phonetic and len(key) >= _MIN_PHONETIC_LEN:
                phonetic.setdefault(key, code)
    max_words = max(len(n.split()) for names in table.values() for n in names)
    return exact, phonetic, max_words


_LOOKUPS = {
    country: _build_lookup(table, country in PHONETIC_REGION_COUNTRIES)
    for country, table in REGIONS_BY_COUNTRY.items()
}


def canonicalize_regions(tokens: List[str], country: Optional[str]) -> List[str]:
    """Replace region names (exact or phonetically equal spellings) by their canonical code.

    Longest windows are tried first so 'north carolina' is not split into 'north' + 'ca'.
    Single two-letter tokens are only mapped when they already are a code (identity), so
    ordinary words like 'in' or 'or' inside a street name are left alone.
    """
    lookup = _LOOKUPS.get((country or "").strip().lower())
    if not lookup or not tokens:
        return tokens
    exact, phonetic, max_words = lookup
    out, i, n = [], 0, len(tokens)
    while i < n:
        replaced = False
        for w in range(min(max_words, n - i), 0, -1):
            window = tokens[i:i + w]
            if any(t.isdigit() for t in window):
                continue
            joined = " ".join(window)
            code = exact.get(joined)
            if code is None and len(joined) >= 5:
                code = phonetic.get(phonetic_key(joined.replace(" ", "")))
            if code is not None and (w > 1 or len(joined) > 2):
                out.append(code)
                i += w
                replaced = True
                break
        if not replaced:
            out.append(tokens[i])
            i += 1
    return out
