"""
Data Preprocessing and Normalization Module for Business Entity Resolution.

Handles:
- Indic script transliteration (Devanagari, Tamil, Telugu, ... -> Latin; see transliteration.py)
- Unicode normalization and de-accenting (for multilingual data: US, India, France)
- Business name cleaning and legal suffix stripping (US, Indian, and French legal forms)
- Address standardization (country-aware abbreviations, region-name canonicalization)
- Tokenization, postal code extraction, and numerical token isolation
"""

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

from src.regions import GENERIC_ADDRESS_EXPANSIONS, canonicalize_regions, expand_address_tokens
from src.transliteration import transliterate_indic


# Regex for Unicode accents removal
def remove_accents(text: str) -> str:
    """Normalize unicode and strip accents (e.g. Société -> Societe)."""
    if not text:
        return ""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


# Standard legal suffixes across target geographies (US, India, France)
LEGAL_SUFFIXES = [
    # India
    r"\bpvt\s+ltd\b",
    r"\bprivate\s+limited\b",
    r"\bltd\b",
    r"\blimited\b",
    r"\bllp\b",
    r"\bopc\b",
    r"\bco\s+operative\b",
    r"\bcoop\b",
    # US
    r"\bincorporated\b",
    r"\binc\b",
    r"\bcorporation\b",
    r"\bcorp\b",
    r"\bllc\b",
    r"\bl\.l\.c\b",
    r"\bl\.p\b",
    r"\blp\b",
    r"\bco\b",
    r"\bcompany\b",
    # France (Test set)
    r"\bsarl\b",
    r"\bsas\b",
    r"\bsasu\b",
    r"\bsa\b",
    r"\beurl\b",
    r"\bsnc\b",
    r"\bsci\b",
]

# Precompiled regex for stripping legal suffixes
LEGAL_SUFFIX_PATTERN = re.compile(
    r"\s*(?:" + "|".join(LEGAL_SUFFIXES) + r")\s*$",
    re.IGNORECASE
)

# Address abbreviation and region tables are country-aware (see regions.py)
ADDRESS_EXPANSIONS = GENERIC_ADDRESS_EXPANSIONS


def clean_text(text: Optional[str]) -> str:
    """Base cleaner: transliterate Indic scripts, de-accent, lowercase, collapse dot acronyms,
    standardize symbols, strip extra whitespace."""
    if not text or not isinstance(text, str):
        return ""
    # Must run before remove_accents: NFKD + combining-mark removal would strip viramas/nuktas
    text = transliterate_indic(text)
    text = remove_accents(text).lower()
    # Collapse dotted abbreviations like s.a.r.l. -> sarl, l.l.c. -> llc
    text = re.sub(r"(?<=\b[a-z])\.(?=[a-z]\b)", "", text)
    # Replace & with and
    text = re.sub(r"&", " and ", text)
    # Remove special characters, keep alphanumeric and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_business_name(name: Optional[str]) -> Tuple[str, str, str]:
    """Clean business name and extract legal suffix.

    Returns:
        clean_name: fully normalized name with standard spacing
        core_name: name with legal suffix stripped (e.g. 'Walmart Inc' -> 'walmart')
        suffix_extracted: the legal suffix if found, otherwise empty string
    """
    cleaned = clean_text(name)
    if not cleaned:
        return "", "", ""

    # Check for legal suffix
    match = LEGAL_SUFFIX_PATTERN.search(cleaned)
    if match:
        suffix = match.group(0).strip()
        core = cleaned[: match.start()].strip()
    else:
        suffix = ""
        core = cleaned

    return cleaned, core, suffix


def clean_address(address: Optional[str], country: Optional[str] = None) -> str:
    """Standardize street types, abbreviations and region names in an address.

    Expansions depend on the country label ('st' is Street in the US but Saint in
    France); region names ('north carolina', 'maharashtra', transliterated
    'maharashtr') become one canonical code per country. Unknown countries get
    the generic table and no region mapping.
    """
    cleaned = clean_text(address)
    if not cleaned:
        return ""
    tokens = expand_address_tokens(cleaned.split(), country)
    tokens = canonicalize_regions(tokens, country)
    return " ".join(tokens)


def extract_postal_code(address: Optional[str], country: str) -> Optional[str]:
    """Extract standard postal/PIN code based on country."""
    if not address or not isinstance(address, str):
        return None

    c = country.strip().lower()
    if c == "india":
        # 6 digit Indian PIN code (cannot start with 0)
        match = re.search(r"\b([1-9][0-9]{5})\b", address)
        return match.group(1) if match else None
    elif c in ("us", "united states"):
        # 5 digit US ZIP code
        match = re.search(r"\b([0-9]{5})(?:-[0-9]{4})?\b", address)
        return match.group(1) if match else None
    elif c == "france":
        # 5 digit French postal code
        match = re.search(r"\b([0-9]{5})\b", address)
        return match.group(1) if match else None
    else:
        # Generic 5 or 6 digit code fallback
        match = re.search(r"\b([0-9]{5,6})\b", address)
        return match.group(1) if match else None


def extract_numeric_tokens(text: Optional[str]) -> Set[str]:
    """Extract standalone numeric tokens (building numbers, suites, road numbers)."""
    if not text:
        return set()
    return set(re.findall(r"\b\d+\b", text))


def preprocess_record(record: Dict[str, str]) -> Dict[str, any]:
    """Preprocess a single entity record dictionary."""
    entity_id = record.get("entity_id", "")
    raw_name = record.get("business_name", "")
    raw_addr = record.get("business_address", "")
    country = str(record.get("country", "")).strip().lower()

    clean_name, core_name, suffix = clean_business_name(raw_name)
    cleaned_addr = clean_address(raw_addr, country)
    postal_code = extract_postal_code(raw_addr, country)

    return {
        "entity_id": entity_id,
        "clean_name": clean_name,
        "core_name": core_name,
        "legal_suffix": suffix,
        "clean_address": cleaned_addr,
        "country": country,
        "postal_code": postal_code or "",
    }


def preprocess_dataframe(df: "pd.DataFrame") -> "pd.DataFrame":
    """Preprocess an entire pandas DataFrame chunk efficiently.

    Expects columns: ['entity_id', 'business_name', 'business_address', 'country']
    Returns DataFrame with normalized columns.
    """
    records = df.to_dict("records")
    processed = [preprocess_record(r) for r in records]
    import pandas as pd
    return pd.DataFrame(processed)

