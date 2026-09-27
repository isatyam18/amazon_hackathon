"""
Stage 3: pair features for (S1, candidate) pairs, computed in bulk.

All string similarities use rapidfuzz's multithreaded pairwise `cpdist`. Token
overlaps are row-wise dot products of the binary token matrices held by the
RecordStore, so a pair costs microseconds, which matters at 50-70M pairs per
split.

Feature groups:
  name     fuzzy ratios on the normalised name (legal/filler tokens removed),
           Jaro-Winkler / partial ratio on the space-free name (domain names,
           joined words), phonetic ratio (transliteration), token Jaccard,
           first-token agreement, lengths.
  address  fuzzy ratios, token Jaccard, numeric-token overlap and conflict,
           house-number equality / prefix agreement ('245' vs '2454'),
           missing-address flags. Address features are NaN, not 0, when a side has
           no address: LightGBM treats NaN as "unknown", which is what it is.
  blocking the Stage 2 retrieval cosines (score_*), exact similarities (sim_*),
           candidate score and per-source rank.
  context  see context.py (query- and target-level competition).

No feature uses the country label, so the model applies unchanged to countries
not seen in training (France).
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from src.blocking.rescore import pair_cosine as pair_dot  # binary rows -> intersection counts
from src.matching.records import RecordStore

# (feature, RecordStore field, scorer, scale); fuzz scorers return 0-100
STRING_FEATURES = [
    ("name_ratio", "name", fuzz.ratio, 100.0),
    ("name_token_sort", "name", fuzz.token_sort_ratio, 100.0),
    ("name_token_set", "name", fuzz.token_set_ratio, 100.0),
    ("name_partial", "name_compact", fuzz.partial_ratio, 100.0),
    ("name_jw", "name_compact", JaroWinkler.normalized_similarity, 1.0),
    ("name_phon_ratio", "name_phon", fuzz.ratio, 100.0),
    ("addr_ratio", "addr", fuzz.ratio, 100.0),
    ("addr_token_set", "addr", fuzz.token_set_ratio, 100.0),
    ("addr_token_sort", "addr", fuzz.token_sort_ratio, 100.0),
    ("addr_partial_token_set", "addr", fuzz.partial_token_set_ratio, 100.0),
]

BLOCKING_COLUMNS = [
    "candidate_score", "source_rank", "score_name", "score_addr", "score_name_addr",
    "sim_name_char", "sim_name_phon", "sim_addr_word", "sim_addr_bigram", "sim_name_addr",
]


def _jaccard(inter: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    union = a + b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, np.nan).astype(np.float32)


def pair_features(
    store: RecordStore,
    qi: np.ndarray,
    ti: np.ndarray,
    extra: Optional[Dict[str, np.ndarray]] = None,
    workers: int = -1,
) -> pd.DataFrame:
    """Features for pairs (store row qi[j], store row ti[j]); `extra` columns are appended as-is."""
    qi = np.asarray(qi, dtype=np.int64)
    ti = np.asarray(ti, dtype=np.int64)
    f: Dict[str, np.ndarray] = {}

    q_has_addr = store.token_counts["addr_word"][qi] > 0
    c_has_addr = store.token_counts["addr_word"][ti] > 0
    both_addr = q_has_addr & c_has_addr

    strings = {}  # materialise each field's Python strings once per call
    for field in dict.fromkeys(f for _, f, _, _ in STRING_FEATURES):
        column = getattr(store, field)
        strings[field] = (column[qi], column[ti])
    for name, field, scorer, scale in STRING_FEATURES:
        a, b = strings[field]
        sim = cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32) / np.float32(scale)
        if field == "addr":
            sim[~both_addr] = np.nan
        f[name] = sim

    # Token overlaps
    for view, prefix in (("name_word", "name"), ("addr_word", "addr"), ("addr_num", "num")):
        m = store.tokens[view]
        inter = pair_dot(m, m, qi, ti)
        a = store.token_counts[view][qi]
        b = store.token_counts[view][ti]
        f[f"{prefix}_inter"] = inter
        f[f"{prefix}_jaccard"] = _jaccard(inter, a, b)
        f[f"{prefix}_count_q"] = a
        f[f"{prefix}_count_c"] = b
    both_num = (f["num_count_q"] > 0) & (f["num_count_c"] > 0)
    f["num_conflict"] = np.where(both_num, (f["num_inter"] == 0).astype(np.float32), np.nan)
    for key in ("addr_inter", "addr_jaccard"):
        f[key][~both_addr] = np.nan

    # House number (first numeric address token)
    hq, hc = store.house[qi], store.house[ti]
    has_both = np.array([bool(a) and bool(b) for a, b in zip(hq, hc)])
    eq = np.array([a == b for a, b in zip(hq, hc)])
    prefix = np.array([bool(a) and bool(b) and (a.startswith(b) or b.startswith(a)) for a, b in zip(hq, hc)])
    f["house_eq"] = np.where(has_both, eq, np.nan).astype(np.float32)
    f["house_prefix"] = np.where(has_both, prefix, np.nan).astype(np.float32)

    # Name structure
    f["first_token_eq"] = ((store.first_token_hash[qi] == store.first_token_hash[ti])
                           & (store.first_token_hash[qi] != -1)).astype(np.float32)
    f["name_len_q"] = store.name_len[qi]
    f["name_len_c"] = store.name_len[ti]
    f["has_addr_q"] = q_has_addr.astype(np.float32)
    f["has_addr_c"] = c_has_addr.astype(np.float32)

    frame = pd.DataFrame(f)
    for key, values in (extra or {}).items():
        frame[key] = np.asarray(values, dtype=np.float32)
    return frame


def feature_names(extra_columns: List[str]) -> List[str]:
    """Column order produced by `pair_features` for a given list of extra columns."""
    names = [n for n, *_ in STRING_FEATURES]
    for prefix in ("name", "addr", "num"):
        names += [f"{prefix}_inter", f"{prefix}_jaccard", f"{prefix}_count_q", f"{prefix}_count_c"]
    names += ["num_conflict", "house_eq", "house_prefix", "first_token_eq", "name_len_q", "name_len_c",
              "has_addr_q", "has_addr_c"]
    return names + list(extra_columns)
