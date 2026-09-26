"""
Stage 3: Fine-grained feature extraction for candidate pairs.

Extracts discriminative string, numeric, and structural similarity features
from candidate pairs using RapidFuzz and Stage 2 retrieval scores.
"""

import re
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from rapidfuzz import fuzz


def extract_numbers(text: Optional[str]) -> Set[str]:
    """Extract standalone numeric tokens (building numbers, suites, road numbers)."""
    if not text or not isinstance(text, str):
        return set()
    return set(re.findall(r"\b\d+\b", text))


def compute_pair_features(
    q_name: str,
    q_core: str,
    q_addr: str,
    q_zip: str,
    c_name: str,
    c_core: str,
    c_addr: str,
    c_zip: str,
) -> Dict[str, float]:
    """Compute detailed string, token, and numeric features between two records."""
    q_name = str(q_name or "").strip()
    c_name = str(c_name or "").strip()
    q_core = str(q_core or q_name).strip()
    c_core = str(c_core or c_name).strip()
    q_addr = str(q_addr or "").strip()
    c_addr = str(c_addr or "").strip()
    q_zip = str(q_zip or "").strip()
    c_zip = str(c_zip or "").strip()

    # 1. Exact string matches
    exact_name = 1.0 if q_name and q_name == c_name else 0.0
    exact_core = 1.0 if q_core and q_core == c_core else 0.0

    # 2. RapidFuzz Name Similarities (0.0 to 1.0 scale)
    name_ratio = fuzz.ratio(q_name, c_name) / 100.0 if q_name and c_name else 0.0
    name_sort_ratio = fuzz.token_sort_ratio(q_name, c_name) / 100.0 if q_name and c_name else 0.0
    name_set_ratio = fuzz.token_set_ratio(q_name, c_name) / 100.0 if q_name and c_name else 0.0

    core_ratio = fuzz.ratio(q_core, c_core) / 100.0 if q_core and c_core else 0.0
    core_sort_ratio = fuzz.token_sort_ratio(q_core, c_core) / 100.0 if q_core and c_core else 0.0

    # 3. Name Length differences
    len_q, len_c = len(q_name), len(c_name)
    len_diff = abs(len_q - len_c)
    len_ratio = min(len_q, len_c) / max(len_q, len_c) if max(len_q, len_c) > 0 else 0.0

    # 4. Address Similarities
    has_addr_both = 1.0 if q_addr and c_addr else 0.0
    if has_addr_both:
        addr_ratio = fuzz.ratio(q_addr, c_addr) / 100.0
        addr_set_ratio = fuzz.token_set_ratio(q_addr, c_addr) / 100.0
    else:
        addr_ratio = 0.0
        addr_set_ratio = 0.0

    # 5. Numeric tokens (door numbers, plots, floors)
    nums_q = extract_numbers(q_addr)
    nums_c = extract_numbers(c_addr)
    common_nums = len(nums_q & nums_c)

    # Conflict: both have numbers, but share 0 numbers (strong negative signal)
    if nums_q and nums_c and common_nums == 0:
        num_conflict = 1.0
    else:
        num_conflict = 0.0

    # 6. Postal code match
    if q_zip and c_zip:
        postal_match = 1.0 if q_zip == c_zip else -1.0
    else:
        postal_match = 0.0

    return {
        "feat_exact_name": exact_name,
        "feat_exact_core": exact_core,
        "feat_name_ratio": name_ratio,
        "feat_name_sort_ratio": name_sort_ratio,
        "feat_name_set_ratio": name_set_ratio,
        "feat_core_ratio": core_ratio,
        "feat_core_sort_ratio": core_sort_ratio,
        "feat_len_diff": float(len_diff),
        "feat_len_ratio": float(len_ratio),
        "feat_has_addr_both": has_addr_both,
        "feat_addr_ratio": addr_ratio,
        "feat_addr_set_ratio": addr_set_ratio,
        "feat_common_nums": float(common_nums),
        "feat_num_conflict": num_conflict,
        "feat_postal_match": postal_match,
    }


def extract_pair_features(
    candidates_df: pd.DataFrame,
    preprocessed_records: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    """Extract full feature set combining Stage 2 scores + fine-grained metrics.

    Args:
        candidates_df: DataFrame with candidate pairs. Supports either
            ['source1_entity_id', 'candidate_entity_id'] or ['query_id', 'candidate_id'].
        preprocessed_records: mapping from entity_id -> preprocessed dict
            containing 'clean_name', 'core_name', 'clean_address', 'postal_code'

    Returns:
        DataFrame of feature rows matching candidate rows.
    """
    rows = []
    q_col = "source1_entity_id" if "source1_entity_id" in candidates_df.columns else "query_id"
    c_col = "candidate_entity_id" if "candidate_entity_id" in candidates_df.columns else "candidate_id"

    q_ids = candidates_df[q_col].values
    c_ids = candidates_df[c_col].values

    for i in range(len(candidates_df)):
        qid = q_ids[i]
        cid = c_ids[i]

        qr = preprocessed_records.get(qid, {})
        cr = preprocessed_records.get(cid, {})

        f = compute_pair_features(
            q_name=qr.get("clean_name", ""),
            q_core=qr.get("core_name", ""),
            q_addr=qr.get("clean_address", ""),
            q_zip=qr.get("postal_code", ""),
            c_name=cr.get("clean_name", ""),
            c_core=cr.get("core_name", ""),
            c_addr=cr.get("clean_address", ""),
            c_zip=cr.get("postal_code", ""),
        )
        rows.append(f)

    feat_df = pd.DataFrame(rows)

    # Bring over all Stage 2 precomputed features if available
    stage2_cols = [c for c in candidates_df.columns if c.startswith("score_") or c.startswith("sim_")]
    stage2_cols.extend(["candidate_score", "source_rank"])
    for col in set(stage2_cols):
        if col in candidates_df.columns:
            feat_df[col] = candidates_df[col].astype(float).values
        else:
            feat_df[col] = 0.0

    # Source indicator (S2 vs S3)
    if "source" in candidates_df.columns:
        feat_df["is_s2"] = (candidates_df["source"] == "S2").astype(float).values
    else:
        feat_df["is_s2"] = pd.Series(c_ids).str.startswith("S2-").astype(float).values

    return feat_df


def extract_features_dataframe(
    candidates_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convenience wrapper indexing source dataframes into preprocessed_records."""
    records = {}

    def add_records(df):
        for _, row in df.iterrows():
            eid = row.get("entity_id", "")
            if eid:
                records[eid] = {
                    "clean_name": row.get("clean_name", row.get("business_name", "")),
                    "core_name": row.get("core_name", row.get("clean_name", "")),
                    "clean_address": row.get("clean_address", row.get("business_address", "")),
                    "postal_code": row.get("postal_code", ""),
                }

    add_records(s1_df)
    add_records(s2_df)
    add_records(s3_df)

    return extract_pair_features(candidates_df, records)
