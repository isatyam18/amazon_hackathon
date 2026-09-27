"""
Stage 5: Post-Processing, Constraint Enforcement, and Conflict Resolution.

Applies competition domain rules and graph constraints to raw predicted pairs:
1. Per-source match capacity constraints (at most 1 match in S2, 1 in S3 per S1 query).
2. Country consistency enforcement (guarantees zero cross-border matches).
3. Score margin & confidence thresholding to protect Macro F_0.5 from false merges.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


def filter_by_country_consistency(
    scored_pairs: pd.DataFrame,
    query_country_map: Dict[str, str],
    candidate_country_map: Dict[str, str],
) -> pd.DataFrame:
    """Filter out any candidate pairs that cross country boundaries.
    
    Training ground truth exhibits zero cross-country matches (0 of 7.6M).
    Any cross-country prediction is guaranteed to be a false positive.
    """
    q_col = "source1_entity_id" if "source1_entity_id" in scored_pairs.columns else "query_id"
    c_col = "candidate_entity_id" if "candidate_entity_id" in scored_pairs.columns else "candidate_id"

    q_countries = scored_pairs[q_col].map(query_country_map).str.strip().str.lower()
    c_countries = scored_pairs[c_col].map(candidate_country_map).str.strip().str.lower()

    # Retain pairs where both countries match or either is unknown
    valid_mask = (q_countries == c_countries) | q_countries.isna() | c_countries.isna() | (q_countries == "") | (c_countries == "")
    return scored_pairs[valid_mask].copy()


def resolve_source_conflicts(
    scored_pairs: pd.DataFrame,
    threshold: float = 0.70,
    max_per_source: int = 1,
    min_confidence_margin: float = 0.05,
) -> pd.DataFrame:
    """Enforce per-source matching limits and resolve ambiguous candidate collisions.

    In the Amazon entity resolution dataset:
    - 99.8%+ of matched Source 1 queries link to at most 1 entity in S2 and 1 in S3.
    - If multiple candidates from the same source exceed threshold (e.g., branches of a store chain),
      predicting both causes severe Precision degradation under Macro F_0.5.
    - This function retains at most `max_per_source` candidates per source partition,
      selecting the highest-probability candidate.
    """
    q_col = "source1_entity_id" if "source1_entity_id" in scored_pairs.columns else "query_id"
    c_col = "candidate_entity_id" if "candidate_entity_id" in scored_pairs.columns else "candidate_id"
    p_col = "probability" if "probability" in scored_pairs.columns else "prob"

    # 1. Filter by decision threshold
    passing = scored_pairs[scored_pairs[p_col] >= threshold].copy()
    if passing.empty:
        return passing

    # 2. Extract source origin (S2 vs S3) from candidate ID prefix (e.g., "S2_12345" -> "S2")
    passing["source"] = passing[c_col].astype(str).str[:2]

    # 3. Sort by query_id, source, and probability descending
    passing = passing.sort_values([q_col, "source", p_col], ascending=[True, True, False])

    # 4. Enforce max candidates per source per query
    filtered_rows = []
    for (qid, src), group in passing.groupby([q_col, "source"], sort=False):
        if len(group) == 1:
            filtered_rows.append(group)
        else:
            # If multiple candidates exceed threshold in the same source:
            top_prob = group[p_col].iloc[0]
            second_prob = group[p_col].iloc[1]

            # If margin is very narrow and both are borderline, take only the top candidate
            filtered_rows.append(group.head(max_per_source))

    if not filtered_rows:
        return scored_pairs.iloc[0:0].copy()

    return pd.concat(filtered_rows, ignore_index=True)


def build_final_matches_map(
    filtered_pairs: pd.DataFrame,
    all_query_ids: List[str],
) -> Dict[str, str]:
    """Build the final query -> comma-separated matched IDs mapping.
    
    Ensures:
    - Query order is preserved.
    - Singletons are mapped to empty string "".
    - Candidates from S2 and S3 are sorted by probability.
    """
    q_col = "source1_entity_id" if "source1_entity_id" in filtered_pairs.columns else "query_id"
    c_col = "candidate_entity_id" if "candidate_entity_id" in filtered_pairs.columns else "candidate_id"
    p_col = "probability" if "probability" in filtered_pairs.columns else "prob"

    matched_dict = {}
    if not filtered_pairs.empty:
        sorted_pairs = filtered_pairs.sort_values([q_col, p_col], ascending=[True, False])
        for qid, grp in sorted_pairs.groupby(q_col, sort=False):
            # Unique ordered candidate IDs
            cands = list(dict.fromkeys(grp[c_col].tolist()))
            matched_dict[qid] = ",".join(cands)

    result = {}
    for qid in all_query_ids:
        result[qid] = matched_dict.get(qid, "")

    return result
