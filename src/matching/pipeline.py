"""
Stage 4: End-to-end matching pipeline and submission generator.

Outputs matching_results.tsv compliant with official competition rules.
"""

import os
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from .features import extract_pair_features
from .model import MatcherModel
from .threshold import evaluate_f05


def generate_matching_results(
    candidates_df: pd.DataFrame,
    probabilities: np.ndarray,
    all_query_ids: List[str],
    threshold: float = 0.70,
) -> pd.DataFrame:
    """Generate final matching results table.

    Every query in all_query_ids has exactly one row.
    If no candidate passes the threshold, matched_entity_ids is empty.
    """
    q_col = "source1_entity_id" if "source1_entity_id" in candidates_df.columns else "query_id"
    c_col = "candidate_entity_id" if "candidate_entity_id" in candidates_df.columns else "candidate_id"
    df = pd.DataFrame({
        "query_id": candidates_df[q_col].values,
        "candidate_id": candidates_df[c_col].values,
        "prob": probabilities,
    })

    # Filter by threshold
    passing = df[df["prob"] >= threshold]

    # Group by query and collect sorted matches
    matched_map = {}
    if not passing.empty:
        # Sort by query_id and prob descending
        passing_sorted = passing.sort_values(["query_id", "prob"], ascending=[True, False])
        for qid, group in passing_sorted.groupby("query_id"):
            # Deduplicate while preserving rank order
            seen = set()
            unique_matches = []
            for cid in group["candidate_id"]:
                if cid not in seen:
                    seen.add(cid)
                    unique_matches.append(cid)
            matched_map[qid] = ",".join(unique_matches)

    # Build final rows for every query ID in original order
    rows = []
    for qid in all_query_ids:
        rows.append({
            "source1_entity_id": qid,
            "matched_entity_ids": matched_map.get(qid, ""),
        })

    return pd.DataFrame(rows)


def write_submission(df: pd.DataFrame, output_path: str) -> None:
    """Save submission TSV with single tab delimiter, no quoting."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Saved matching results to: {output_path} ({len(df):,} queries)")


class MatchingPipeline:
    """End-to-end Stage 3 & 4 pipeline runner."""

    def __init__(self, model: Optional[MatcherModel] = None, threshold: float = 0.70):
        self.model = model
        self.threshold = threshold

    def predict(
        self,
        candidates_df: pd.DataFrame,
        features_df: pd.DataFrame,
        all_query_ids: List[str],
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """Run model inference and generate matching_results DataFrame."""
        if self.model is None:
            raise ValueError("Model is not initialized.")
        t = threshold if threshold is not None else self.threshold
        probs = self.model.predict_proba(features_df)
        return generate_matching_results(candidates_df, probs, all_query_ids, threshold=t)

