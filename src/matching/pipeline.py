"""
Stage 4/5: submission table helpers.

Outputs matching_results.tsv compliant with the official competition rules. The
full train/evaluate/infer workflow lives in workflow.py and scripts/run_matching.py.
"""

import os
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from .model import MatcherModel


def generate_matching_results(
    candidates_df: pd.DataFrame,
    probabilities: np.ndarray,
    all_query_ids: List[str],
    threshold: float = 0.70,
) -> pd.DataFrame:
    """Final matching table from pair probabilities and a global threshold.

    Every query in `all_query_ids` gets exactly one row (in that order);
    matched ids are de-duplicated and ordered by descending probability, and
    the list is empty when nothing passes the threshold. The production path
    (scripts/run_matching.py) uses the tuned decision rule in decision.py
    instead of a single threshold.
    """
    q_col = "source1_entity_id" if "source1_entity_id" in candidates_df.columns else "query_id"
    c_col = "candidate_entity_id" if "candidate_entity_id" in candidates_df.columns else "candidate_id"
    df = pd.DataFrame({
        "query_id": candidates_df[q_col].to_numpy(),
        "candidate_id": candidates_df[c_col].to_numpy(),
        "prob": np.asarray(probabilities),
    })
    passing = df[df["prob"] >= threshold].sort_values(["query_id", "prob"], ascending=[True, False])
    passing = passing.drop_duplicates(["query_id", "candidate_id"])
    matched = passing.groupby("query_id", sort=False)["candidate_id"].agg(",".join)
    ids = pd.Index(all_query_ids)
    return pd.DataFrame({
        "source1_entity_id": ids,
        "matched_entity_ids": matched.reindex(ids).fillna("").to_numpy(),
    })


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

