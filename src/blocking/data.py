"""
Loading of Stage 1 (preprocessed) sources and ground truth for the blocking stage.
"""

import os
from typing import Iterable, List, Optional

import pandas as pd

BLOCKING_COLUMNS = ["entity_id", "clean_name", "clean_address", "country"]


def preprocessed_path(preprocessed_dir: str, split: str, source: int) -> str:
    """Path of a Stage 1 output file; parquet is preferred over TSV when both exist."""
    stem = os.path.join(preprocessed_dir, f"{split}_source{source}")
    for ext in (".parquet", ".tsv"):
        if os.path.exists(stem + ext):
            return stem + ext
    raise FileNotFoundError(
        f"No preprocessed file for {split}_source{source} in {preprocessed_dir}. Run Stage 1 first:\n"
        f"  python scripts/preprocess_dataset.py --split {split} --format parquet --n-jobs 8"
    )


def load_source(
    preprocessed_dir: str,
    split: str,
    source: int,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load one preprocessed source with string columns ('' for missing values)."""
    columns = columns or BLOCKING_COLUMNS
    path = preprocessed_path(preprocessed_dir, split, source)
    if path.endswith(".parquet"):
        df = pd.read_parquet(path, columns=columns)
    else:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=columns)
    df = df.fillna("")
    # Country is an open set of labels: normalise only whitespace/case, never filter
    df["country"] = df["country"].astype(str).str.strip().str.lower()
    return df.reset_index(drop=True)


def load_ground_truth(path: str) -> pd.DataFrame:
    """Ground truth as a DataFrame: source1_entity_id, matched_entity_ids (''=singleton)."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def explode_id_lists(df: pd.DataFrame, id_col: str, list_col: str, out_col: str) -> pd.DataFrame:
    """One row per (S1 id, listed id) from a comma-separated id-list column."""
    ex = df[[id_col, list_col]].copy()
    ex[out_col] = ex[list_col].fillna("").astype(str).str.split(",")
    ex = ex.explode(out_col)
    ex[out_col] = ex[out_col].str.strip()
    ex = ex[ex[out_col].notna() & (ex[out_col] != "")]
    return ex[[id_col, out_col]].reset_index(drop=True)


def ground_truth_pairs(gt: pd.DataFrame) -> pd.DataFrame:
    """Exploded ground truth: source1_entity_id, candidate_entity_id."""
    return explode_id_lists(gt, "source1_entity_id", "matched_entity_ids", "candidate_entity_id")


def read_candidate_tsv(path: str) -> pd.DataFrame:
    """Exploded pairs from a candidate_pairs.tsv-format file (source1_entity_id, candidate_entity_id)."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return explode_id_lists(df, "source1_entity_id", "candidate_entity_ids", "candidate_entity_id")


def read_id_set(path: str, column: str = "source1_entity_id") -> set:
    """Set of ids from one column of a TSV (e.g. the S1 ids of a validation ground truth)."""
    return set(pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=[column])[column])


def subset_ids(df: pd.DataFrame, ids: Iterable[str]) -> pd.DataFrame:
    ids = set(ids)
    return df[df["entity_id"].isin(ids)].reset_index(drop=True)
