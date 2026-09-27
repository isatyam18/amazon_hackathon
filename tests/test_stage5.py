"""
Unit tests for Stage 5 Post-Processing, Consistency, and Candidate Optimization.
"""

import os
import pandas as pd
import pytest

from src.postprocessing.consistency import (
    filter_by_country_consistency,
    resolve_source_conflicts,
    build_final_matches_map,
)
from src.postprocessing.candidate_optimizer import (
    optimize_candidate_set,
    compute_candidate_efficiency_metrics,
)


def test_country_consistency_filter():
    pairs = pd.DataFrame([
        {"source1_entity_id": "S1_1", "candidate_entity_id": "S2_1"},
        {"source1_entity_id": "S1_2", "candidate_entity_id": "S2_2"},
    ])
    q_country = {"S1_1": "us", "S1_2": "india"}
    c_country = {"S2_1": "us", "S2_2": "us"}  # S1_2 is india vs us (cross-country)

    filtered = filter_by_country_consistency(pairs, q_country, c_country)
    assert len(filtered) == 1
    assert filtered.iloc[0]["source1_entity_id"] == "S1_1"


def test_resolve_source_conflicts():
    pairs = pd.DataFrame([
        {"source1_entity_id": "S1_1", "candidate_entity_id": "S2_1", "probability": 0.95},
        {"source1_entity_id": "S1_1", "candidate_entity_id": "S2_2", "probability": 0.80},  # Duplicate S2
        {"source1_entity_id": "S1_1", "candidate_entity_id": "S3_1", "probability": 0.90},  # S3 match
        {"source1_entity_id": "S1_2", "candidate_entity_id": "S2_3", "probability": 0.40},  # Below threshold
    ])

    resolved = resolve_source_conflicts(pairs, threshold=0.70, max_per_source=1)
    assert len(resolved) == 2  # S2_1 and S3_1 for S1_1
    assert set(resolved["candidate_entity_id"]) == {"S2_1", "S3_1"}


def test_candidate_optimizer_preserves_matches():
    candidates = pd.DataFrame([
        {"source1_entity_id": "S1_1", "candidate_entity_id": "S2_1", "candidate_score": 0.95},
        {"source1_entity_id": "S1_1", "candidate_entity_id": "S2_2", "candidate_score": 0.30},
        {"source1_entity_id": "S1_1", "candidate_entity_id": "S2_3", "candidate_score": 0.20},
    ])
    matches_map = {"S1_1": "S2_1"}
    all_queries = ["S1_1", "S1_2"]  # S1_2 is singleton

    opt = optimize_candidate_set(candidates, matches_map, all_queries, max_candidates_per_source=1)
    assert len(opt) == 2
    row1 = opt[opt["source1_entity_id"] == "S1_1"].iloc[0]
    assert "S2_1" in row1["candidate_entity_ids"]

    row2 = opt[opt["source1_entity_id"] == "S1_2"].iloc[0]
    assert row2["candidate_entity_ids"] == ""
