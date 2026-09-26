"""Unit tests for Stage 3 & 4 matching pipeline."""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.matching.features import compute_pair_features, extract_pair_features
from src.matching.threshold import compute_entity_f05, evaluate_f05, find_optimal_threshold
from src.matching.model import train_matcher, MatcherModel
from src.matching.pipeline import generate_matching_results


def test_compute_pair_features():
    f = compute_pair_features(
        q_name="Walmart Supercenter",
        q_core="walmart supercenter",
        q_addr="123 Main Street Suite 100",
        q_zip="94043",
        c_name="Wal-Mart Supercenter Inc",
        c_core="wal-mart supercenter",
        c_addr="123 Main St Ste 100",
        c_zip="94043",
    )
    assert f["feat_name_ratio"] > 0.80
    assert f["feat_core_ratio"] > 0.95
    assert f["feat_core_sort_ratio"] > 0.95
    assert f["feat_common_nums"] == 2.0  # 123 and 100
    assert f["feat_num_conflict"] == 0.0
    assert f["feat_postal_match"] == 1.0


def test_singleton_scoring():
    # True singleton predicted empty -> 1.0
    assert compute_entity_f05(set(), set()) == 1.0
    # True singleton predicted non-empty -> 0.0
    assert compute_entity_f05({"S2-1"}, set()) == 0.0
    # Non-singleton predicted empty -> 0.0
    assert compute_entity_f05(set(), {"S2-1"}) == 0.0

    # Perfect match -> 1.0
    assert compute_entity_f05({"S2-1", "S3-1"}, {"S2-1", "S3-1"}) == 1.0

    # Precision = 2/3, Recall = 2/2 = 1.0 -> F0.5 = (1.25 * 0.667 * 1.0) / (0.25 * 0.667 + 1.0) = 0.714
    score = compute_entity_f05({"S2-1", "S2-2", "S3-1"}, {"S2-1", "S3-1"})
    assert 0.71 < score < 0.72


def test_train_and_predict():
    # Create synthetic candidate pairs
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "feat_name_ratio": np.random.uniform(0, 1, n),
        "feat_addr_ratio": np.random.uniform(0, 1, n),
        "feat_common_nums": np.random.choice([0, 1, 2], n),
        "feat_postal_match": np.random.choice([-1, 0, 1], n),
        "candidate_score": np.random.uniform(0, 1, n),
    })
    # Labels correlate with feat_name_ratio and candidate_score
    y = ((X["feat_name_ratio"] + X["candidate_score"]) > 1.2).astype(int).values

    model = train_matcher(X, y, num_boost_round=20)
    preds = model.predict_proba(X)

    assert len(preds) == n
    assert (preds >= 0.0).all() and (preds <= 1.0).all()


def test_generate_matching_results():
    candidates = pd.DataFrame({
        "query_id": ["S1-1", "S1-1", "S1-2", "S1-3"],
        "candidate_id": ["S2-1", "S3-1", "S2-2", "S2-3"],
    })
    probs = np.array([0.95, 0.40, 0.88, 0.20])
    all_queries = ["S1-1", "S1-2", "S1-3", "S1-4"]  # S1-4 has no candidates

    results = generate_matching_results(candidates, probs, all_queries, threshold=0.70)
    assert len(results) == 4
    assert list(results["source1_entity_id"]) == all_queries

    # S1-1 should have S2-1 (0.95 >= 0.70, S3-1 0.40 dropped)
    r1 = results[results["source1_entity_id"] == "S1-1"]["matched_entity_ids"].values[0]
    assert r1 == "S2-1"

    # S1-2 should have S2-2
    r2 = results[results["source1_entity_id"] == "S1-2"]["matched_entity_ids"].values[0]
    assert r2 == "S2-2"

    # S1-3 dropped (0.20 < 0.70) -> singleton
    r3 = results[results["source1_entity_id"] == "S1-3"]["matched_entity_ids"].values[0]
    assert r3 == ""

    # S1-4 had no candidates -> singleton
    r4 = results[results["source1_entity_id"] == "S1-4"]["matched_entity_ids"].values[0]
    assert r4 == ""
