"""Unit tests for Stages 3-5 (matching). Run: python -m pytest tests/test_matching.py -q"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.matching.context import context_features
from src.matching.decision import (
    Calibrator,
    DecisionRule,
    apply_rule,
    enforce_exclusivity,
    owner_normalize,
    select_expected_f,
)
from src.matching.model import train_matcher
from src.matching.pipeline import generate_matching_results
from src.matching.records import RecordStore
from src.matching.features import pair_features
from src.matching.threshold import compute_entity_f05, per_query_f05
from src.preprocessing import preprocess_dataframe


# --------------------------------------------------------------------------- metric

def test_singleton_scoring():
    assert compute_entity_f05(set(), set()) == 1.0
    assert compute_entity_f05({"S2-1"}, set()) == 0.0
    assert compute_entity_f05(set(), {"S2-1"}) == 0.0
    assert compute_entity_f05({"S2-1", "S3-1"}, {"S2-1", "S3-1"}) == 1.0
    # Problem-statement example: P = 2/3, R = 1 -> 0.714
    assert compute_entity_f05({"S2-1", "S2-2", "S3-1"}, {"S2-1", "S3-1"}) == pytest.approx(0.7142857, abs=1e-6)


def test_vectorised_metric_equals_reference():
    rng = np.random.default_rng(0)
    n_q, pool = 300, 12
    truth = {q: set(rng.choice(pool, rng.integers(0, 4), replace=False)) for q in range(n_q)}
    pred = {q: set(rng.choice(pool, rng.integers(0, 4), replace=False)) for q in range(n_q)}
    tq = np.array([q for q, s in truth.items() for _ in s], np.int64)
    tk = np.array([(q << 32) | t for q, s in truth.items() for t in s], np.int64)
    pq_ = np.array([q for q, s in pred.items() for _ in s], np.int64)
    pk = np.array([(q << 32) | t for q, s in pred.items() for t in s], np.int64)
    fast = per_query_f05(n_q, pq_, pk, tq, tk)
    ref = [compute_entity_f05(pred[q], truth[q]) for q in range(n_q)]
    np.testing.assert_allclose(fast, ref, atol=1e-12)


# --------------------------------------------------------------------------- decision

def test_expected_f_selection():
    query = np.array([0, 0, 0, 1, 1, 2])
    p = np.array([0.95, 0.9, 0.1, 0.05, 0.02, 0.6])
    sel = select_expected_f(query, p, beta=0.5)
    assert list(sel) == [True, True, False, False, False, True]  # top-2 / likely singleton / single


def test_owner_normalize_and_exclusivity():
    target = np.array([7, 7, 8])
    p = np.array([0.9, 0.6, 0.5])
    np.testing.assert_allclose(owner_normalize(target, p), [0.6, 0.4, 0.5])
    kept = enforce_exclusivity(target, p, np.array([True, True, True]))
    assert list(kept) == [True, False, True]  # record 7 stays with its most probable S1


def test_apply_rule_threshold_and_calibrator_roundtrip():
    rule = DecisionRule("threshold", threshold=0.5, owner_normalize=False, exclusive=False)
    assert list(apply_rule(rule, np.array([0, 1]), np.array([5, 6]), np.array([0.7, 0.3]))) == [True, False]
    cal = Calibrator().fit(np.linspace(0, 1, 200), (np.linspace(0, 1, 200) > 0.5).astype(float))
    cal2 = Calibrator.from_dict(cal.to_dict())
    np.testing.assert_allclose(cal.transform([0.2, 0.9]), cal2.transform([0.2, 0.9]))


# --------------------------------------------------------------------------- features

@pytest.fixture(scope="module")
def store():
    raw = pd.DataFrame([
        ["S1-1", "Complete Property Solutions LLC", "10528 Cedar Falls Loop, Hillsboro, OR", "US"],
        ["S2-1", "LLC Complete Property Soluiscos", "#10528 Cedar Falls Loop, Hillsboro, Oregon", "US"],
        ["S2-2", "Complete Property Solutions LLC", "", "US"],
        ["S3-1", "Unrelated Bakery", "77 Elm St, Dayton, OH", "US"],
    ], columns=["entity_id", "business_name", "business_address", "country"])
    return RecordStore(preprocess_dataframe(raw), n_jobs=1)


def test_pair_features(store):
    f = pair_features(store, np.array([0, 0, 0]), np.array([1, 2, 3]))
    assert f.loc[0, "house_eq"] == 1.0 and f.loc[0, "addr_jaccard"] > 0.8  # 'oregon' == 'or' after Stage 1
    assert f.loc[1, "name_ratio"] == 1.0
    assert np.isnan(f.loc[1, "addr_ratio"]) and f.loc[1, "has_addr_c"] == 0.0  # missing address -> NaN
    assert f.loc[2, "name_token_set"] < 0.5 and f.loc[2, "house_eq"] == 0.0


def test_context_features_margins():
    q = np.array([0, 0, 1])
    t = np.array([5, 6, 5])
    ctx = context_features(q, t, np.array([0, 0, 0]), np.array([3.0, 1.0, 2.0]), np.array([1.0, 0.5, 1.0]))
    np.testing.assert_allclose(ctx["q_margin"], [2.0, -2.0, np.nan])
    np.testing.assert_allclose(ctx["t_margin"], [1.0, np.nan, -1.0])  # query 0 is the best S1 for record 5
    np.testing.assert_allclose(ctx["t_competitors"], [2, 1, 2])


# --------------------------------------------------------------------------- model & submission

def test_train_and_predict():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"a": rng.uniform(0, 1, 400), "b": rng.uniform(0, 1, 400)})
    y = ((X["a"] + X["b"]) > 1.2).astype(int).values
    model = train_matcher(X, y, num_boost_round=20, params={"min_child_samples": 5})
    preds = model.predict_proba(X)
    assert len(preds) == 400 and (preds >= 0).all() and (preds <= 1).all()


def test_generate_matching_results():
    candidates = pd.DataFrame({"query_id": ["S1-1", "S1-1", "S1-2", "S1-3"],
                               "candidate_id": ["S2-1", "S3-1", "S2-2", "S2-3"]})
    results = generate_matching_results(candidates, np.array([0.95, 0.40, 0.88, 0.20]),
                                        ["S1-1", "S1-2", "S1-3", "S1-4"], threshold=0.70)
    assert list(results["source1_entity_id"]) == ["S1-1", "S1-2", "S1-3", "S1-4"]
    assert list(results["matched_entity_ids"]) == ["S2-1", "S2-2", "", ""]


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_submission", os.path.join(ROOT, "student_resource", "utils", "validate_submission.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_end_to_end_outputs_pass_validator(tmp_path):
    """Blocking -> candidate parquet -> per-country graphs -> model -> rule -> both TSVs -> official validator."""
    from src.blocking import BlockingConfig, BlockingPipeline
    from src.blocking.data import ground_truth_pairs
    from src.matching.workflow import (CandidateGraph, Partition, PartitionResult, Universe, candidate_countries,
                                       evaluate, predict_and_write, select_all, truth_for)

    rows, gt_rows = [], []
    for i in range(40):
        country = "US" if i < 25 else "France"
        rows.append([f"S1-{i}", f"Alpha{i} Beta{i} Traders", f"{100 + i} Main Street, Town{i}", country])
        rows.append([f"S2-{i}", f"Alpha{i} Beta{i} Traders LLC", f"{100 + i} Main St, Town{i}", country])
        rows.append([f"S3-{i}", f"Gamma{i} Unrelated", f"{900 + i} Other Road, City{i}", country])
        gt_rows.append([f"S1-{i}", f"S2-{i}"])
    rows.append(["S1-99", "Lonely Entity", "1 Nowhere Road", "Japan"])  # no targets in its country
    gt_rows.append(["S1-99", ""])
    raw = pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])
    pre = preprocess_dataframe(raw)
    src_of = pre["entity_id"].str[:2]
    pre_dir = tmp_path / "pre"
    pre_dir.mkdir()
    for n in (1, 2, 3):
        pre[src_of == f"S{n}"].to_parquet(pre_dir / f"toy_source{n}.parquet", index=False)

    cands = BlockingPipeline(BlockingConfig(n_jobs=1, n_threads=1, hash_bits=18), log=lambda *_: None).run(
        pre[src_of == "S1"].reset_index(drop=True),
        {"S2": pre[src_of == "S2"].reset_index(drop=True), "S3": pre[src_of == "S3"].reset_index(drop=True)})
    parquet = tmp_path / "cands.parquet"
    cands.write_parquet(str(parquet))

    gt = pd.DataFrame(gt_rows, columns=["source1_entity_id", "matched_entity_ids"])
    universe = Universe.load(str(pre_dir), "toy")
    results, X_all, y_all = [], [], []
    for country in candidate_countries(str(parquet)):
        graph = CandidateGraph.load(str(parquet), universe, country)
        tq, tkey = truth_for(graph, ground_truth_pairs(gt))
        part = Partition.build(graph, str(pre_dir), "toy", n_jobs=1)
        res = PartitionResult(country, graph.q, graph.t, np.zeros(len(graph), np.float32), tq, tkey, graph.query_rows())
        X_all.append(part.features(np.arange(len(graph))))
        y_all.append(res.labels())
        results.append(res)
    model = train_matcher(pd.concat(X_all), np.concatenate(y_all).astype(int), num_boost_round=30,
                          params={"min_child_samples": 2})
    for res, X in zip(results, X_all):
        res.prob = model.predict_proba(X).astype(np.float32)
    rule = DecisionRule()
    report = evaluate(results, select_all(rule, results, [r.prob for r in results]), len(universe),
                      np.ones(len(universe), bool), universe.s1_country)
    assert report["queries"] == 41 and report["macro_f05"] > 0.9
    assert set(report["macro_f05_by_country"]) == {"us", "france", "japan"}

    matching, candidates = tmp_path / "matching_results.tsv", tmp_path / "candidate_pairs.tsv"
    stats = predict_and_write(str(parquet), str(pre_dir), "toy", [model], Calibrator(), rule,
                              str(matching), str(candidates), n_jobs=1, log=lambda *_: None)
    assert stats["s1_without_candidates"] == 1

    validator = _load_validator()
    required = set(pre.loc[src_of == "S1", "entity_id"])
    valid_ids = set(pre.loc[src_of != "S1", "entity_id"])
    errors = []
    m = validator.validate_id_list_file(str(matching), validator.MATCHING_HEADER, "matched_entity_ids",
                                        required, valid_ids, errors)
    c = validator.validate_id_list_file(str(candidates), validator.CANDIDATE_HEADER, "candidate_entity_ids",
                                        required, valid_ids, errors)
    assert errors == []
    assert all(m[q] <= c[q] for q in m)                      # matches are a subset of candidates
    assert m["S1-99"] == set() and c["S1-99"] == set()        # no candidates -> empty rows in both files

    # Feature cache: exact at the models' precision, records never rebuilt, write-through while scoring
    from src.matching.workflow import partition_cache
    country = candidate_countries(str(parquet))[0]
    graph = CandidateGraph.load(str(parquet), universe, country)
    idx_all = np.arange(len(graph))
    live32 = Partition.build(graph, str(pre_dir), "toy", n_jobs=1).features(idx_all)
    live16 = Partition.build(graph, str(pre_dir), "toy", n_jobs=1, precision="float16").features(idx_all)
    cache = partition_cache(str(tmp_path / "cache"), str(parquet), str(pre_dir), "toy", graph)
    assert not cache.complete
    Partition.build(graph, str(pre_dir), "toy", n_jobs=1, cache=cache).ensure_cache(log=lambda *_: None)
    assert cache.complete
    cached_part = Partition.build(graph, str(pre_dir), "toy", n_jobs=1, cache=cache, precision="float16")
    cached = cached_part.features(idx_all)
    assert cached_part._store is None                          # served from disk
    assert list(cached.columns) == list(live32.columns)
    np.testing.assert_array_equal(cached.to_numpy(), live16.to_numpy())   # bit-identical to training input
    # float32 models ignore the float16 cache and are scored on exact live values
    exact = Partition.build(graph, str(pre_dir), "toy", n_jobs=1, cache=cache).features(idx_all)
    np.testing.assert_array_equal(exact.to_numpy(), live32.to_numpy())

    class Constant:
        def predict_proba(self, X):
            return np.full(len(X), 0.5)

    cache2 = partition_cache(str(tmp_path / "cache2"), str(parquet), str(pre_dir), "toy", graph)
    Partition.build(graph, str(pre_dir), "toy", n_jobs=1, cache=cache2).score(
        lambda idx: [(np.ones(len(idx), bool), [Constant()])], chunk=7, log=lambda *_: None)
    assert cache2.complete                                     # filled while scoring


def test_stacking_learns_competition_and_stays_out_of_fold():
    from src.matching.stacking import apply_stack, prob_context_features, train_stack
    from src.matching.workflow import PartitionResult, pair_key

    rng = np.random.default_rng(0)
    n_q, per_q = 4000, 6
    q = np.repeat(np.arange(n_q), per_q)
    t = rng.integers(0, 6000, len(q))  # shared records -> competition between queries
    true_t = t[np.arange(n_q) * per_q]  # first candidate of each query is its true record
    truth_key = pair_key(np.arange(n_q), true_t)
    label = np.isin(pair_key(q, t), truth_key)
    p = np.clip(np.where(label, 0.7, 0.3) + rng.normal(0, 0.2, len(q)), 0, 1).astype(np.float32)
    res = PartitionResult("x", q.astype(np.int32), t.astype(np.int32), p, np.arange(n_q), truth_key, np.arange(n_q))
    folds = (np.arange(n_q) % 2).astype(np.int8)

    ctx = prob_context_features(res.q, res.t, res.prob)
    assert list(ctx.columns)[:2] == ["p", "logit_p"] and len(ctx) == len(q)
    models, stacked = train_stack([res], folds, n_train_queries=n_q, num_boost_round=50, log=lambda *_: None)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(label, stacked[0]) >= roc_auc_score(label, p) - 0.01  # context never hurts much
    assert apply_stack(models, res.q, res.t, res.prob).shape == (len(q),)


def test_support_features_reward_agreement_with_confident_candidates():
    from src.matching.stacking import support_features

    raw = pd.DataFrame([
        ["S2-1", "Supreme Interiors Inc", "9122 Duane Street, Houston, TX", "US"],   # confident true match
        ["S2-2", "Tavowex", "9122 DUANE ST, HOUSTON, TX", "US"],                     # random DBA name, same address
        ["S3-1", "Supreme Interiors", "77 Elm St, Dayton, OH", "US"],                # namesake elsewhere
    ], columns=["entity_id", "business_name", "business_address", "country"])
    store = RecordStore(preprocess_dataframe(raw), n_jobs=1)
    q = np.array([0, 0, 0])
    p = np.array([0.95, 0.3, 0.3], np.float32)
    f = support_features(q, p, np.array([0, 1, 2]), store, k=5)
    assert f.loc[1, "sup_addr_pmax"] > 0.8 and f.loc[2, "sup_addr_pmax"] < 0.3
    assert f.loc[2, "sup_name_max"] > f.loc[1, "sup_name_max"]
    assert f.loc[1, "sup_addr_top"] == f.loc[1, "sup_addr_max"]  # top anchor of row 1 is row 0
