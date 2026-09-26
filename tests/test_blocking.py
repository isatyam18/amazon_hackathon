"""
Unit tests for Stage 2 (blocking). Run from the project root:

    python -m pytest tests/test_blocking.py -q
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.blocking import BlockingConfig, BlockingPipeline, evaluate_candidates
from src.blocking.data import read_candidate_tsv
from src.blocking.evaluation import macro_f05
from src.blocking.fusion import select_top, union_results
from src.blocking.normalize import address_tokens, fix_leet, name_tokens, phonetic_key, phonetic_name
from src.blocking.topk import sparse_topk
from src.preprocessing import clean_text, preprocess_dataframe
from src.transliteration import transliterate_indic


# --------------------------------------------------------------------------- normalization

def test_transliteration_indic_scripts():
    assert transliterate_indic("राम मार्केटिंग") == "ram marketing"
    assert transliterate_indic("ಶಿವ್ ಎಂಟರ್‌ಪ್ರೈಸಸ್") == "shiv entarpraisas"
    assert transliterate_indic("గుజరాత్ Logistics") == "gujarat Logistics"
    assert transliterate_indic("സിൽവർ") == "silvar"
    assert transliterate_indic("plain ascii") == "plain ascii"


def test_clean_text_keeps_transliterated_names():
    # Before transliteration existed, Indic names were stripped to ''
    assert clean_text("राम मार्केटिंग प्राइवेट लिमिटेड") == "ram marketing praivet limited"
    assert clean_text("Société Générale") == "societe generale"


def test_name_tokens_drop_legal_and_noise():
    assert name_tokens("ltd ontime pvt services") == ["ontime", "services"]
    assert name_tokens("m s ashutoshbrosprivate com") == ["ashutoshbrosprivate"]
    assert name_tokens("anselma s holdings id 14307") == ["anselma", "s", "holdings"]
    assert name_tokens("ram marketing praivet limited") == ["ram", "marketing"]
    assert name_tokens("llc") == ["llc"]  # nothing left -> fall back to raw tokens


def test_fix_leet():
    assert fix_leet("s0reis") == "soreis"
    assert fix_leet("5ervices") == "services"
    assert fix_leet("127a") == "127a"
    assert fix_leet("3rd") == "3rd"
    assert fix_leet("2024") == "2024"


def test_phonetic_key_bridges_transliteration():
    assert phonetic_name(["global", "business"]) == phonetic_name(["kulopal", "pichinas"])
    assert phonetic_key("enterprises") == phonetic_key("entarpraisas")
    assert phonetic_key("software") == phonetic_key("sophtaveyar")


def test_address_tokens():
    assert address_tokens("h number 127a colonial circle null") == ["127a", "127", "colonial", "circle"]


# --------------------------------------------------------------------------- top-k engine

def test_sparse_topk_matches_brute_force():
    rng = np.random.default_rng(0)
    Q = sp.random(300, 500, density=0.03, format="csr", dtype=np.float32, random_state=1)
    T = sp.random(2000, 500, density=0.03, format="csr", dtype=np.float32, random_state=2)
    k = 5
    # tiny product budget forces many chunks and exercises the threaded path
    q, t, s, r = sparse_topk(Q, T.T.tocsr(), k=k, min_score=1e-6, n_threads=4, max_product_nnz=500)
    dense = (Q @ T.T).toarray()
    for qi in range(Q.shape[0]):
        got = s[q == qi]
        expect = np.sort(dense[qi][dense[qi] >= 1e-6])[::-1][:k]
        np.testing.assert_allclose(got, expect, rtol=1e-5)
        assert list(r[q == qi]) == list(range(len(got)))
    assert np.all(np.diff(q) >= 0)


def test_union_keeps_per_pass_scores_and_best_rank():
    a = (np.array([0, 0]), np.array([10, 11]), np.array([0.9, 0.8], np.float32), np.array([0, 1]))
    b = (np.array([0, 1]), np.array([11, 12]), np.array([0.7, 0.6], np.float32), np.array([0, 0]))
    u = union_results({"a": a, "b": b}, ["a", "b"])
    pairs = {(int(q), int(t)): (tuple(s), int(r)) for q, t, s, r in zip(u.query, u.target, u.pass_scores, u.best_rank)}
    assert pairs[(0, 11)] == ((np.float32(0.8), np.float32(0.7)), 0)
    assert pairs[(0, 10)] == ((np.float32(0.9), np.float32(0.0)), 0)
    assert pairs[(1, 12)] == ((np.float32(0.0), np.float32(0.6)), 0)


def test_select_top_budget_and_threshold():
    groups = np.array([0, 0, 0, 1, 1])
    score = np.array([0.1, 0.9, 0.5, 0.3, 0.05], np.float32)
    keep, rank = select_top(groups, score, max_per_group=2, min_score=0.1)
    assert list(keep) == [1, 2, 3] and list(rank) == [0, 1, 0]


# --------------------------------------------------------------------------- end-to-end pipeline

def _frame(rows):
    raw = pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])
    return preprocess_dataframe(raw)


@pytest.fixture(scope="module")
def toy_data():
    s1 = _frame([
        ["S1-1", "Complete Property Solutions LLC", "10528 Cedar Falls Loop, Hillsboro, OR", "US"],
        ["S1-2", "Shiv Enterprises Private Limited", "Wework Embassy Techvillage, Bellandur, Bangalore", "India"],
        ["S1-3", "Pharmacie du Madeleine", "37 Rue des Augustins, Bordeaux", "France"],
        ["S1-4", "Lonely Singleton Widgets Inc", "1 Nowhere Road, Nome, AK", "US"],
        ["S1-5", "Supreme Interiors Inc", "9122 Duane Street, Houston, TX", "US"],
        ["S1-6", "Kiwi Imports Ltd", "5 Queen Street, Auckland", "New Zealand"],
    ])
    s2 = _frame([
        ["S2-1", "LLC Complete Property Soluiscos", "#10528 Cedar Falls Loop, Hillsboro, Oregon", "US"],
        ["S2-2", "ಶಿವ್ ಎಂಟರ್‌ಪ್ರೈಸಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "WEWORK EMBASSY TECHVILLAGE, BELLANDUR, Karnataka", "India"],
        ["S2-3", "Tavowex", "9122 DUANE STREET, HOUSTON, TX", "US"],
        ["S2-4", "Unrelated Bakery LLC", "77 Elm St, Dayton, OH", "US"],
        ["S2-5", "Complete Property Solutions LLC", "10528 Cedar Falls Loop, Hillsboro, OR", "India"],  # wrong country
        ["S2-6", "Kiwi Imports Limited", "5 Queen St, Auckland", "Australia"],
    ])
    s3 = _frame([
        ["S3-1", "Pharmacie Madeleine SARL", "37 R des Augustins, Bordeaux, Nouvelle-Aquitaine", "France"],
        ["S3-2", "shiventerprises.com", "Bellandur, Bangalore South, ಕರ್ನಾಟಕ", "India"],
        ["S3-3", "Supreme Interi0rs", "#9122 Duane St, Texas, Houston", "US"],
        ["S3-4", "Random Unrelated Name", "12 Other Ave, Paris", "France"],
    ])
    gt = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2", "S1-3", "S1-4", "S1-5", "S1-6"],
        "matched_entity_ids": ["S2-1", "S2-2,S3-2", "S3-1", "", "S2-3,S3-3", "S2-6"],
    })
    return s1, {"S2": s2, "S3": s3}, gt


@pytest.fixture(scope="module")
def toy_candidates(toy_data):
    s1, targets, _ = toy_data
    cfg = BlockingConfig(n_jobs=1, n_threads=2, hash_bits=18)
    return BlockingPipeline(cfg, log=lambda *_: None).run(s1, targets)


def test_pipeline_finds_true_matches(toy_data, toy_candidates):
    _, _, gt = toy_data
    report = evaluate_candidates(toy_candidates.to_frame(), gt, n_targets=10)
    assert report["pair_recall"] == 1.0
    assert report["f05_ceiling"] == 1.0


def test_pipeline_respects_country_partition(toy_candidates):
    df = toy_candidates.to_frame()
    s1_1 = set(df.loc[df.source1_entity_id == "S1-1", "candidate_entity_id"])
    assert "S2-5" not in s1_1  # identical record but labelled India
    s1_3 = set(df.loc[df.source1_entity_id == "S1-3", "candidate_entity_id"])
    assert s1_3 <= {"S3-1", "S3-4"}  # France only


def test_pipeline_unseen_country_falls_back_to_all_targets(toy_candidates):
    df = toy_candidates.to_frame()
    assert "S2-6" in set(df.loc[df.source1_entity_id == "S1-6", "candidate_entity_id"])


def _load_validator():
    path = os.path.join(ROOT, "student_resource", "utils", "validate_submission.py")
    spec = importlib.util.spec_from_file_location("validate_submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_candidate_tsv_passes_official_validator(tmp_path, toy_data, toy_candidates):
    s1, _, _ = toy_data
    path = tmp_path / "candidate_pairs.tsv"
    toy_candidates.write_tsv(str(path))

    validator = _load_validator()
    errors = []
    mapping = validator.validate_id_list_file(
        str(path), validator.CANDIDATE_HEADER, "candidate_entity_ids",
        set(s1.entity_id), None, errors,
    )
    assert errors == []
    assert set(mapping) == set(s1.entity_id)  # every S1 entity has exactly one row

    exploded = read_candidate_tsv(str(path))
    assert len(exploded) == toy_candidates.n_pairs


def test_parquet_roundtrip(tmp_path, toy_candidates):
    path = tmp_path / "pairs.parquet"
    toy_candidates.write_parquet(str(path))
    df = pd.read_parquet(path)
    assert len(df) == toy_candidates.n_pairs
    assert {"candidate_score", "source_rank", "score_name", "sim_name_char"} <= set(df.columns)


def test_macro_f05_matches_problem_statement_example():
    gt = pd.DataFrame({"source1_entity_id": ["S1-00001", "S1-00003"],
                       "matched_entity_ids": ["S2-00047,S3-00812", ""]})
    pred = pd.DataFrame({"source1_entity_id": ["S1-00001"] * 3,
                         "candidate_entity_id": ["S2-00047", "S2-00193", "S3-00812"]})
    # 0.714 for S1-00001 (example in the problem statement) and 1.0 for the empty singleton
    assert macro_f05(pred, gt) == pytest.approx((0.7142857 + 1.0) / 2, abs=1e-6)


def test_dense_pass_with_injected_encoder(toy_data):
    pytest.importorskip("faiss")
    s1, targets, gt = toy_data
    from sklearn.feature_extraction.text import HashingVectorizer

    hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=256, alternate_sign=False)
    encoder = lambda texts: hv.transform(texts).toarray()
    cfg = BlockingConfig(n_jobs=1, n_threads=1, hash_bits=18)
    cfg.dense.enabled = True
    cfg.dense.min_score = 0.0
    cands = BlockingPipeline(cfg, log=lambda *_: None, dense_encoder=encoder).run(s1, targets)
    df = cands.to_frame()
    assert "score_dense" in df.columns
    assert evaluate_candidates(df, gt)["pair_recall"] == 1.0


def test_tuning_collects_labels_and_fits_weights(toy_data):
    from src.blocking.tuning import collect_labelled_union, fit_rescore_weights, recall_at_budgets, score_with_weights

    s1, targets, gt = toy_data
    cfg = BlockingConfig(n_jobs=1, n_threads=1, hash_bits=18)
    pipe = BlockingPipeline(cfg, log=lambda *_: None)
    df = collect_labelled_union(pipe, s1, targets, gt, list(cfg.rescore_weights))
    assert df["label"].sum() >= 6  # every true pair within a labelled country is retrieved
    weights = fit_rescore_weights(df, list(cfg.rescore_weights))
    assert set(weights) == set(cfg.rescore_weights)
    rec = recall_at_budgets(df, score_with_weights(df, cfg.rescore_weights), int(df["label"].sum()), [25])
    assert rec[25] == 1.0
