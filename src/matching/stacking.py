"""
Stage 4b: second-stage (stacked) model on first-stage probabilities in context.

The strongest first-stage feature compares Stage 2 scores between competing S1
entities for the same S2/S3 record (context.py). With first-stage probabilities
the same comparison becomes sharper: "another S1 already matches this record with
p=0.98", "this S1 has two other candidates above 0.9". The second stage learns
from these probability-context features only (no string features are recomputed),
so it is cheap to train and apply.

Leakage control: every probability used here is out-of-fold (train) or from the
averaged fold models (test). The second stage itself is trained with the same two
query folds as the first stage, so every pair again gets an out-of-fold
second-stage probability for calibration, tuning and reporting.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.matching.context import _group_margin

STACK_FEATURES = [
    "p", "logit_p",
    "q_margin_p", "q_rank_p", "q_sum_p", "q_n_above_half",
    "t_margin_p", "t_rank_p", "t_sum_p", "t_competitors",
]


def prob_context_features(q: np.ndarray, t: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """Probability-context features for every pair of one partition."""
    q = np.asarray(q, np.int64)
    t = np.asarray(t, np.int64)
    p = np.asarray(p, np.float32)
    f: Dict[str, np.ndarray] = {"p": p}
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    f["logit_p"] = np.log(pc / (1 - pc)).astype(np.float32)
    f["q_margin_p"], f["q_rank_p"], _, _ = _group_margin(q, p)
    f["t_margin_p"], f["t_rank_p"], f["t_competitors"], _ = _group_margin(t, p)
    f["q_sum_p"] = np.bincount(q, weights=p)[q].astype(np.float32)
    f["t_sum_p"] = np.bincount(t, weights=p)[t].astype(np.float32)
    f["q_n_above_half"] = np.bincount(q, weights=(p >= 0.5).astype(np.float64))[q].astype(np.float32)
    return pd.DataFrame({k: np.asarray(f[k], np.float32) for k in STACK_FEATURES})


STACK_PARAMS = {
    "num_leaves": 31,
    "min_child_samples": 500,
    "learning_rate": 0.05,
    "colsample_bytree": 1.0,
}


def train_stack(results, folds: np.ndarray, n_train_queries: int = 300_000, seed: int = 7,
                num_boost_round: int = 600, log=print):
    """Two-fold second stage on OOF probabilities.

    Returns (models [fold0-trained, fold1-trained], list of stacked OOF probabilities per partition).
    """
    from src.matching.model import train_matcher

    rng = np.random.default_rng(seed)
    parts = {0: [], 1: []}
    ys = {0: [], 1: []}
    for r in results:
        X = prob_context_features(r.q, r.t, r.prob)
        labels = r.labels()
        pair_fold = folds[r.q]
        for f in (0, 1):
            fq = np.unique(r.q[pair_fold == f])
            share = len(fq) / max(int((folds == f).sum()), 1)
            take = rng.choice(fq, size=min(len(fq), int(n_train_queries * share) + 1), replace=False)
            rows = np.flatnonzero(np.isin(r.q, take))
            parts[f].append(X.iloc[rows])
            ys[f].append(labels[rows])
        del X
    models = []
    for f in (0, 1):
        X_f = pd.concat(parts[f], ignore_index=True)
        y_f = np.concatenate(ys[f]).astype(int)
        n_es = len(X_f) // 10
        perm = np.random.default_rng(seed + f).permutation(len(X_f))
        es, tr = perm[:n_es], perm[n_es:]
        log(f"[stacking] fold {f}: training on {len(tr):,} pairs")
        models.append(train_matcher(X_f.iloc[tr], y_f[tr], X_f.iloc[es], y_f[es], params=dict(STACK_PARAMS),
                                    num_boost_round=num_boost_round, early_stopping_rounds=30))
    stacked = []
    for r in results:
        X = prob_context_features(r.q, r.t, r.prob)
        pair_fold = folds[r.q]
        out = np.empty(len(r.q), np.float32)
        for f in (0, 1):
            m = pair_fold == f
            if m.any():
                out[m] = models[1 - f].predict_proba(X[m])  # out-of-fold again
        stacked.append(out)
        del X
    return models, stacked


def apply_stack(models: List, q: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Second-stage probability (mean of the stack models) for one partition."""
    X = prob_context_features(q, t, p)
    return np.mean([m.predict_proba(X) for m in models], axis=0).astype(np.float32)
