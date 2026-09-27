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


def _stack_frame(r, extra: Optional[pd.DataFrame]) -> pd.DataFrame:
    X = prob_context_features(r.q, r.t, r.prob)
    if extra is not None:
        X = pd.concat([X, extra.reset_index(drop=True)], axis=1)
    return X


def train_stack(results, folds: np.ndarray, n_train_queries: int = 300_000, seed: int = 7,
                num_boost_round: int = 600, log=print, extras: Optional[List[pd.DataFrame]] = None):
    """Two-fold second stage on OOF probabilities (+ optional extra columns per partition).

    Returns (models [fold0-trained, fold1-trained], list of stacked OOF probabilities per partition).
    """
    extras = extras if extras is not None else [None] * len(results)
    from src.matching.model import train_matcher

    rng = np.random.default_rng(seed)
    parts = {0: [], 1: []}
    ys = {0: [], 1: []}
    for r, extra in zip(results, extras):
        X = _stack_frame(r, extra)
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
    for r, extra in zip(results, extras):
        X = _stack_frame(r, extra)
        pair_fold = folds[r.q]
        out = np.empty(len(r.q), np.float32)
        for f in (0, 1):
            m = pair_fold == f
            if m.any():
                out[m] = models[1 - f].predict_proba(X[m])  # out-of-fold again
        stacked.append(out)
        del X
    return models, stacked


def apply_stack(models: List, q: np.ndarray, t: np.ndarray, p: np.ndarray,
                extra: Optional[pd.DataFrame] = None) -> np.ndarray:
    """Second-stage probability (mean of the stack models) for one partition."""
    X = prob_context_features(q, t, p)
    if extra is not None:
        X = pd.concat([X, extra.reset_index(drop=True)], axis=1)
    return np.mean([m.predict_proba(X) for m in models], axis=0).astype(np.float32)


# --- Support features: agreement with the S1 entity's other confident candidates --------

SUPPORT_VIEWS = (("name_word", "name"), ("addr_word", "addr"), ("addr_num", "num"))


def support_features(q: np.ndarray, p: np.ndarray, t_rows: np.ndarray, store, k: int = 5,
                     chunk: int = 5_000_000) -> pd.DataFrame:
    """Overlap of each candidate with the query's top-k most probable *other* candidates.

    Records of one business in S2/S3 are noisy copies of the same entity, so a
    doubtful candidate (random DBA name, truncated house number) that shares its
    address or name with another confident candidate of the same S1 is probably
    a match, and one that agrees with none of them probably is not.

    q: query id per pair; p: first-stage probability; t_rows: RecordStore row of
    the candidate; store: RecordStore with binary token matrices.
    Returns per view v in (name, addr, num): sup_{v}_max (best Jaccard with an
    anchor), sup_{v}_pmax (best probability-weighted Jaccard) and sup_{v}_top
    (Jaccard with the most probable other candidate).
    """
    from src.blocking.rescore import pair_cosine as pair_dot
    from src.blocking.topk import rank_within_groups

    n = len(q)
    order, rank = rank_within_groups(np.asarray(q, np.int64), np.asarray(p, np.float32))
    q_sorted = np.asarray(q)[order]
    starts = np.flatnonzero(np.r_[True, q_sorted[1:] != q_sorted[:-1]])
    sizes = np.diff(np.r_[starts, n])
    group_start = np.repeat(starts, sizes)       # sorted position of each row's group start
    group_size = np.repeat(sizes, sizes)
    row_of_sorted = order                        # sorted position -> original row
    p = np.asarray(p, np.float32)
    t_rows = np.asarray(t_rows, np.int64)

    out = {}
    for _, name in SUPPORT_VIEWS:
        out[f"sup_{name}_max"] = np.zeros(n, np.float32)
        out[f"sup_{name}_pmax"] = np.zeros(n, np.float32)
        out[f"sup_{name}_top"] = np.full(n, np.nan, np.float32)

    # anchors: the top-(k+1) of each query (one of them may be the row itself)
    for j in range(k + 1):
        has = group_size > j
        anchor_sorted = np.where(has, group_start + j, 0)
        anchor = row_of_sorted[anchor_sorted]            # original row of anchor j, per sorted position
        self_row = row_of_sorted                          # original row at each sorted position
        valid = has & (anchor != self_row)
        # 'top' = most probable other candidate: anchor 0, or anchor 1 for the top row itself
        is_top = valid & ((j == 0) | ((j == 1) & (rank == 0)))
        for a in range(0, n, chunk):
            sl = slice(a, min(a + chunk, n))
            v = np.flatnonzero(valid[sl]) + a
            if len(v) == 0:
                continue
            rows, anc = self_row[v], anchor[v]
            for view, name in SUPPORT_VIEWS:
                m = store.tokens[view]
                inter = pair_dot(m, m, t_rows[rows], t_rows[anc])
                cnt = store.token_counts[view]
                union = cnt[t_rows[rows]] + cnt[t_rows[anc]] - inter
                jac = np.where(union > 0, inter / np.maximum(union, 1), 0.0).astype(np.float32)
                # each row appears at most once per anchor pass: plain vectorised maximum
                out[f"sup_{name}_max"][rows] = np.maximum(out[f"sup_{name}_max"][rows], jac)
                out[f"sup_{name}_pmax"][rows] = np.maximum(out[f"sup_{name}_pmax"][rows], jac * p[anc])
                top = is_top[v]
                out[f"sup_{name}_top"][rows[top]] = jac[top]
    return pd.DataFrame(out)
