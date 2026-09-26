"""
Fitting the rescoring weights (BlockingConfig.rescore_weights) from labelled candidates.

Procedure (scripts/fit_rescore_weights.py):
  1. run the retrieval passes for ground-truth queries (e.g. the validation split)
     against the full target sources,
  2. compute the exact similarity of every retrieved pair on each candidate view,
  3. label pairs with the ground truth and fit a logistic regression (true pair vs
     not) on half of the queries,
  4. measure recall at each per-source budget on the other half, comparing the
     fitted weights with the current ones.

The logistic-regression coefficients are then used directly as linear ranking
weights. Ranking within a query does not depend on the intercept.
"""

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from src.blocking.data import ground_truth_pairs
from src.blocking.fusion import union_results
from src.blocking.pipeline import BlockingPipeline
from src.blocking.rescore import candidate_score, pair_cosine


def collect_labelled_union(
    pipeline: BlockingPipeline,
    queries: pd.DataFrame,
    targets: Dict[str, pd.DataFrame],
    gt: pd.DataFrame,
    sim_views: Sequence[str],
) -> pd.DataFrame:
    """Retrieved (query, target) union with exact similarities on `sim_views` and a GT label.

    `pipeline.config` must include every view in `sim_views` in its rescore_weights
    (so the index keeps their normalised target rows).
    """
    truth = ground_truth_pairs(gt)
    truth_keys = set(zip(truth["source1_entity_id"], truth["candidate_entity_id"]))
    q_country = queries["country"].to_numpy(object)
    frames = []
    for country in sorted(pd.unique(q_country)):
        q_rows = np.flatnonzero(q_country == country)
        q_feats = pipeline.featurize(queries, q_rows)
        for src, tdf in targets.items():
            t_rows = np.flatnonzero(tdf["country"].to_numpy(object) == country)
            if len(t_rows) == 0:
                continue
            index = pipeline.build_index(tdf, t_rows)
            union = union_results(pipeline.retrieve(index, q_feats), pipeline.pass_names)
            frame = {
                "source1_entity_id": queries["entity_id"].to_numpy(object)[q_rows[union.query]],
                "candidate_entity_id": tdf["entity_id"].to_numpy(object)[t_rows[union.target]],
                "source": src,
                "country": country,
            }
            q_has = np.diff(q_feats["addr_word"].indptr) > 0 if "addr_word" in q_feats else None
            if q_has is not None and index.has_address is not None:
                frame["missing_address"] = ~(q_has[union.query] & index.has_address[union.target])
            for view in sim_views:
                frame[f"sim_{view}"] = pair_cosine(
                    index.normalize_queries(view, q_feats[view]), index.rescore[view], union.query, union.target
                )
            df = pd.DataFrame(frame)
            df["label"] = [k in truth_keys for k in zip(df["source1_entity_id"], df["candidate_entity_id"])]
            frames.append(df)
            pipeline.log(f"[tuning] {country}/{src}: {len(df):,} retrieved pairs, {int(df['label'].sum()):,} true")
            del index
    return pd.concat(frames, ignore_index=True)


def score_with_weights(df: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
    sims = {v: df[f"sim_{v}"].to_numpy(np.float32) for v in weights}
    missing = df["missing_address"].to_numpy() if "missing_address" in df else None
    return candidate_score(sims, weights, missing)


def recall_at_budgets(
    df: pd.DataFrame, score: np.ndarray, n_true: int, budgets: Iterable[int]
) -> Dict[int, float]:
    """Fraction of `n_true` true pairs kept when each (query, source) keeps its top-b by score."""
    ranked = df[["source1_entity_id", "source", "label"]].assign(score=score)
    ranked = ranked.sort_values(["source1_entity_id", "source", "score"], ascending=[True, True, False])
    rank = ranked.groupby(["source1_entity_id", "source"]).cumcount().to_numpy()
    label = ranked["label"].to_numpy()
    return {int(b): float(label[rank < b].sum() / max(n_true, 1)) for b in budgets}


def fit_rescore_weights(df: pd.DataFrame, views: Sequence[str], C: float = 1.0) -> Dict[str, float]:
    """Logistic-regression coefficients (true pair vs not) on pairs with both addresses present."""
    from sklearn.linear_model import LogisticRegression

    fit_rows = ~df["missing_address"].to_numpy() if "missing_address" in df else np.ones(len(df), bool)
    X = df.loc[fit_rows, [f"sim_{v}" for v in views]].to_numpy()
    y = df.loc[fit_rows, "label"].to_numpy()
    lr = LogisticRegression(C=C, max_iter=1000, class_weight="balanced").fit(X, y)
    return {v: round(float(c), 3) for v, c in zip(views, lr.coef_[0])}


def split_queries(query_ids: Sequence[str], holdout_frac: float, seed: int) -> List[str]:
    """Random subset of query ids used for fitting (the rest is held out)."""
    rng = np.random.default_rng(seed)
    ids = np.asarray(sorted(set(query_ids)), dtype=object)
    n_fit = int(round(len(ids) * (1.0 - holdout_frac)))
    return list(rng.choice(ids, size=n_fit, replace=False))
