"""
Stage 4/5: turning pair probabilities into per-query match sets that score well
under the leaderboard metric (macro F0.5 over S1 entities, singletons included).

Pieces (all selected / tuned on out-of-fold predictions, see scripts/run_matching.py):

- Calibrator: isotonic map from model score to P(match), fitted on OOF data.
- owner_normalize: each S2/S3 record belongs to at most one S1 entity, so the
  probabilities of one record across competing S1 entities should sum to at most 1.
  p(q, t) / max(1, sum_q' p(q', t)).
- select_expected_f: per query, pick the top-k candidates (by probability) that
  maximise the expected F0.5, or the empty set when "no match" is the better bet.
  With F_beta = (1+b^2) TP / (|S| + b^2 |T|), the expectation is approximated by
  E_k = (1+b^2) sum_{i<=k} p_i / (k + b^2 * E|T|), with E|T| = sum_i p_i + m, where m is
  the expected number of true matches blocking missed. The empty set scores
  1.0 exactly when the query has no match: P(empty) = prod_i (1 - p_i) * exp(-m).
  Queries with many likely matches predict more, uncertain ones fewer, and
  likely singletons nothing: the metric's per-query trade-off.
- select_threshold: the plain global threshold, kept as a baseline.
- enforce_exclusivity: a record predicted for several S1 entities stays only with
  the most probable one.
"""

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from src.blocking.topk import rank_within_groups


class Calibrator:
    """Monotone (isotonic) calibration of model scores, stored as interpolation knots."""

    def __init__(self, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None):
        self.x = x
        self.y = y

    def fit(self, scores: np.ndarray, labels: np.ndarray, max_samples: int = 5_000_000, seed: int = 0) -> "Calibrator":
        from sklearn.isotonic import IsotonicRegression

        if len(scores) > max_samples:
            idx = np.random.default_rng(seed).choice(len(scores), max_samples, replace=False)
            scores, labels = scores[idx], labels[idx]
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(scores, labels)
        self.x, self.y = iso.X_thresholds_, iso.y_thresholds_
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self.x is None:
            return np.asarray(scores, np.float64)
        return np.interp(scores, self.x, self.y)

    def to_dict(self) -> dict:
        return {"x": [float(v) for v in self.x], "y": [float(v) for v in self.y]} if self.x is not None else {}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        return cls(np.array(d["x"]), np.array(d["y"])) if d else cls()


@dataclass
class DecisionRule:
    method: str = "expected_f"      # "expected_f" or "threshold"
    threshold: float = 0.5          # threshold method: minimum probability
    missing_mass: float = 0.035     # expected_f: expected true matches per query missed by blocking
    owner_normalize: bool = True    # divide by the record's total probability across S1 entities
    exclusive: bool = True          # a record keeps at most one S1 entity
    beta: float = 0.5
    min_singleton_prob: float = 0.60  # require at least one candidate with p >= 0.60 to break singleton

    def to_dict(self) -> dict:
        return asdict(self)


def owner_normalize(target: np.ndarray, p: np.ndarray) -> np.ndarray:
    total = np.bincount(target, weights=p)
    return p / np.maximum(1.0, total[target])


def select_threshold(query: np.ndarray, p: np.ndarray, threshold: float) -> np.ndarray:
    return p >= threshold


def select_expected_f(query: np.ndarray, p: np.ndarray, beta: float = 0.5, missing_mass: float = 0.0,
                      sorted_by_query=None, min_singleton_prob: float = 0.60) -> np.ndarray:
    """Boolean mask of the expected-F_beta-optimal candidate set of every query.

    `sorted_by_query` may pass a precomputed `rank_within_groups(query, p)` result
    (reused when several rules are evaluated on the same probabilities).
    """
    n = len(p)
    if n == 0:
        return np.zeros(0, bool)
    b2 = beta * beta
    order, rank = sorted_by_query if sorted_by_query is not None else rank_within_groups(query, p)
    rank = rank.astype(np.int64)  # int32 + int64 sentinel would overflow under NumPy 2 promotion
    q_s, p_s = query[order], np.clip(p[order], 0.0, 1.0)
    starts = np.flatnonzero(np.r_[True, q_s[1:] != q_s[:-1]])
    sizes = np.diff(np.r_[starts, n])
    group = np.repeat(np.arange(len(starts)), sizes)

    csum = np.cumsum(p_s)
    base = np.repeat(np.r_[0.0, csum[starts[1:] - 1]] if len(starts) > 1 else np.zeros(1), sizes)
    prefix = csum - base                                      # sum of the top-(rank+1) probabilities
    total = np.bincount(group, weights=p_s)[group] + missing_mass
    k = rank + 1.0
    expected = (1 + b2) * prefix / (k + b2 * total)

    log_empty = np.bincount(group, weights=np.log1p(-np.minimum(p_s, 1 - 1e-12))) - missing_mass
    empty_score = np.exp(log_empty)

    # rows are sorted by query, so segment reductions replace slow ufunc.at scatters
    best = np.maximum.reduceat(expected, starts)
    is_best = expected >= best[group] - 1e-12
    big = np.iinfo(np.int64).max
    best_rank = np.minimum.reduceat(np.where(is_best, rank, big), starts)  # first rank reaching the best
    top_p = np.maximum.reduceat(p_s, starts)  # highest candidate probability for the query (shape: len(starts))
    keep_group = (best > empty_score) & (top_p >= min_singleton_prob)
    chosen = keep_group[group] & (rank <= best_rank[group])

    mask = np.zeros(n, bool)
    mask[order] = chosen
    return mask


def enforce_exclusivity(target: np.ndarray, p: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Keep each selected target only for its most probable query."""
    idx = np.flatnonzero(selected)
    if len(idx) == 0:
        return selected
    order, rank = rank_within_groups(target[idx], p[idx])
    out = np.zeros_like(selected)
    out[idx[order[rank == 0]]] = True
    return out


def apply_rule(rule: DecisionRule, query: np.ndarray, target: np.ndarray, p: np.ndarray,
               cache: Optional[dict] = None) -> np.ndarray:
    """Selected-pair mask for calibrated probabilities `p` under `rule`.

    `cache` (a dict kept by the caller) reuses the owner-normalised probabilities
    and their per-query sort order across rules evaluated on the same `p`.
    """
    cache = {} if cache is None else cache
    key = ("prob", rule.owner_normalize)
    if key not in cache:
        cache[key] = owner_normalize(target, p) if rule.owner_normalize else p
    q = cache[key]
    if rule.method == "threshold":
        selected = select_threshold(query, q, rule.threshold)
    elif rule.method == "expected_f":
        skey = ("sorted", rule.owner_normalize)
        if skey not in cache:
            cache[skey] = rank_within_groups(query, q)
        min_p = getattr(rule, "min_singleton_prob", 0.60)
        selected = select_expected_f(query, q, rule.beta, rule.missing_mass,
                                     sorted_by_query=cache[skey], min_singleton_prob=min_p)
    else:
        raise ValueError(f"Unknown decision method {rule.method!r}")
    return enforce_exclusivity(target, q, selected) if rule.exclusive else selected
