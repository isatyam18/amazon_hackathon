"""
Stage 3: context (competition) features computed over the whole candidate graph.

A pair's own similarities do not say whether a *better* explanation exists:
- Query side: 'Super Developers' in Ballia may have five S2 candidates called
  'Super Developers'. The best-addressed one is the likely match, the rest are
  branches or namesakes. Features: the pair's margin over the best other
  candidate of the same query (overall and within its source), its rank, and how
  many near-equal candidates the query has.
- Target side: every S2/S3 record belongs to at most one S1 entity (this holds for
  all 7.6M training pairs). If another S1 entity matches the same record much
  better, this pair is unlikely. Features: number of S1 entities competing for
  the record, and this pair's margin over the best competing S1.

The features use only Stage 2 scores, never labels, so they are computed
identically for training, validation and test. Complete competition needs every
S1 entity of the split, which is why Stage 2 blocks the full train S1 set.
"""

from typing import Dict

import numpy as np


def _group_margin(keys: np.ndarray, score: np.ndarray):
    """Per row: margin over the best *other* row of the same key, rank in key, group size, group best."""
    # float32 / int32 temporaries: this runs on 40M+ rows per partition
    order = np.lexsort((-score, keys))
    k, s = keys[order], score[order].astype(np.float32)
    n = len(k)
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]]).astype(np.int32) if n else np.zeros(0, np.int32)
    del k
    sizes = np.diff(np.r_[starts, n]).astype(np.int32)
    group_start = np.repeat(starts, sizes)
    rank = np.arange(n, dtype=np.int32) - group_start
    best = s[group_start]
    second = s[np.minimum(group_start + 1, n - 1)]
    del group_start
    second[np.repeat(sizes, sizes) <= 1] = np.nan
    other_best = np.where(rank == 0, second, best)
    del second

    margin = np.empty(n, np.float32)
    rank_out = np.empty(n, np.float32)
    size_out = np.empty(n, np.float32)
    best_out = np.empty(n, np.float32)
    margin[order] = s - other_best
    rank_out[order] = rank
    size_out[order] = np.repeat(sizes, sizes)
    best_out[order] = best
    return margin, rank_out, size_out, best_out


def context_features(
    query: np.ndarray, target: np.ndarray, source: np.ndarray, score: np.ndarray, name_sim: np.ndarray
) -> Dict[str, np.ndarray]:
    """Context features for every pair of a split.

    query/target: integer ids; source: small int code (S2/S3); score: Stage 2
    candidate score; name_sim: exact name similarity (sim_name_char).
    """
    query = np.asarray(query, np.int64)
    target = np.asarray(target, np.int64)
    score = np.asarray(score, np.float32)
    out = {}

    out["q_margin"], out["q_rank"], out["q_size"], q_best = _group_margin(query, score)
    qs_key = query * 4 + np.asarray(source, np.int64)
    out["qs_margin"], _, _, _ = _group_margin(qs_key, score)
    out["t_margin"], out["t_rank"], out["t_competitors"], _ = _group_margin(target, score)

    # Relative strength within the query and how many near-duplicates it has
    rel = score / np.where(q_best > 0, q_best, 1.0)
    out["q_rel_score"] = rel.astype(np.float32)
    strong = (rel >= 0.9).astype(np.float64)
    out["q_n_strong"] = np.bincount(query, weights=strong)[query].astype(np.float32)
    name_dupe = (np.asarray(name_sim) >= 0.9).astype(np.float64)
    out["q_n_name_dupes"] = np.bincount(query, weights=name_dupe)[query].astype(np.float32)
    out["t_n_name_dupes"] = np.bincount(target, weights=name_dupe)[target].astype(np.float32)
    return out
