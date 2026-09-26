"""
Union of multiple retrieval passes and budgeted top-N selection.

Each pass yields (query, target, cosine, rank). The union keeps every distinct
(query, target) pair, the cosine each pass gave it (0 where a pass did not
retrieve it), and its best rank across passes. The per-pass cosines become
Stage 3 features. Final ordering comes from the exact rescoring similarities
(rescore.py), not from these partial retrieval scores.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.blocking.topk import rank_within_groups

PassResult = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]  # (query, target, score, rank)


@dataclass
class CandidateUnion:
    query: np.ndarray        # int32 query row
    target: np.ndarray       # int32 target row (within the partition)
    pass_scores: np.ndarray  # float32 (n_pairs, n_passes)
    best_rank: np.ndarray    # int32 best rank over passes (0 = someone's top hit)

    def __len__(self) -> int:
        return len(self.query)


def union_results(results: Dict[str, PassResult], pass_names: List[str]) -> CandidateUnion:
    n_passes = len(pass_names)
    parts = [(pi, results[name]) for pi, name in enumerate(pass_names) if name in results and len(results[name][0])]
    if not parts:
        z = np.zeros(0, np.int32)
        return CandidateUnion(z, z, np.zeros((0, n_passes), np.float32), z)

    q = np.concatenate([r[0] for _, r in parts]).astype(np.int64)
    t = np.concatenate([r[1] for _, r in parts]).astype(np.int64)
    score = np.concatenate([r[2] for _, r in parts])
    rank = np.concatenate([r[3] for _, r in parts]).astype(np.int32)
    pid = np.concatenate([np.full(len(r[0]), pi, np.int32) for pi, r in parts])

    uniq, inv = np.unique((q << 32) | t, return_inverse=True)
    pass_scores = np.zeros((len(uniq), n_passes), np.float32)
    pass_scores[inv, pid] = score  # a (pair, pass) combination occurs at most once
    best_rank = np.full(len(uniq), np.iinfo(np.int32).max, np.int32)
    np.minimum.at(best_rank, inv, rank)
    return CandidateUnion(
        query=(uniq >> 32).astype(np.int32),
        target=(uniq & 0xFFFFFFFF).astype(np.int32),
        pass_scores=pass_scores,
        best_rank=best_rank,
    )


def select_top(groups: np.ndarray, score: np.ndarray, max_per_group: int, min_score: float = -np.inf):
    """Indices of the best `max_per_group` rows per group (score >= min_score), and their ranks."""
    eligible = np.flatnonzero(score >= min_score)
    order, rank = rank_within_groups(groups[eligible], score[eligible])
    keep = rank < max_per_group
    return eligible[order[keep]], rank[keep]
