"""
Chunked, multithreaded sparse top-k retrieval.

Given L2-normalised query rows Q (m x F) and an inverted index TT = T^T (F x n,
CSR, i.e. one posting list per feature), `sparse_topk` returns for every query
the k targets with the highest dot product (cosine) at or above `min_score`.

The product Q @ TT is computed in query chunks. The chunk size is chosen from
the exact upper bound on the product's size — the sum of posting-list lengths of
a query's features — so peak memory stays under `max_product_nnz` per thread
regardless of how common a query's features are. scipy's sparse matmul releases
the GIL, so chunks run in parallel threads.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp


def rank_within_groups(groups: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Order rows by (group asc, score desc) and return (order, rank-within-group)."""
    order = np.lexsort((-scores, groups))
    g = groups[order]
    if len(g) == 0:
        return order, np.zeros(0, dtype=np.int32)
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    lengths = np.diff(np.r_[starts, len(g)])
    rank = np.arange(len(g)) - np.repeat(starts, lengths)
    return order, rank.astype(np.int32)


def _chunk_bounds(cost: np.ndarray, budget: int, max_rows: int) -> List[Tuple[int, int]]:
    """Split rows into contiguous chunks whose summed cost stays under `budget`."""
    bounds = []
    m = len(cost)
    start = 0
    csum = np.cumsum(cost, dtype=np.int64)
    while start < m:
        base = csum[start - 1] if start > 0 else 0
        end = int(np.searchsorted(csum, base + budget, side="right"))
        end = max(end, start + 1)             # a single expensive row gets its own chunk
        end = min(end, start + max_rows, m)
        bounds.append((start, end))
        start = end
    return bounds


def _topk_chunk(Q: sp.csr_matrix, TT: sp.csr_matrix, start: int, end: int, k: int, min_score: float):
    C = Q[start:end] @ TT
    indptr, data = C.indptr, C.data
    counts = np.diff(indptr)

    # Rows with <= k products keep everything; longer rows keep their k best via
    # argpartition (linear time). Sorting then only touches <= k entries per row.
    sel = np.ones(len(data), bool)
    for i in np.flatnonzero(counts > k):
        s, e = indptr[i], indptr[i + 1]
        row_sel = np.zeros(e - s, bool)
        row_sel[np.argpartition(-data[s:e], k)[:k]] = True
        sel[s:e] = row_sel
    sel &= data >= min_score

    rows = np.repeat(np.arange(start, end, dtype=np.int32), counts)[sel]
    cols = C.indices[sel].astype(np.int32, copy=False)
    vals = data[sel].astype(np.float32, copy=False)
    order, rank = rank_within_groups(rows, vals)
    return rows[order], cols[order], vals[order], rank


def sparse_topk(
    Q: sp.csr_matrix,
    TT: sp.csr_matrix,
    k: int,
    min_score: float = 0.0,
    n_threads: int = 8,
    max_product_nnz: int = 15_000_000,
    max_rows_per_chunk: int = 20_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Top-k cosine neighbours of each query row.

    Returns (query_idx, target_idx, score, rank) arrays sorted by query then
    descending score; rank is 0-based within each query.
    """
    Q = Q.tocsr()
    m = Q.shape[0]
    empty = (np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32), np.zeros(0, np.int32))
    if m == 0 or TT.shape[1] == 0 or Q.nnz == 0:
        return empty

    posting_len = np.diff(TT.indptr).astype(np.int64)
    row_of_nz = np.repeat(np.arange(m), np.diff(Q.indptr))
    cost = np.bincount(row_of_nz, weights=posting_len[Q.indices], minlength=m).astype(np.int64)

    bounds = _chunk_bounds(cost, max_product_nnz, max_rows_per_chunk)
    if n_threads > 1 and len(bounds) > 1:
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            parts = list(ex.map(lambda b: _topk_chunk(Q, TT, b[0], b[1], k, min_score), bounds))
    else:
        parts = [_topk_chunk(Q, TT, a, b, k, min_score) for a, b in bounds]

    parts = [p for p in parts if len(p[0])]
    if not parts:
        return empty
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))
