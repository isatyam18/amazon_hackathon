"""
Exact per-pair similarity rescoring of the retrieved candidate union.

Retrieval passes index only rare keys, so their scores are partial. Once the
union of candidates is known, the full TF-IDF cosine of each pair is cheap to
compute directly (a row-wise sparse dot product per pair) on views too dense to
index at this scale (character n-grams of the whole name, the phonetic
skeleton, all address tokens). These similarities rank the candidates for the
final budget cut and are passed on to Stage 3 as features.
"""

from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp


def pair_cosine(
    Qn: sp.csr_matrix, Tn: sp.csr_matrix, qi: np.ndarray, ti: np.ndarray, chunk: int = 1_000_000
) -> np.ndarray:
    """Cosine of row qi[j] of Qn with row ti[j] of Tn (both row-L2-normalised)."""
    out = np.zeros(len(qi), np.float32)
    for a in range(0, len(qi), chunk):
        b = min(a + chunk, len(qi))
        prod = Qn[qi[a:b]].multiply(Tn[ti[a:b]])
        out[a:b] = np.asarray(prod.sum(axis=1), dtype=np.float32).ravel()
    return out


def is_address_view(view: str) -> bool:
    """Views whose similarity depends on the address (0 whenever an address is missing)."""
    return view.startswith("addr_") or view == "name_addr"


def candidate_score(
    sims: Dict[str, np.ndarray],
    weights: Dict[str, float],
    missing_address: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Weighted sum of rescoring similarities (weights: BlockingConfig.rescore_weights).

    About 3% of S2/S3 records have no address. For those pairs the address
    similarities are 0 by construction, not evidence of a mismatch, so the pair
    is scored on its name evidence alone, rescaled to the same range:
    name_part * (sum of all weights / sum of name weights).
    """
    n = len(next(iter(sims.values()))) if sims else 0
    name_part = np.zeros(n, np.float32)
    addr_part = np.zeros(n, np.float32)
    for view, w in weights.items():
        part = addr_part if is_address_view(view) else name_part
        part += np.float32(w) * sims[view]
    score = name_part + addr_part
    name_w = sum(w for v, w in weights.items() if not is_address_view(v))
    total_w = sum(weights.values())
    if missing_address is not None and name_w > 0 and total_w > name_w:
        score = np.where(missing_address, name_part * np.float32(total_w / name_w), score)
    return score
