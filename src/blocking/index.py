"""
TF-IDF inverted index over one target partition (one source, one country).

IDF is fitted on the partition's own targets, so every country (including
France, unseen in training) gets statistics from its own data rather than a
vocabulary learned on US/India.

For each pass the index stores TT = T^T (features x targets, CSR): one posting
list per feature. Features whose document frequency exceeds the pass's `max_df`
are left out of the postings. They still count in the query/target L2 norms,
so the scores remain true cosine contributions from the rarer, more informative
features. This caps the work (and memory) spent on features like 'ent' or
'road' that every other record shares.
"""

from typing import Dict

import numpy as np
import scipy.sparse as sp

from src.blocking.config import BlockingConfig, PassConfig


def tfidf_normalize(X: sp.csr_matrix, idf: np.ndarray, copy: bool = True) -> sp.csr_matrix:
    """Binary term matrix -> row-L2-normalised TF-IDF (empty rows stay empty).

    With copy=False the matrix's data array is overwritten (saves memory on large indexes).
    """
    Y = X.tocsr(copy=copy)
    Y.data = Y.data * idf[Y.indices]
    rows = np.repeat(np.arange(Y.shape[0]), np.diff(Y.indptr))
    norms = np.sqrt(np.bincount(rows, weights=Y.data.astype(np.float64) ** 2, minlength=Y.shape[0]))
    norms[norms == 0] = 1.0
    Y.data = (Y.data / norms[rows]).astype(np.float32)
    return Y


def _drop_columns(X: sp.csr_matrix, drop: np.ndarray) -> sp.csr_matrix:
    Y = X.copy()
    Y.data[drop[Y.indices]] = 0.0
    Y.eliminate_zeros()
    return Y


class TargetIndex:
    """Per-pass inverted indexes plus rescoring matrices for one (source, country) partition."""

    def __init__(self, view_mats: Dict[str, sp.csr_matrix], config: BlockingConfig):
        """Build the index. Takes ownership of `view_mats` (the dict is emptied to free memory)."""
        self.config = config
        n = next(iter(view_mats.values())).shape[0] if view_mats else 0
        self.n_targets = n

        # Targets without any address token (their address similarities are uninformative)
        self.has_address = (np.diff(view_mats["addr_word"].indptr) > 0) if "addr_word" in view_mats else None

        self.idf: Dict[str, np.ndarray] = {}
        self.df: Dict[str, np.ndarray] = {}
        normalized: Dict[str, sp.csr_matrix] = {}
        for view in list(view_mats):
            X = view_mats.pop(view)  # the index takes ownership: normalised in place, no copy
            df = np.bincount(X.indices, minlength=X.shape[1])
            self.df[view] = df
            self.idf[view] = (np.log((n + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
            normalized[view] = tfidf_normalize(X, self.idf[view], copy=False)

        self.postings: Dict[str, sp.csr_matrix] = {}
        for p in config.active_passes:
            blocks = []
            for view, weight in p.views.items():
                block = _drop_columns(normalized[view], self.df[view] > p.max_df)
                block.data *= np.float32(np.sqrt(weight))
                blocks.append(block)
            T = sp.hstack(blocks, format="csr") if len(blocks) > 1 else blocks[0]
            del blocks
            self.postings[p.name] = T.T.tocsr()
            del T

        # Full (unpruned) normalised target rows for exact pair rescoring
        self.rescore: Dict[str, sp.csr_matrix] = {v: normalized[v] for v in config.rescore_weights}

    def normalize_queries(self, view: str, query_view_mat: sp.csr_matrix) -> sp.csr_matrix:
        """Query rows of one view as TF-IDF (partition IDF), row-L2-normalised."""
        return tfidf_normalize(query_view_mat, self.idf[view])

    def query_matrix(self, p: PassConfig, query_view_mats: Dict[str, sp.csr_matrix]) -> sp.csr_matrix:
        """Queries in this pass's feature space, weighted with the partition's IDF."""
        blocks = []
        for view, weight in p.views.items():
            block = self.normalize_queries(view, query_view_mats[view])
            block.data *= np.float32(np.sqrt(weight))
            blocks.append(block)
        return sp.hstack(blocks, format="csr") if len(blocks) > 1 else blocks[0]

    def posting_stats(self) -> Dict[str, dict]:
        """Posting-list size diagnostics per pass (for logging/tuning)."""
        out = {}
        for name, TT in self.postings.items():
            lens = np.diff(TT.indptr)
            nz = lens[lens > 0]
            out[name] = {
                "indexed_features": int(len(nz)),
                "postings_nnz": int(TT.nnz),
                "max_posting": int(nz.max()) if len(nz) else 0,
            }
        return out
