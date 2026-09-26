"""
Optional dense-retrieval pass (off by default: `DenseConfig.enabled`).

Encodes "name | address" with a small Apache-2.0 multilingual sentence encoder
(default: paraphrase-multilingual-MiniLM-L12-v2, 118M params) and retrieves the
nearest targets per country/source partition with FAISS inner-product search.
The pass targets semantic and transliteration variants that n-gram overlap misses.
Its results are fused with the sparse passes like any other pass.

Cost: CPU encoding runs at roughly 0.5-2k records/s, so a full test run adds
hours. Enable it on a GPU machine or for targeted experiments.

Requires: `pip install sentence-transformers` (FAISS: `faiss-cpu`).
"""

from typing import Callable, Optional, Sequence

import numpy as np

from src.blocking.config import DenseConfig
from src.blocking.topk import rank_within_groups

Encoder = Callable[[Sequence[str]], np.ndarray]


def record_texts(names: Sequence[str], addresses: Sequence[str]) -> list:
    return [f"{n} | {a}" if a else n for n, a in zip(names, addresses)]


class DenseRetriever:
    """Dense index over one target partition. `encoder` can be injected (tests, custom models)."""

    def __init__(self, config: DenseConfig, encoder: Optional[Encoder] = None):
        self.config = config
        self._encoder = encoder
        self.index = None

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "The dense blocking pass needs sentence-transformers: pip install sentence-transformers"
                ) from exc
            model = SentenceTransformer(self.config.model_name, device=self.config.device)
            self._encoder = lambda xs: model.encode(
                list(xs), batch_size=self.config.batch_size, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=len(xs) > 50_000,
            )
        emb = np.asarray(self._encoder(texts), dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return np.ascontiguousarray(emb / norms)

    def build(self, names: Sequence[str], addresses: Sequence[str]) -> None:
        import faiss

        emb = self._encode(record_texts(names, addresses))
        n, d = emb.shape
        if n < 50_000:
            index = faiss.IndexFlatIP(d)
        else:
            nlist = int(4 * np.sqrt(n))
            index = faiss.index_factory(d, f"IVF{nlist},SQ8", faiss.METRIC_INNER_PRODUCT)
            rng = np.random.default_rng(0)
            sample = emb[rng.choice(n, size=min(n, 50 * nlist), replace=False)]
            index.train(sample)
            index.nprobe = 16
        index.add(emb)
        self.index = index

    def search(self, names: Sequence[str], addresses: Sequence[str]):
        """(query, target, score, rank) arrays in the same layout as sparse_topk."""
        k = min(self.config.k, self.index.ntotal)
        emb = self._encode(record_texts(names, addresses))
        scores, targets = self.index.search(emb, k)
        q = np.repeat(np.arange(len(emb), dtype=np.int32), k)
        t = targets.ravel().astype(np.int32)
        s = scores.ravel().astype(np.float32)
        keep = (t >= 0) & (s >= self.config.min_score)
        q, t, s = q[keep], t[keep], s[keep]
        order, rank = rank_within_groups(q, s)
        return q[order], t[order], s[order], rank
