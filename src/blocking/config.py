"""
Configuration for the blocking / candidate-generation stage.

A blocking run has two steps:

1. Retrieval passes. Each pass scores (S1 query, S2/S3 target) pairs by TF-IDF
   cosine over a weighted combination of feature views (see features.py),
   indexing only features rarer than `max_df`, and keeps the top-k targets per
   query and per source. The union of all passes is the retrieved set.
2. Rescoring. Every retrieved pair gets exact cosines on the `rescore_weights`
   views. Their weighted sum ranks the candidates, and the best
   `max_candidates_per_source` per source are kept.

Defaults were chosen on the local validation split (see README, Stage 2) and can
be overridden with a JSON file: `python scripts/run_blocking.py --config my.json`.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class PassConfig:
    name: str
    views: Dict[str, float]      # view name -> weight (weights are squared-root applied per side)
    k: int = 20                  # top-k targets per query per source
    min_score: float = 0.1       # drop retrieved pairs below this cosine
    max_df: int = 3_000          # features whose target document frequency exceeds this are not indexed
    enabled: bool = True


@dataclass
class DenseConfig:
    enabled: bool = False
    # Apache-2.0 multilingual sentence encoder (handles FR + transliterated IN names)
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    k: int = 10
    min_score: float = 0.5
    batch_size: int = 256
    device: Optional[str] = None


def default_passes() -> List[PassConfig]:
    return [
        # Name keys: exact tokens, unordered token pairs (word-order noise), phonetic
        # key pairs (transliteration), compact-name prefix (domain names / joined words)
        PassConfig(
            name="name",
            views={"name_word": 1.0, "name_pair": 1.0, "name_pkey_pair": 1.0, "name_compact": 0.5},
            k=50, min_score=0.05, max_df=1_000,
        ),
        # Address keys: tokens and adjacent bigrams ('10528 cedar', 'cedar falls').
        # Catches DBA / random trade names registered at the same address
        PassConfig(
            name="addr",
            views={"addr_word": 1.0, "addr_bigram": 1.0},
            k=50, min_score=0.05, max_df=1_000,
        ),
        # Cross keys (name token x address token): survive heavy noise on either side
        # and separate branches of chains / common names by location
        PassConfig(
            name="name_addr",
            views={"name_addr": 1.0, "name_pair": 0.5, "addr_bigram": 0.5},
            k=50, min_score=0.05, max_df=1_000,
        ),
    ]


def default_rescore_weights() -> Dict[str, float]:
    # Logistic-regression coefficients fitted on validation candidates (true pair vs not),
    # then applied as a linear ranking score. Reproduce / refit with scripts/fit_rescore_weights.py.
    return {
        "name_char": 2.81,
        "name_phon": 4.33,
        "addr_word": 8.45,
        "addr_bigram": 2.14,
        "name_addr": 2.95,
    }


@dataclass
class BlockingConfig:
    passes: List[PassConfig] = field(default_factory=default_passes)
    dense: DenseConfig = field(default_factory=DenseConfig)

    # Exact-similarity views used to rank the retrieved union, with their weights
    rescore_weights: Dict[str, float] = field(default_factory=default_rescore_weights)
    # Final budget: at most this many candidates per S1 entity from EACH of S2 and S3
    max_candidates_per_source: int = 25
    # Candidates whose rescored score is below this are dropped (0 keeps everything retrieved)
    min_candidate_score: float = 0.0

    char_ngram: int = 3
    hash_bits: int = 22

    # Execution
    n_threads: int = 12               # threads for sparse top-k (scipy releases the GIL)
    n_jobs: int = 8                   # processes for feature hashing
    query_block_size: int = 200_000   # queries processed per block (bounds result memory)
    max_product_nnz: int = 15_000_000 # per-thread cap on sparse product size (bounds peak memory)

    @property
    def active_passes(self) -> List[PassConfig]:
        return [p for p in self.passes if p.enabled]

    @property
    def views(self) -> List[str]:
        """Every feature view needed by the retrieval passes and the rescoring step."""
        pass_views = [v for p in self.active_passes for v in p.views]
        return list(dict.fromkeys(pass_views + list(self.rescore_weights)))

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "BlockingConfig":
        d = dict(d)
        passes = [PassConfig(**p) for p in d.pop("passes", [])] or default_passes()
        dense = DenseConfig(**d.pop("dense", {}))
        return cls(passes=passes, dense=dense, **d)

    @classmethod
    def load(cls, path: Optional[str]) -> "BlockingConfig":
        if not path:
            return cls()
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
