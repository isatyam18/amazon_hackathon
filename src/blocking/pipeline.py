"""
Stage 2 orchestration: country-partitioned candidate generation.

For every country label present among the queries (open set: US, India,
France, anything else) and every target source (S2, S3):

  1. hash the partition's targets into feature views and build a TF-IDF index
     (IDF fitted on that partition only),
  2. for each memory-bounded block of that country's queries:
       a. run the retrieval passes (sparse top-k over rare keys, optional dense),
       b. take the union of retrieved pairs,
       c. rescore every pair with exact cosines on the rescoring views,
       d. keep the best `max_candidates_per_source` targets per query.

Queries never meet targets with a different country label. On the training data
no true pair crosses countries (0 of 7.6M), so this loses no recall and cuts the
search space. If a source has no targets for a query's country, those queries
are searched against the whole source instead of being dropped.
"""

import time
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.blocking.candidates import CandidateSet
from src.blocking.config import BlockingConfig
from src.blocking.dense import DenseRetriever, Encoder
from src.blocking.features import featurize
from src.blocking.fusion import select_top, union_results
from src.blocking.index import TargetIndex
from src.blocking.rescore import candidate_score, pair_cosine
from src.blocking.topk import sparse_topk


class BlockingPipeline:
    def __init__(
        self,
        config: Optional[BlockingConfig] = None,
        log: Callable[[str], None] = print,
        dense_encoder: Optional[Encoder] = None,
    ):
        self.config = config or BlockingConfig()
        self.log = log
        self.dense_encoder = dense_encoder
        self.stats: List[dict] = []

    @property
    def pass_names(self) -> List[str]:
        names = [p.name for p in self.config.active_passes]
        if self.config.dense.enabled:
            names.append("dense")
        return names

    @property
    def feature_names(self) -> List[str]:
        return [f"score_{p}" for p in self.pass_names] + [f"sim_{v}" for v in self.config.rescore_weights]

    def featurize(self, df: pd.DataFrame, rows: np.ndarray):
        cfg = self.config
        return featurize(
            df["clean_name"].to_numpy(object)[rows],
            df["clean_address"].to_numpy(object)[rows],
            cfg.views,
            hash_bits=cfg.hash_bits,
            ngram=cfg.char_ngram,
            n_jobs=cfg.n_jobs,
        )

    def build_index(self, targets: pd.DataFrame, rows: np.ndarray) -> TargetIndex:
        return TargetIndex(self.featurize(targets, rows), self.config)

    def _timed(self, name: str, n_queries: int, fn):
        t0 = time.time()
        res = fn()
        self.stats.append({"step": name, "queries": n_queries, "seconds": time.time() - t0,
                           "pairs": int(len(res[0])) if isinstance(res, tuple) else int(len(res))})
        return res

    def retrieve(self, index: TargetIndex, query_feats: Dict, dense: Optional[DenseRetriever] = None,
                 query_text=None) -> Dict[str, tuple]:
        """Run every retrieval pass for one block of featurized queries."""
        cfg = self.config
        n = next(iter(query_feats.values())).shape[0]
        results = {}
        for p in cfg.active_passes:
            results[p.name] = self._timed(p.name, n, lambda: sparse_topk(
                index.query_matrix(p, query_feats), index.postings[p.name], k=p.k, min_score=p.min_score,
                n_threads=cfg.n_threads, max_product_nnz=cfg.max_product_nnz,
            ))
        if dense is not None and query_text is not None:
            results["dense"] = self._timed("dense", n, lambda: dense.search(*query_text))
        return results

    def score_block(self, index: TargetIndex, query_feats: Dict, results: Dict[str, tuple]):
        """Union the pass results, rescore the pairs exactly and cut to the per-source budget."""
        cfg = self.config
        union = union_results(results, self.pass_names)
        sims = {}
        for view in cfg.rescore_weights:
            Qn = index.normalize_queries(view, query_feats[view])
            sims[view] = pair_cosine(Qn, index.rescore[view], union.query, union.target)
        missing_address = None
        if index.has_address is not None and "addr_word" in query_feats:
            q_has = np.diff(query_feats["addr_word"].indptr) > 0
            missing_address = ~(q_has[union.query] & index.has_address[union.target])
        score = candidate_score(sims, cfg.rescore_weights, missing_address)
        keep, rank = select_top(union.query, score, cfg.max_candidates_per_source, cfg.min_candidate_score)
        features = np.hstack([union.pass_scores] + [sims[v][:, None] for v in cfg.rescore_weights])
        return union.query[keep], union.target[keep], score[keep], rank, features[keep]

    def run(self, queries: pd.DataFrame, targets: Dict[str, pd.DataFrame]) -> CandidateSet:
        """Generate candidates for every query row.

        queries: preprocessed Source 1 frame (entity_id, clean_name, clean_address, country)
        targets: {"S2": frame, "S3": frame} in the same schema
        """
        cfg = self.config
        source_names = list(targets.keys())
        q_country = queries["country"].to_numpy(object)
        t_country = {s: targets[s]["country"].to_numpy(object) for s in source_names}

        out = {k: [] for k in ("q", "src", "t", "score", "rank", "features")}
        t_start = time.time()
        for country in sorted(pd.unique(q_country)):
            q_rows = np.flatnonzero(q_country == country)
            self.log(f"[blocking] country={country!r}: {len(q_rows):,} queries")
            q_feats = self.featurize(queries, q_rows)

            for code, src in enumerate(source_names):
                t_rows = np.flatnonzero(t_country[src] == country)
                if len(t_rows) == 0:
                    self.log(f"[blocking]   no {src} targets labelled {country!r}; searching all {src} targets")
                    t_rows = np.arange(len(targets[src]))
                if len(t_rows) == 0:
                    continue

                t0 = time.time()
                index = self.build_index(targets[src], t_rows)
                dense = None
                if cfg.dense.enabled:
                    dense = DenseRetriever(cfg.dense, encoder=self.dense_encoder)
                    dense.build(targets[src]["clean_name"].to_numpy(object)[t_rows],
                                targets[src]["clean_address"].to_numpy(object)[t_rows])
                self.log(f"[blocking]   {src}: indexed {len(t_rows):,} targets in {time.time() - t0:.1f}s")

                t0 = time.time()
                n_pairs = 0
                for b in range(0, len(q_rows), cfg.query_block_size):
                    blk = slice(b, b + cfg.query_block_size)
                    block_feats = {v: m[blk] for v, m in q_feats.items()}
                    query_text = None
                    if dense is not None:
                        rows = q_rows[blk]
                        query_text = (queries["clean_name"].to_numpy(object)[rows],
                                      queries["clean_address"].to_numpy(object)[rows])
                    results = self.retrieve(index, block_feats, dense, query_text)
                    qi, ti, score, rank, features = self.score_block(index, block_feats, results)
                    out["q"].append(q_rows[b + qi].astype(np.int32))
                    out["src"].append(np.full(len(qi), code, np.int8))
                    out["t"].append(t_rows[ti].astype(np.int32))
                    out["score"].append(score)
                    out["rank"].append(rank.astype(np.int16))
                    out["features"].append(features.astype(np.float16))  # stored as float16 in parquet anyway
                    n_pairs += len(qi)
                    done = min(b + cfg.query_block_size, len(q_rows))
                    if done < len(q_rows):
                        self.log(f"[blocking]     {src}: {done:,}/{len(q_rows):,} queries "
                                 f"({time.time() - t0:.0f}s)")
                self.log(f"[blocking]   {src}: {n_pairs:,} candidate pairs "
                         f"({n_pairs / max(len(q_rows), 1):.1f}/query) in {time.time() - t0:.1f}s")
                del index, dense

        self.log(f"[blocking] done in {(time.time() - t_start) / 60:.1f} min")

        def cat(key, dtype, shape=(0,)):
            return np.concatenate(out[key]) if out[key] else np.zeros(shape, dtype)

        return CandidateSet(
            query_ids=queries["entity_id"].to_numpy(object),
            query_country=q_country,
            source_names=source_names,
            target_ids={s: targets[s]["entity_id"].to_numpy(object) for s in source_names},
            feature_names=self.feature_names,
            q=cat("q", np.int32),
            src=cat("src", np.int8),
            t=cat("t", np.int32),
            score=cat("score", np.float32),
            source_rank=cat("rank", np.int16),
            features=cat("features", np.float16, (0, len(self.feature_names))),
        )
