"""
Stage 3-5 orchestration: country-partitioned candidate graphs, out-of-fold training,
decision tuning, honest evaluation and test inference.

Evaluation protocol (mirrors the leaderboard):
  * The "universe" is the whole train split: every train S1 entity with its Stage 2
    candidates from the full train S2/S3, so candidate lists and the competition
    between S1 entities for the same S2/S3 record look exactly as they will on test.
  * S1 entities are split into two folds by a hash of their id. Model A trains on
    (a sample of) fold 0 and scores fold 1, model B the reverse, so every pair has an
    out-of-fold (OOF) probability.
  * The calibrator and decision rule are fitted on fold 0 OOF predictions only. Fold 1
    is scored with them and never used for any choice: it is the reported estimate.
  * The metric is the leaderboard's: per-S1 F0.5 averaged over ALL S1 entities,
    singletons scoring 1.0 only when predicted empty, true matches outside the
    candidate set counting as misses.
Test inference averages the two fold models and applies the fitted calibrator and rule.

Streaming by country: no pair, record, competition feature or exclusivity
constraint crosses countries (Stage 2 never compares across them, and no true pair
does), so each country partition is processed on its own. Its pairs are read with a
parquet filter, its records are built, then everything is freed before the next
one. Per-query F0.5 values from the partitions combine into the exact macro
average. Peak memory is that of the largest single partition.
"""

import gc
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.blocking.data import ground_truth_pairs, load_source
from src.matching.context import context_features
from src.matching.decision import Calibrator, DecisionRule, apply_rule
from src.matching.cache import FeatureCache, fingerprint
from src.matching.features import BLOCKING_COLUMNS, feature_names, pair_features
from src.matching.model import MatcherModel, train_matcher
from src.matching.records import RecordStore
from src.matching.threshold import per_query_f05

MASK32 = np.int64(0xFFFFFFFF)
NO_TARGET = np.int64(0xFFFFFFFF)  # target code for true matches that are not among the candidates


def pair_key(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (np.asarray(q, np.int64) << 32) | (np.asarray(t, np.int64) & MASK32)


def release_memory() -> None:
    """Return freed memory to the OS (Python GC + Arrow's allocator keeps freed pages otherwise)."""
    gc.collect()
    try:
        import pyarrow as pa

        pa.default_memory_pool().release_unused()
    except Exception:  # pragma: no cover - older pyarrow
        pass


def candidate_countries(candidates_path: str) -> List[str]:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    col = pq.read_table(candidates_path, columns=["country"]).column(0)
    return [str(c) for c in pc.unique(col).to_pylist()]


# --- Universe (all S1 entities of a split) ---------------------------------------------

def orphan_drop_fraction(preprocessed_dir: str, train_split: str = "train", test_split: str = "test") -> float:
    """Fraction of train S1 entities to drop so train has test's records-per-S1 ratio.

    Test has more S2/S3 records per S1 entity than train, so more of its records
    are orphans (belong to no S1). Dropping S1 entities (their records stay)
    reproduces that ratio. Computed from file row counts, nothing hard-coded.
    """
    from src.blocking.data import preprocessed_path
    import pyarrow.parquet as pq

    def rows(split, source):
        path = preprocessed_path(preprocessed_dir, split, source)
        if path.endswith(".parquet"):
            return pq.ParquetFile(path).metadata.num_rows
        with open(path, encoding="utf-8") as fh:
            return sum(1 for _ in fh) - 1

    ratio_train = (rows(train_split, 2) + rows(train_split, 3)) / rows(train_split, 1)
    ratio_test = (rows(test_split, 2) + rows(test_split, 3)) / rows(test_split, 1)
    return float(max(0.0, 1.0 - ratio_train / ratio_test))



@dataclass
class Universe:
    s1_ids: np.ndarray       # every S1 entity of the split, in file order
    s1_country: np.ndarray   # its country label
    active: Optional[np.ndarray] = None  # S1 entities kept in the universe (None = all)

    def __post_init__(self):
        if self.active is None:
            self.active = np.ones(len(self.s1_ids), bool)

    @classmethod
    def load(cls, preprocessed_dir: str, split: str, drop_fraction: float = 0.0,
             drop_seed: int = 2026) -> "Universe":
        """All S1 entities of a split, optionally with a random `drop_fraction` removed.

        Dropping S1 entities turns their S2/S3 records into orphans (records that
        belong to no S1 entity), used to reproduce the test set's higher orphan rate
        (see orphan_drop_fraction).
        """
        s1 = load_source(preprocessed_dir, split, 1, columns=["entity_id", "country"])
        ids = s1["entity_id"].to_numpy(object)
        active = None
        if drop_fraction > 0:
            u = (pd.util.hash_array(ids, hash_key="orphans" + str(drop_seed).rjust(9, "0")) % np.uint64(10**6)) / 1e6
            active = u >= drop_fraction
        return cls(ids, s1["country"].to_numpy(object), active)

    def __len__(self) -> int:
        return len(self.s1_ids)


# --- One country's candidate graph ------------------------------------------------------

@dataclass
class CandidateGraph:
    country: str
    universe: Universe
    t_ids: np.ndarray                   # distinct candidate ids of this partition
    q: np.ndarray                       # int32: S1 universe row per pair
    t: np.ndarray                       # int32: index into t_ids per pair
    src: np.ndarray                     # int8: 0 = S2, 1 = S3
    blocking: Dict[str, np.ndarray]     # float16 Stage 2 scores / similarities
    context: Dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.q)

    @classmethod
    def load(cls, candidates_path: str, universe: Universe, country: str) -> "CandidateGraph":
        import pyarrow.parquet as pq

        id_cols = ["source1_entity_id", "candidate_entity_id"]
        present = pq.ParquetFile(candidates_path).schema.names
        cols = id_cols + [c for c in BLOCKING_COLUMNS if c in present]
        table = pq.read_table(candidates_path, columns=cols, filters=[("country", "=", country)],
                              read_dictionary=id_cols).unify_dictionaries()

        def codes(name):
            chunks = table.column(name).chunks
            if not chunks:
                return np.zeros(0, object), np.zeros(0, np.int32)
            dictionary = chunks[0].dictionary.to_numpy(zero_copy_only=False).astype(object)
            idx = np.concatenate([c.indices.to_numpy(zero_copy_only=False) for c in chunks]).astype(np.int32)
            return dictionary, idx

        q_dict, q_codes = codes("source1_entity_id")
        q_rows = pd.Index(universe.s1_ids).get_indexer(pd.Index(q_dict))
        if (q_rows < 0).any():
            raise ValueError("Candidate S1 ids missing from the preprocessed S1 file")
        t_ids, t_codes = codes("candidate_entity_id")
        blocking = {c: table.column(c).to_numpy().astype(np.float16) for c in cols[2:]}
        del table
        q = q_rows[q_codes].astype(np.int32)
        del q_codes
        keep = universe.active[q]
        if not keep.all():  # dropped S1 entities vanish; their records remain as orphans
            q, t_codes = q[keep], t_codes[keep]
            blocking = {c: v[keep] for c, v in blocking.items()}
        src = (pd.Series(t_ids, dtype=object).str.slice(0, 2).to_numpy() == "S3").astype(np.int8)
        graph = cls(country=country, universe=universe, t_ids=t_ids, q=q,
                    t=t_codes, src=src[t_codes], blocking=blocking)
        release_memory()
        ctx = context_features(graph.q, graph.t, graph.src, graph.blocking["candidate_score"],
                               graph.blocking["sim_name_char"])
        graph.context = {k: v.astype(np.float16) for k, v in ctx.items()}
        del ctx
        release_memory()
        return graph

    def query_rows(self) -> np.ndarray:
        """Universe rows of the active S1 entities labelled with this partition's country."""
        return np.flatnonzero((self.universe.s1_country == self.country) & self.universe.active)


@dataclass
class Partition:
    """One country partition: its graph plus lazily built records and an optional feature cache.

    The RecordStore (record strings / token matrices) is built only when a feature
    has to be computed, so a run whose features are all cached never builds it.
    """
    graph: CandidateGraph
    preprocessed_dir: str
    split: str
    n_jobs: int = 2
    cache: Optional[FeatureCache] = None
    precision: str = "float32"  # feature precision the scoring models were trained on
    _store: Optional[RecordStore] = None
    q_local: Optional[np.ndarray] = None
    t_local: Optional[np.ndarray] = None

    @classmethod
    def build(cls, graph: CandidateGraph, preprocessed_dir: str, split: str, n_jobs: int = 2,
              cache: Optional[FeatureCache] = None, precision: str = "float32") -> "Partition":
        if precision not in ("float32", "float16"):
            raise ValueError(f"unknown feature precision {precision!r}")
        return cls(graph, preprocessed_dir, split, n_jobs, cache, precision)

    @property
    def store(self) -> RecordStore:
        if self._store is None:
            graph = self.graph
            uq, q_inv = np.unique(graph.q, return_inverse=True)
            ut, t_inv = np.unique(graph.t, return_inverse=True)
            frames = []
            for source, ids in ((1, graph.universe.s1_ids[uq]), (2, graph.t_ids[ut]), (3, graph.t_ids[ut])):
                df = load_source(self.preprocessed_dir, self.split, source)
                frames.append(df[df["entity_id"].isin(pd.Index(ids))])
                del df
            frame = pd.concat(frames, ignore_index=True)
            del frames
            release_memory()
            self._store = RecordStore(frame, n_jobs=self.n_jobs)
            del frame
            self.q_local = self._store.rows(graph.universe.s1_ids[uq])[q_inv]
            self.t_local = self._store.rows(graph.t_ids[ut])[t_inv]
            release_memory()
        return self._store

    def _live_features(self, idx: np.ndarray) -> pd.DataFrame:
        store = self.store
        return pair_features(store, self.q_local[idx], self.t_local[idx])

    def _at_precision(self, base: pd.DataFrame) -> pd.DataFrame:
        """Round live features exactly as the cache stores them when the models were trained on float16."""
        if self.precision == "float16":
            for c in base.columns:
                base[c] = base[c].to_numpy(np.float32).astype(np.float16).astype(np.float32)
        return base

    def _base_features(self, idx: np.ndarray) -> pd.DataFrame:
        """Record-based pair features at the models' training precision (from the cache when valid)."""
        if self.precision == "float16" and self.cache is not None and self.cache.complete:
            return self.cache.load(idx)
        return self._at_precision(self._live_features(idx))

    def ensure_cache(self, log: Callable = print) -> None:
        """Compute and store the record-based features of every pair once (no-op when cached)."""
        if self.cache is None or self.cache.complete:
            return
        store = self.store
        self.cache.build(len(self.graph), lambda idx: pair_features(store, self.q_local[idx], self.t_local[idx]),
                         log=log, label=f"{self.split}/{self.graph.country}")

    def features(self, idx: np.ndarray, base: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Feature frame for pair positions `idx` of the partition graph."""
        g = self.graph
        frame = self._base_features(idx) if base is None else base
        for k, v in g.blocking.items():
            frame[k] = np.asarray(v[idx], dtype=np.float32)
        for k, v in g.context.items():
            frame[k] = np.asarray(v[idx], dtype=np.float32)
        frame["is_s2"] = (g.src[idx] == 0).astype(np.float32)
        return frame

    def score(self, models_for: Callable, chunk: int = 2_000_000, log: Callable = print) -> np.ndarray:
        """Probabilities for every pair; `models_for(idx)` -> list of (mask over idx, models to average)."""
        n = len(self.graph)
        out = np.empty(n, np.float32)
        # Write-through: when the cache is missing, the live features computed for scoring fill it too
        writer = self.cache.writer(n) if self.cache is not None and not self.cache.complete else None
        t0 = time.time()
        for a in range(0, n, chunk):
            idx = np.arange(a, min(a + chunk, n))
            base = None
            if writer is not None:
                base = self._live_features(idx)
                writer.write(idx, base)
                base = self._at_precision(base)
            X = self.features(idx, base)
            for mask, models in models_for(idx):
                if mask.any():
                    out[idx[mask]] = np.mean([m.predict_proba(X[mask]) for m in models], axis=0)
            del X, base
            log(f"[matching]   {self.graph.country}: scored {idx[-1] + 1:,}/{n:,} pairs ({time.time() - t0:.0f}s)")
        if writer is not None:
            writer.publish()
        return out


def partition_cache(cache_root: Optional[str], candidates_path: str, preprocessed_dir: str, split: str,
                    graph: CandidateGraph, drop_fraction: float = 0.0) -> Optional[FeatureCache]:
    """Feature cache for a partition graph, or None when caching is disabled."""
    if not cache_root:
        return None
    key = fingerprint(candidates_path, preprocessed_dir, split, graph.country, drop_fraction)
    return FeatureCache(cache_root, split, graph.country, key, feature_names([]))


# --- Ground truth and per-partition results ------------------------------------------------

def truth_for(graph: CandidateGraph, gt_pairs: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """(query rows, pair keys) of the true pairs of the partition's S1 entities."""
    q = pd.Index(graph.universe.s1_ids).get_indexer(pd.Index(gt_pairs["source1_entity_id"]))
    keep = q >= 0
    keep[keep] = (graph.universe.s1_country[q[keep]] == graph.country) & graph.universe.active[q[keep]]
    q = q[keep].astype(np.int64)
    t = pd.Index(graph.t_ids).get_indexer(pd.Index(gt_pairs["candidate_entity_id"].to_numpy()[keep])).astype(np.int64)
    t[t < 0] = NO_TARGET
    return q, pair_key(q, t)


@dataclass
class PartitionResult:
    """Compact per-partition arrays for tuning / evaluation (no strings, no features)."""
    country: str
    q: np.ndarray           # int32 universe row per pair
    t: np.ndarray           # int32 partition-local candidate code per pair
    prob: np.ndarray        # float32 model probability (OOF on train)
    truth_q: np.ndarray     # int64 universe row per true pair
    truth_key: np.ndarray   # int64 pair key per true pair
    query_rows: np.ndarray  # universe rows of the partition's S1 entities

    def labels(self) -> np.ndarray:
        return np.isin(pair_key(self.q, self.t), self.truth_key)

    def save(self, path: str) -> None:
        np.savez(path, country=self.country, q=self.q, t=self.t, prob=self.prob, truth_q=self.truth_q,
                 truth_key=self.truth_key, query_rows=self.query_rows)

    @classmethod
    def load(cls, path: str) -> "PartitionResult":
        d = np.load(path, allow_pickle=False)
        return cls(str(d["country"]), d["q"], d["t"], d["prob"], d["truth_q"], d["truth_key"], d["query_rows"])


def per_query_scores(res: PartitionResult, selected: np.ndarray, n_universe: int) -> np.ndarray:
    """Per-query F0.5 over the partition's S1 entities (universe-length array, NaN elsewhere)."""
    sel_keys = pair_key(res.q[selected], res.t[selected])
    f = per_query_f05(n_universe, res.q[selected], sel_keys, res.truth_q, res.truth_key)
    out = np.full(n_universe, np.nan)
    out[res.query_rows] = f[res.query_rows]
    return out


def evaluate(results: Sequence[PartitionResult], selections: Sequence[np.ndarray], n_universe: int,
             query_mask: np.ndarray, universe_country: np.ndarray, detailed: bool = True) -> Dict:
    """Leaderboard metric over the S1 entities in `query_mask` (all partitions combined).

    S1 entities not covered by any partition (no candidates at all) score 1.0 if
    they are true singletons and 0.0 otherwise, exactly as the leaderboard would.
    """
    f = np.full(n_universe, np.nan)
    n_true = np.zeros(n_universe, np.int64)
    n_pred = np.zeros(n_universe, np.int64)
    ceiling = np.full(n_universe, np.nan)
    hits_total = pred_total = 0
    for res, sel in zip(results, selections):
        fq = per_query_scores(res, sel, n_universe)
        f[res.query_rows] = fq[res.query_rows]
        n_true += np.bincount(res.truth_q, minlength=n_universe)
        n_pred += np.bincount(res.q[sel], minlength=n_universe)
        if detailed:
            keys = pair_key(res.q, res.t)
            is_true = np.isin(keys, res.truth_key)
            cq = per_query_scores(res, is_true, n_universe)
            ceiling[res.query_rows] = cq[res.query_rows]
            in_mask = query_mask[res.q[sel]]
            hits_total += int(np.isin(keys[sel][in_mask], res.truth_key).sum())
            pred_total += int(in_mask.sum())
    uncovered = np.isnan(f)
    f[uncovered] = (n_true[uncovered] == 0).astype(float)
    ceiling[np.isnan(ceiling)] = f[np.isnan(ceiling)]

    rows = np.flatnonzero(query_mask)
    fr = f[rows]
    report = {"queries": int(len(rows)), "macro_f05": float(fr.mean())}
    if not detailed:
        return report
    singles = n_true[rows] == 0
    country = universe_country[rows]
    report.update({
        "macro_f05_ceiling": float(ceiling[rows].mean()),
        "pair_precision": hits_total / max(pred_total, 1),
        "pair_recall": hits_total / max(int(n_true[rows].sum()), 1),
        "singleton_share": float(singles.mean()),
        "singleton_accuracy": float((n_pred[rows][singles] == 0).mean()) if singles.any() else 1.0,
        "matched_query_f05": float(fr[~singles].mean()) if (~singles).any() else 1.0,
        "predicted_empty_share": float((n_pred[rows] == 0).mean()),
        "mean_predicted_per_query": float(n_pred[rows].mean()),
        "macro_f05_by_country": {str(c): float(fr[country == c].mean()) for c in pd.unique(country)},
    })
    return report


# --- Training ------------------------------------------------------------------------

def query_folds(s1_ids: np.ndarray, n_folds: int = 2) -> np.ndarray:
    """Deterministic fold per S1 entity from a hash of its id."""
    return (pd.util.hash_array(np.asarray(s1_ids, dtype=object)) % np.uint64(n_folds)).astype(np.int8)


def sample_roles(universe: Universe, folds: np.ndarray, n_train_queries: int, es_fraction: float,
                 seed: int) -> np.ndarray:
    """Per S1 entity: 0 unused, 1 training, 2 early stopping (n_train_queries per fold)."""
    rng = np.random.default_rng(seed)
    role = np.zeros(len(universe), np.int8)
    for f in (0, 1):
        pool = np.flatnonzero((folds == f) & universe.active)
        sample = rng.choice(pool, size=min(n_train_queries, len(pool)), replace=False)
        n_es = max(1, int(len(sample) * es_fraction))
        role[sample[:n_es]] = 2
        role[sample[n_es:]] = 1
    return role


def train_oof(
    candidates_path: str,
    preprocessed_dir: str,
    gt: pd.DataFrame,
    split: str = "train",
    n_train_queries: int = 120_000,
    es_fraction: float = 0.1,
    seed: int = 42,
    params: Optional[dict] = None,
    n_jobs: int = 2,
    results_dir: Optional[str] = None,
    log: Callable = print,
    drop_fraction: float = 0.0,
    num_boost_round: int = 1000,
    cache_root: Optional[str] = None,
):
    """Train one model per fold, then score every pair out-of-fold, one country at a time.

    Returns (models, universe, folds, [PartitionResult per country]).
    """
    # With a feature cache the models are trained (and later scored) on its float16 values
    precision = "float16" if cache_root else "float32"
    universe = Universe.load(preprocessed_dir, split, drop_fraction=drop_fraction)
    if drop_fraction > 0:
        log(f"[matching] test-like universe: {int((~universe.active).sum()):,} of {len(universe):,} S1 "
            f"entities dropped ({drop_fraction:.1%}); their records become orphans")
    folds = query_folds(universe.s1_ids)
    role = sample_roles(universe, folds, n_train_queries, es_fraction, seed)
    gt_pairs = ground_truth_pairs(gt)
    countries = candidate_countries(candidates_path)

    samples: Dict[Tuple[int, int], List[pd.DataFrame]] = {}
    sample_y: Dict[Tuple[int, int], List[np.ndarray]] = {}
    for country in countries:
        t0 = time.time()
        graph = CandidateGraph.load(candidates_path, universe, country)
        tq, tkey = truth_for(graph, gt_pairs)
        labels = np.isin(pair_key(graph.q, graph.t), tkey)
        part = Partition.build(graph, preprocessed_dir, split, n_jobs,
                               partition_cache(cache_root, candidates_path, preprocessed_dir, split, graph,
                                               drop_fraction), precision)
        part.ensure_cache(log)  # all pairs once; later passes and runs read the cache
        idx = np.flatnonzero(role[graph.q] > 0)
        X = part.features(idx)
        pair_fold, pair_role = folds[graph.q[idx]], role[graph.q[idx]]
        for f in (0, 1):
            for r in (1, 2):
                m = (pair_fold == f) & (pair_role == r)
                samples.setdefault((f, r), []).append(X[m])
                sample_y.setdefault((f, r), []).append(labels[idx[m]])
        log(f"[matching] {country}: {len(graph):,} pairs, training features for {len(idx):,} "
            f"({time.time() - t0:.0f}s)")
        del graph, part, X, labels
        release_memory()

    models = []
    for f in (0, 1):
        X_tr, y_tr = pd.concat(samples.pop((f, 1)), ignore_index=True), np.concatenate(sample_y.pop((f, 1)))
        X_es, y_es = pd.concat(samples.pop((f, 2)), ignore_index=True), np.concatenate(sample_y.pop((f, 2)))
        log(f"[matching] fold {f}: training on {len(X_tr):,} pairs ({y_tr.mean():.2%} positive), "
            f"early stopping on {len(X_es):,}")
        model = train_matcher(X_tr, y_tr.astype(int), X_es, y_es.astype(int), params=params,
                              num_boost_round=num_boost_round)
        log(f"[matching] fold {f}: {model.booster.best_iteration or model.booster.current_iteration()} trees")
        models.append(model)
        del X_tr, X_es
        release_memory()

    results = []
    for country in countries:
        graph = CandidateGraph.load(candidates_path, universe, country)
        tq, tkey = truth_for(graph, gt_pairs)
        part = Partition.build(graph, preprocessed_dir, split, n_jobs,
                               partition_cache(cache_root, candidates_path, preprocessed_dir, split, graph,
                                               drop_fraction), precision)
        fold_of = folds[graph.q]
        log(f"[matching] OOF scoring {country}: fold-1 model on fold 0, fold-0 model on fold 1")
        prob = part.score(lambda idx: [(fold_of[idx] == 0, [models[1]]), (fold_of[idx] == 1, [models[0]])], log=log)
        res = PartitionResult(country, graph.q, graph.t, prob, tq, tkey, graph.query_rows())
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            res.save(os.path.join(results_dir, f"oof_{country}.npz"))
        results.append(res)
        del graph, part
        release_memory()
    return models, universe, folds, results


# --- Decision tuning --------------------------------------------------------------------

def default_rule_grid(missing_mass_estimate: float) -> List[DecisionRule]:
    grid = []
    for owner in (True, False):
        for exclusive in (True, False):
            for m in sorted({0.0, round(missing_mass_estimate, 4), 0.1, 0.2}):
                grid.append(DecisionRule("expected_f", missing_mass=m, owner_normalize=owner, exclusive=exclusive))
            for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                grid.append(DecisionRule("threshold", threshold=thr, owner_normalize=owner, exclusive=exclusive))
    return grid


def select_all(rule: DecisionRule, results: Sequence[PartitionResult], probs: Sequence[np.ndarray],
               caches: Optional[List[dict]] = None) -> List[np.ndarray]:
    caches = caches if caches is not None else [None] * len(results)
    return [apply_rule(rule, r.q, r.t, p, cache=c) for r, p, c in zip(results, probs, caches)]


def tune_decision(results: Sequence[PartitionResult], probs: Sequence[np.ndarray], universe: Universe,
                  query_mask: np.ndarray, grid: List[DecisionRule], log: Callable = print):
    """Best rule (macro F0.5 over `query_mask`) and the score of every rule tried."""
    caches = [{} for _ in results]
    scored = []
    for rule in grid:
        sel = select_all(rule, results, probs, caches)
        score = evaluate(results, sel, len(universe), query_mask, universe.s1_country, detailed=False)["macro_f05"]
        scored.append((score, rule))
        log(f"[matching]   {score:.5f}  {rule.to_dict()}")
    best_score, best_rule = max(scored, key=lambda r: r[0])
    return best_rule, [{"macro_f05": s, **r.to_dict()} for s, r in scored]


# --- Test inference and output -----------------------------------------------------------

def write_rows(fh, s1_ids: np.ndarray, rows: np.ndarray, pair_q: np.ndarray, pair_ids: np.ndarray,
               order_score: np.ndarray) -> None:
    """Append one line per S1 universe row in `rows`: id, tab, ids joined by commas (best first)."""
    order = np.lexsort((-np.asarray(order_score, np.float64), pair_q))
    q_sorted = np.asarray(pair_q)[order]
    ids_sorted = np.asarray(pair_ids, dtype=object)[order]
    lo = np.searchsorted(q_sorted, rows, side="left")
    hi = np.searchsorted(q_sorted, rows, side="right")
    for a in range(0, len(rows), 200_000):
        fh.writelines(f"{s1_ids[r]}\t{','.join(ids_sorted[l:h])}\n"
                      for r, l, h in zip(rows[a:a + 200_000], lo[a:a + 200_000], hi[a:a + 200_000]))


def write_id_lists(path: str, header: List[str], s1_ids: np.ndarray, pair_q: np.ndarray,
                   pair_ids: np.ndarray, order_score: np.ndarray) -> None:
    """One row per S1 entity (in `s1_ids` order): id, tab, ids joined by commas (best first)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(header) + "\n")
        write_rows(fh, s1_ids, np.arange(len(s1_ids)), pair_q, pair_ids, order_score)


def predict_and_write(
    candidates_path: str,
    preprocessed_dir: str,
    split: str,
    models: List[MatcherModel],
    calibrator: Calibrator,
    rule: DecisionRule,
    matching_path: str,
    candidate_path: str,
    n_jobs: int = 2,
    log: Callable = print,
    stack_models: Optional[List[MatcherModel]] = None,
    cache_root: Optional[str] = None,
    precision: str = "float32",
    stack_support: bool = False,
) -> Dict:
    """Score every candidate pair (country by country) and write both submission files.

    Row order in the files follows the partitions. The submission rules only
    require exactly one row per S1 entity. S1 entities without any candidate get
    an empty row in both files.
    """
    universe = Universe.load(preprocessed_dir, split)
    written = np.zeros(len(universe), bool)
    stats = {}
    for p in (matching_path, candidate_path):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(matching_path, "w", encoding="utf-8", newline="\n") as fm, \
            open(candidate_path, "w", encoding="utf-8", newline="\n") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for country in candidate_countries(candidates_path):
            graph = CandidateGraph.load(candidates_path, universe, country)
            part = Partition.build(graph, preprocessed_dir, split, n_jobs,
                                   partition_cache(cache_root, candidates_path, preprocessed_dir, split, graph),
                                   precision)
            prob = part.score(lambda idx: [(np.ones(len(idx), bool), models)], log=log)
            if stack_models:
                from src.matching.stacking import apply_stack, support_features

                extra = None
                if stack_support:
                    store = part.store  # records + token matrices (also sets part.t_local)
                    extra = support_features(graph.q, prob, part.t_local, store)
                prob = apply_stack(stack_models, graph.q, graph.t, prob, extra)
            prob = calibrator.transform(prob)
            selected = apply_rule(rule, graph.q, graph.t, prob)
            rows = np.unique(graph.q)
            write_rows(fm, universe.s1_ids, rows, graph.q[selected], graph.t_ids[graph.t[selected]], prob[selected])
            write_rows(fc, universe.s1_ids, rows, graph.q, graph.t_ids[graph.t], prob)
            written[rows] = True
            n_pred = np.bincount(graph.q[selected], minlength=len(universe))[rows]
            stats[country] = {"s1": int(len(rows)), "pairs": int(len(graph)),
                              "predicted_empty_share": float((n_pred == 0).mean()),
                              "mean_matches": float(n_pred.mean())}
            log(f"[matching] {country}: {len(rows):,} S1, empty {stats[country]['predicted_empty_share']:.2%}, "
                f"matches/S1 {stats[country]['mean_matches']:.2f}")
            del graph, part, prob, selected
            release_memory()
        missing = np.flatnonzero(~written)
        fm.writelines(f"{universe.s1_ids[r]}\t\n" for r in missing)
        fc.writelines(f"{universe.s1_ids[r]}\t\n" for r in missing)
    stats["s1_without_candidates"] = int(len(missing))
    return stats
