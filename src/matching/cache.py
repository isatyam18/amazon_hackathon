"""
Disk cache for the record-based pair features (the expensive part of Stage 3).

The ~30 string / token / house-number features of a pair depend only on the two
records, never on the model, the universe or the decision rule. Computing them
for 50-70M pairs takes most of a training or inference run, so they are
computed once per (split, country) and stored as one memory-mapped float16
.npy file per column. Context and Stage 2 columns are cheap and stay live.

Safety:
  * The cache directory is keyed by a fingerprint of everything that determines
    the values: the feature / normalisation source files, the candidate parquet
    and preprocessed inputs (size + mtime), the split, country and universe
    settings. Changing any of them selects a new, empty cache, so stale features
    are never used. Older fingerprints of the same split/country are deleted.
  * A cache only counts once complete: columns are written into a temporary
    directory that is renamed at the end, so an interrupted build leaves nothing.
  * Precision consistency: a model is always scored at the feature precision it
    was trained on (recorded as `feature_precision` in matching_config.json).
    Models trained from this cache see float16 values at training and at
    inference (from the cache, or live features rounded identically). Models
    trained on live float32 features are always scored on live float32
    features; the cache is only written alongside for later runs.
"""

import hashlib
import json
import os
import shutil
import time
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

CACHE_FORMAT = "v1"

# Source files whose code determines the cached feature values
_SOURCE_FILES = [
    "src/matching/features.py",
    "src/matching/records.py",
    "src/blocking/normalize.py",
    "src/blocking/features.py",
    "src/blocking/rescore.py",
    "src/phonetic.py",
]


def _root_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def fingerprint(candidates_path: str, preprocessed_dir: str, split: str, country: str,
                drop_fraction: float = 0.0) -> str:
    from src.blocking.data import preprocessed_path

    h = hashlib.sha1(CACHE_FORMAT.encode())
    root = _root_dir()
    for rel in _SOURCE_FILES:
        with open(os.path.join(root, rel), "rb") as fh:
            h.update(fh.read())
    paths = [candidates_path] + [preprocessed_path(preprocessed_dir, split, s) for s in (1, 2, 3)]
    for p in paths:
        st = os.stat(p)
        h.update(f"{os.path.basename(p)}:{st.st_size}:{st.st_mtime_ns}".encode())
    h.update(f"{split}|{country}|{drop_fraction:.6f}".encode())
    return h.hexdigest()[:16]


class FeatureCache:
    """Memory-mapped float16 feature columns for every pair of one partition graph."""

    def __init__(self, root: str, split: str, country: str, key: str, columns: List[str]):
        self.base = os.path.join(root, split, country)
        self.dir = os.path.join(self.base, key)
        self.columns = list(columns)
        self._arrays = None

    @property
    def complete(self) -> bool:
        meta = os.path.join(self.dir, "meta.json")
        if not os.path.exists(meta):
            return False
        with open(meta, encoding="utf-8") as fh:
            cols = json.load(fh)["columns"]
        return all(c in cols for c in self.columns)

    def n_rows(self) -> int:
        with open(os.path.join(self.dir, "meta.json"), encoding="utf-8") as fh:
            return int(json.load(fh)["n_rows"])

    def _open(self):
        if self._arrays is None:
            self._arrays = {c: np.load(os.path.join(self.dir, f"{c}.npy"), mmap_mode="r") for c in self.columns}
        return self._arrays

    def load(self, idx: np.ndarray) -> pd.DataFrame:
        arrays = self._open()
        idx = np.asarray(idx)
        return pd.DataFrame({c: np.asarray(arrays[c][idx], dtype=np.float32) for c in self.columns})

    def writer(self, n_rows: int) -> "CacheWriter":
        """Incremental writer (fill rows in any chunks, then publish())."""
        return CacheWriter(self, n_rows)

    def build(self, n_rows: int, compute: Callable[[np.ndarray], pd.DataFrame], chunk: int = 2_000_000,
              log: Callable = print, label: str = "") -> None:
        """Compute all rows chunk by chunk (compute(idx) -> frame with self.columns) and publish atomically."""
        w = self.writer(n_rows)
        t0 = time.time()
        for a in range(0, n_rows, chunk):
            idx = np.arange(a, min(a + chunk, n_rows))
            w.write(idx, compute(idx))
            log(f"[cache]   {label}: {idx[-1] + 1:,}/{n_rows:,} pairs ({time.time() - t0:.0f}s)")
        w.publish()


class CacheWriter:
    """Writes feature columns into a temporary directory, published atomically when complete."""

    def __init__(self, cache: FeatureCache, n_rows: int):
        self.cache = cache
        self.n_rows = n_rows
        self.tmp = cache.dir + ".tmp"
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.makedirs(self.tmp)
        self.arrays = {c: np.lib.format.open_memmap(os.path.join(self.tmp, f"{c}.npy"), mode="w+",
                                                    dtype=np.float16, shape=(n_rows,)) for c in cache.columns}
        self.written = np.zeros(n_rows, bool)

    def write(self, idx: np.ndarray, frame: pd.DataFrame) -> None:
        for c in self.cache.columns:
            self.arrays[c][idx] = frame[c].to_numpy(np.float32).astype(np.float16)
        self.written[idx] = True

    def publish(self) -> None:
        if not self.written.all():
            raise RuntimeError("feature cache incomplete; not publishing")
        for c in self.cache.columns:
            self.arrays[c].flush()
        # Windows cannot rename a directory while memory maps into it are open
        self.arrays.clear()
        import gc

        gc.collect()
        with open(os.path.join(self.tmp, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump({"n_rows": int(self.n_rows), "columns": self.cache.columns, "format": CACHE_FORMAT}, fh)
        base, final = self.cache.base, self.cache.dir
        # Drop older fingerprints of this split/country, then publish
        for name in os.listdir(base):
            path = os.path.join(base, name)
            if path not in (self.tmp, final) and os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
        shutil.rmtree(final, ignore_errors=True)
        os.replace(self.tmp, final)
