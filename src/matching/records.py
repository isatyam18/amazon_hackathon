"""
Record store for Stage 3: every S1/S2/S3 record of a split, addressed by integer row.

Pair features are computed for tens of millions of (S1, candidate) pairs, so
record-level work (normalisation, tokenisation, hashing) is done once per record
here, and pairs only index into these arrays. The normalisation matches the
blocking stage (src/blocking/normalize.py): legal/filler tokens removed, leet
digits fixed, address numbers without leading zeros, phonetic skeleton.
"""

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import scipy.sparse as sp

from src.blocking.data import load_source
from src.blocking.features import featurize
from src.blocking.normalize import address_tokens, name_tokens, phonetic_name

TOKEN_VIEWS = ("name_word", "addr_word", "addr_num")
STRING_FIELDS = ("name", "name_compact", "name_phon", "addr", "house")


class StringColumn:
    """Arrow-backed string column (~1 byte/char instead of a Python object per string).

    Indexing with an integer array returns a NumPy object array of Python strings,
    materialised only for the rows asked for (what rapidfuzz's cpdist consumes).
    """

    def __init__(self, arr: pa.Array):
        self.arr = arr

    def __len__(self) -> int:
        return len(self.arr)

    def __getitem__(self, idx):
        if np.isscalar(idx):
            return self.arr[int(idx)].as_py()
        return self.arr.take(pa.array(np.asarray(idx, dtype=np.int64))).to_numpy(zero_copy_only=False)


def _derive_chunk(names: Sequence[str], addresses: Sequence[str]) -> Dict[str, list]:
    out = {"name": [], "name_compact": [], "name_phon": [], "addr": [], "house": []}
    for n, a in zip(names, addresses):
        nt = name_tokens(n)
        at = address_tokens(a)
        out["name"].append(" ".join(nt))
        out["name_compact"].append("".join(nt))
        out["name_phon"].append(phonetic_name(nt))
        out["addr"].append(" ".join(at))
        out["house"].append(next((t for t in at if t.isdigit()), ""))
    return out


class RecordStore:
    """All records of one split with derived matching fields, indexed 0..n-1."""

    def __init__(self, frame: pd.DataFrame, n_jobs: int = 8, chunk_size: int = 200_000):
        frame = frame.reset_index(drop=True)
        self.ids = frame["entity_id"].to_numpy(object)
        self.index = pd.Index(self.ids)
        self.country = frame["country"].to_numpy(object)
        # Source number from the id prefix ('S1-' -> 1); the prefix defines the source
        self.source = frame["entity_id"].str.slice(1, 2).astype(np.int8).to_numpy()

        names = frame["clean_name"].to_numpy(object).tolist()
        addrs = frame["clean_address"].to_numpy(object).tolist()
        bounds = [(i, min(i + chunk_size, len(names))) for i in range(0, len(names), chunk_size)]
        if n_jobs > 1 and len(bounds) > 1:
            from joblib import Parallel, delayed

            parts = Parallel(n_jobs=n_jobs)(delayed(_derive_chunk)(names[a:b], addrs[a:b]) for a, b in bounds)
        else:
            parts = [_derive_chunk(names[a:b], addrs[a:b]) for a, b in bounds]
        for key in STRING_FIELDS:
            chunks = [pa.array(p.pop(key), type=pa.large_string()) for p in parts]
            setattr(self, key, StringColumn(pa.chunked_array(chunks).combine_chunks()))
        del parts
        import pyarrow.compute as pc

        self.name_len = pc.utf8_length(self.name_compact.arr).to_numpy(zero_copy_only=False).astype(np.float32)

        # Binary token matrices: pair intersections via row-wise dot products
        self.tokens: Dict[str, sp.csr_matrix] = featurize(
            names, addrs, TOKEN_VIEWS, hash_bits=22, n_jobs=n_jobs, chunk_size=chunk_size
        )
        self.token_counts = {v: np.diff(m.indptr).astype(np.float32) for v, m in self.tokens.items()}
        import pyarrow.compute as pc

        first = pc.utf8_split_whitespace(self.name.arr, max_splits=1)
        first = pc.list_element(pc.if_else(pc.equal(self.name.arr, ""), pa.scalar([""], pa.list_(pa.large_string())), first), 0)
        first_np = first.to_numpy(zero_copy_only=False)
        self.first_token_hash = pd.util.hash_array(first_np.astype(object)).astype(np.int64)
        self.first_token_hash[first_np == ""] = -1

    def __len__(self) -> int:
        return len(self.ids)

    def rows(self, entity_ids: Iterable[str]) -> np.ndarray:
        """Integer rows of entity ids (-1 for unknown ids)."""
        return self.index.get_indexer(pd.Index(entity_ids)).astype(np.int64)

    @property
    def query_rows(self) -> np.ndarray:
        """Rows of Source 1 records, in file order (the queries every submission must cover)."""
        return np.flatnonzero(self.source == 1)

    @classmethod
    def load(
        cls,
        preprocessed_dir: str,
        split: str,
        n_jobs: int = 8,
        keep_target_ids: Optional[set] = None,
    ) -> "RecordStore":
        """All S1 records of the split plus S2/S3 records (optionally only `keep_target_ids`)."""
        frames = [load_source(preprocessed_dir, split, 1)]
        for s in (2, 3):
            df = load_source(preprocessed_dir, split, s)
            if keep_target_ids is not None:
                df = df[df["entity_id"].isin(keep_target_ids)]
            frames.append(df)
        return cls(pd.concat(frames, ignore_index=True), n_jobs=n_jobs)
