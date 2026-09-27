"""
Container for blocking output and its writers.

Pairs are held as integer indices (query row, source code, target row) plus
scores, and entity-id strings are only materialised per query chunk while
writing. This keeps ~85M test pairs from turning into gigabytes of Python strings.

Outputs:
    candidate_pairs.tsv      submission format: one row per S1 entity (empty
                             list when blocking found nothing), candidate ids
                             ordered by candidate score, S2-/S3- ids only, no duplicates.
    *_candidate_pairs.parquet one row per (S1, candidate) pair with source,
                             candidate score, per-source rank, every retrieval
                             pass's cosine (score_*) and the exact rescoring
                             similarities (sim_*): the input to Stage 3.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List

import numpy as np
import pandas as pd

CANDIDATE_TSV_HEADER = ("source1_entity_id", "candidate_entity_ids")


@dataclass
class CandidateSet:
    query_ids: np.ndarray                 # all S1 entity ids, in output row order
    query_country: np.ndarray             # country label per query row
    source_names: List[str]               # source code -> name, e.g. ["S2", "S3"]
    target_ids: Dict[str, np.ndarray]     # source name -> entity ids by target row
    feature_names: List[str]              # columns of `features`, e.g. score_name, sim_name_char
    q: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    src: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int8))
    t: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    score: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    source_rank: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int16))
    features: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))

    @property
    def n_pairs(self) -> int:
        return int(len(self.q))

    def candidates_per_query(self) -> np.ndarray:
        return np.bincount(self.q, minlength=len(self.query_ids))

    def _sorted_order(self) -> np.ndarray:
        """Pairs ordered by query row, then descending candidate score."""
        return np.lexsort((-self.score, self.q))

    def _candidate_ids(self, sel: np.ndarray) -> np.ndarray:
        out = np.empty(len(sel), dtype=object)
        src = self.src[sel]
        t = self.t[sel]
        for code, name in enumerate(self.source_names):
            m = src == code
            if m.any():
                out[m] = self.target_ids[name][t[m]]
        return out

    def iter_frames(self, chunk_queries: int = 200_000) -> Iterator[pd.DataFrame]:
        """Yield pair DataFrames with string ids, one chunk of query rows at a time."""
        order = self._sorted_order()
        q_sorted = self.q[order]
        n = len(self.query_ids)
        for start in range(0, n, chunk_queries):
            end = min(start + chunk_queries, n)
            lo, hi = np.searchsorted(q_sorted, [start, end])
            sel = order[lo:hi]
            qs = self.q[sel]
            frame = {
                "source1_entity_id": self.query_ids[qs],
                "candidate_entity_id": self._candidate_ids(sel),
                "source": np.array(self.source_names, dtype=object)[self.src[sel]],
                "country": self.query_country[qs],
                "candidate_score": self.score[sel],
                "source_rank": self.source_rank[sel],
            }
            for fi, name in enumerate(self.feature_names):
                # float16 keeps ~3 significant digits, ample for similarity features, at half the size
                frame[name] = self.features[sel, fi].astype(np.float16)
            yield pd.DataFrame(frame)

    def write_tsv(self, path: str, chunk_queries: int = 200_000) -> None:
        """Write candidate_pairs.tsv in the submission format (every S1 entity gets a row)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        order = self._sorted_order()
        q_sorted = self.q[order]
        n = len(self.query_ids)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\t".join(CANDIDATE_TSV_HEADER) + "\n")
            for start in range(0, n, chunk_queries):
                end = min(start + chunk_queries, n)
                lo, hi = np.searchsorted(q_sorted, [start, end])
                ids = self._candidate_ids(order[lo:hi])
                cuts = np.searchsorted(q_sorted[lo:hi], np.arange(start, end + 1))
                lines = [
                    f"{self.query_ids[qi]}\t{','.join(ids[cuts[i]:cuts[i + 1]])}\n"
                    for i, qi in enumerate(range(start, end))
                ]
                f.writelines(lines)

    def write_parquet(self, path: str, chunk_queries: int = 200_000) -> None:
        """Write the scored pair table (Stage 3 input) as zstd-compressed parquet."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        writer = None
        try:
            for frame in self.iter_frames(chunk_queries):
                if frame.empty:
                    continue
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()

    def to_frame(self, query_rows: np.ndarray = None) -> pd.DataFrame:
        """Pairs as one DataFrame with string ids, optionally only for `query_rows`.

        Materialising strings is memory-heavy, so for large runs pass a subset.
        """
        keep = None if query_rows is None else np.isin(self.q, query_rows)
        sub = self if keep is None else CandidateSet(
            query_ids=self.query_ids, query_country=self.query_country, source_names=self.source_names,
            target_ids=self.target_ids, feature_names=self.feature_names, q=self.q[keep], src=self.src[keep],
            t=self.t[keep], score=self.score[keep], source_rank=self.source_rank[keep],
            features=self.features[keep],
        )
        frames = [f for f in sub.iter_frames(chunk_queries=max(len(self.query_ids), 1)) if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
