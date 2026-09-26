"""
High-throughput full dataset preprocessor.

Normalizes both Train and Test splits:
- train_source1.tsv, train_source2.tsv, train_source3.tsv
- test_source1.tsv, test_source2.tsv, test_source3.tsv

Outputs to dataset/preprocessed/ in TSV (default) or Parquet format.
Processes data in memory-safe streaming chunks (e.g. 100k rows) with progress bars,
optionally fanning chunks out to worker processes (--n-jobs).
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import preprocess_dataframe


def preprocess_file(
    input_path: str,
    output_path: str,
    chunksize: int = 100000,
    n_jobs: int = 1,
    fmt: str = "tsv",
):
    """Preprocess a single large TSV file in streaming chunks."""
    if not os.path.exists(input_path):
        print(f"Warning: {input_path} not found. Skipping.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Estimate total rows for progress bar
    print(f"\nProcessing {os.path.basename(input_path)} -> {os.path.basename(output_path)} ...")
    t0 = time.time()

    first_chunk = True
    total_processed = 0
    parquet_writer = None

    with pd.read_csv(input_path, sep="\t", chunksize=chunksize, dtype=str) as reader:
        pool = Pool(n_jobs) if n_jobs > 1 else None
        # imap preserves chunk order, so output row order matches the input file
        processed_iter = pool.imap(preprocess_dataframe, reader) if pool else map(preprocess_dataframe, reader)
        try:
            for processed_chunk in tqdm(processed_iter, desc=os.path.basename(input_path), unit="chunk"):
                if fmt == "parquet":
                    import pyarrow as pa
                    import pyarrow.parquet as pq

                    table = pa.Table.from_pandas(processed_chunk.astype(str), preserve_index=False)
                    if parquet_writer is None:
                        parquet_writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                    parquet_writer.write_table(table)
                else:
                    # Write mode: write header on first chunk, append thereafter
                    processed_chunk.to_csv(
                        output_path,
                        sep="\t",
                        index=False,
                        mode="w" if first_chunk else "a",
                        header=first_chunk,
                    )
                first_chunk = False
                total_processed += len(processed_chunk)
        finally:
            if pool:
                pool.close()
                pool.join()
            if parquet_writer is not None:
                parquet_writer.close()

    elapsed = time.time() - t0
    rate = total_processed / elapsed if elapsed > 0 else 0
    print(f"Completed {total_processed:,} records in {elapsed:.1f}s ({rate:,.0f} records/s).")


def preprocess_complete_dataset(
    input_base_dir: str = "student_resource/dataset",
    output_base_dir: str = "dataset/preprocessed",
    split: str = "all",
    chunksize: int = 100000,
    n_jobs: int = 1,
    fmt: str = "tsv",
    skip_existing: bool = False,
):
    """Run preprocessing across requested dataset splits."""
    tasks = []

    if split in ("all", "train"):
        tasks.extend([
            (
                os.path.join(input_base_dir, "train", "train_source1.tsv"),
                os.path.join(output_base_dir, "train_source1.tsv"),
            ),
            (
                os.path.join(input_base_dir, "train", "train_source2.tsv"),
                os.path.join(output_base_dir, "train_source2.tsv"),
            ),
            (
                os.path.join(input_base_dir, "train", "train_source3.tsv"),
                os.path.join(output_base_dir, "train_source3.tsv"),
            ),
        ])

    if split in ("all", "test"):
        tasks.extend([
            (
                os.path.join(input_base_dir, "test", "test_source1.tsv"),
                os.path.join(output_base_dir, "test_source1.tsv"),
            ),
            (
                os.path.join(input_base_dir, "test", "test_source2.tsv"),
                os.path.join(output_base_dir, "test_source2.tsv"),
            ),
            (
                os.path.join(input_base_dir, "test", "test_source3.tsv"),
                os.path.join(output_base_dir, "test_source3.tsv"),
            ),
        ])

    print(f"=== Starting Preprocessing (Split: {split}) ===")
    print(f"Input base : {input_base_dir}")
    print(f"Output base: {output_base_dir}")
    print(f"Total files: {len(tasks)}")
    print(f"Format     : {fmt} | workers: {n_jobs}")

    overall_t0 = time.time()
    for in_path, out_path in tasks:
        if fmt == "parquet":
            out_path = os.path.splitext(out_path)[0] + ".parquet"
        if skip_existing and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"\nSkipping {out_path} (already exists).")
            continue
        preprocess_file(in_path, out_path, chunksize=chunksize, n_jobs=n_jobs, fmt=fmt)

    total_elapsed = time.time() - overall_t0
    print(f"\nAll tasks finished in {total_elapsed/60:.1f} minutes!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess complete competition dataset")
    parser.add_argument(
        "--input-dir",
        default="student_resource/dataset",
        help="Path to dataset directory containing train/ and test/",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/preprocessed",
        help="Output directory for preprocessed files",
    )
    parser.add_argument(
        "--split",
        choices=["all", "train", "test"],
        default="all",
        help="Which split to preprocess: all, train, or test",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100000,
        help="Chunk size for streaming processing",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Worker processes for chunk preprocessing (1 = sequential)",
    )
    parser.add_argument(
        "--format",
        choices=["tsv", "parquet"],
        default="tsv",
        help="Output format (parquet is ~4x smaller on disk; blocking reads either)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files whose output already exists (resume an interrupted run)",
    )
    args = parser.parse_args()

    preprocess_complete_dataset(
        input_base_dir=args.input_dir,
        output_base_dir=args.output_dir,
        split=args.split,
        chunksize=args.chunksize,
        n_jobs=args.n_jobs,
        fmt=args.format,
        skip_existing=args.skip_existing,
    )
