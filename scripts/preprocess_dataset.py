"""
High-throughput full dataset preprocessor.

Normalizes both Train and Test splits:
- train_source1.tsv, train_source2.tsv, train_source3.tsv
- test_source1.tsv, test_source2.tsv, test_source3.tsv

Outputs to dataset/preprocessed/ in TSV format.
Processes data in memory-safe streaming chunks (e.g. 100k rows) with progress bars.
"""

import argparse
import os
import sys
import time
import pandas as pd
from tqdm import tqdm

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import preprocess_dataframe


def preprocess_file(input_path: str, output_path: str, chunksize: int = 100000):
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

    with pd.read_csv(input_path, sep="\t", chunksize=chunksize, dtype=str) as reader:
        for chunk in tqdm(reader, desc=os.path.basename(input_path), unit="chunk"):
            processed_chunk = preprocess_dataframe(chunk)

            # Write mode: write header on first chunk, append thereafter
            processed_chunk.to_csv(
                output_path,
                sep="\t",
                index=False,
                mode="w" if first_chunk else "a",
                header=first_chunk,
            )
            first_chunk = False
            total_processed += len(chunk)

    elapsed = time.time() - t0
    rate = total_processed / elapsed if elapsed > 0 else 0
    print(f"Completed {total_processed:,} records in {elapsed:.1f}s ({rate:,.0f} records/s).")


def preprocess_complete_dataset(
    input_base_dir: str = "student_resource/dataset",
    output_base_dir: str = "dataset/preprocessed",
    split: str = "all",
    chunksize: int = 100000,
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

    overall_t0 = time.time()
    for in_path, out_path in tasks:
        preprocess_file(in_path, out_path, chunksize=chunksize)

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
        help="Output directory for preprocessed TSVs",
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
    args = parser.parse_args()

    preprocess_complete_dataset(
        input_base_dir=args.input_dir,
        output_base_dir=args.output_dir,
        split=args.split,
        chunksize=args.chunksize,
    )
