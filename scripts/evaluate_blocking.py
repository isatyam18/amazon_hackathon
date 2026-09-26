"""
Evaluate a candidate set against ground truth (blocking recall ceiling).

Works with either output format of scripts/run_blocking.py:

    python scripts/evaluate_blocking.py \
        --candidates dataset/candidates/val_candidate_pairs.parquet \
        --gt dataset/val_split/val_ground_truth.tsv

    python scripts/evaluate_blocking.py \
        --candidates dataset/candidates/val_candidate_pairs.tsv \
        --gt dataset/val_split/val_ground_truth.tsv --max-per-source 10

--max-per-source re-cuts a parquet candidate set to a smaller budget (by
source_rank), which shows the recall/size trade-off without re-running blocking.
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.blocking import evaluate_candidates, format_report
from src.blocking.data import load_ground_truth, read_candidate_tsv


def main():
    parser = argparse.ArgumentParser(description="Evaluate blocking candidates against ground truth")
    parser.add_argument("--candidates", required=True, help="candidate_pairs .tsv or scored .parquet")
    parser.add_argument("--gt", default="dataset/val_split/val_ground_truth.tsv")
    parser.add_argument("--n-targets", type=int, default=None, help="|S2|+|S3| for the reduction ratio")
    parser.add_argument("--max-per-source", type=int, default=None, help="parquet only: re-cut to this budget")
    parser.add_argument("--json", default=None, help="optional path to save the metrics as JSON")
    args = parser.parse_args()

    gt = load_ground_truth(args.gt)
    if args.candidates.endswith(".parquet"):
        pairs = pd.read_parquet(args.candidates)
        if args.max_per_source is not None:
            pairs = pairs[pairs["source_rank"] < args.max_per_source]
        query_country = pairs.drop_duplicates("source1_entity_id").set_index("source1_entity_id")["country"]
    else:
        if args.max_per_source is not None:
            sys.exit("--max-per-source needs the parquet output (it uses source_rank)")
        pairs = read_candidate_tsv(args.candidates)
        query_country = None

    report = evaluate_candidates(pairs, gt, n_targets=args.n_targets, query_country=query_country)
    print(format_report(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
