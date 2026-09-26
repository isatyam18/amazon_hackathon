"""
Stage 2: candidate generation (blocking) CLI.

Reads Stage 1 output from dataset/preprocessed/ and produces candidate pairs.

    # Test set -> output/candidate_pairs.tsv (+ scored parquet for Stage 3)
    python scripts/run_blocking.py --split test

    # Validation: val-split S1 queries vs the FULL train S2/S3, scored against ground truth
    python scripts/run_blocking.py --split val

    # Training candidates for Stage 3 (random train S1 sample, val entities excluded)
    python scripts/run_blocking.py --split train --sample-s1 200000

Outputs (val/train go to --out-dir, default dataset/candidates/):
    {split}_candidate_pairs.tsv       submission format (one row per S1 entity)
    {split}_candidate_pairs.parquet   one row per pair with scores / similarities
    {split}_blocking_report.json      config, timings and (val/train) recall metrics
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.blocking import BlockingConfig, BlockingPipeline, evaluate_candidates, format_report, load_source
from src.blocking.data import load_ground_truth, read_id_set


def load_queries(args):
    """Source 1 query frame and (for val/train) the matching ground-truth rows."""
    if args.split == "test":
        return load_source(args.preprocessed_dir, "test", 1), None

    s1 = load_source(args.preprocessed_dir, "train", 1)
    if args.split == "val":
        if not os.path.exists(args.val_gt):
            sys.exit(f"{args.val_gt} not found. Create it first:\n"
                     "  python scripts/create_val_split.py --num-s1 10000 --num-distractors 0")
        gt = load_ground_truth(args.val_gt)
        queries = s1[s1["entity_id"].isin(set(gt["source1_entity_id"]))].reset_index(drop=True)
        return queries, gt

    # split == "train": optional random sample, never overlapping the validation queries
    gt = load_ground_truth(args.train_gt)
    if os.path.exists(args.val_gt):
        s1 = s1[~s1["entity_id"].isin(read_id_set(args.val_gt))]
    if args.sample_s1:
        s1 = s1.sample(n=min(args.sample_s1, len(s1)), random_state=args.seed)
    queries = s1.reset_index(drop=True)
    gt = gt[gt["source1_entity_id"].isin(set(queries["entity_id"]))].reset_index(drop=True)
    return queries, gt


def main():
    parser = argparse.ArgumentParser(description="Stage 2: blocking / candidate generation")
    parser.add_argument("--split", choices=["test", "val", "train"], default="test")
    parser.add_argument("--preprocessed-dir", default="dataset/preprocessed")
    parser.add_argument("--val-gt", default="dataset/val_split/val_ground_truth.tsv")
    parser.add_argument("--train-gt", default="student_resource/dataset/train/train_ground_truth.tsv")
    parser.add_argument("--sample-s1", type=int, default=0, help="train only: number of S1 queries to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default=None, help="JSON file overriding BlockingConfig defaults")
    parser.add_argument("--out-dir", default="dataset/candidates", help="val/train outputs and all parquet/reports")
    parser.add_argument("--submission-dir", default="output", help="test only: where candidate_pairs.tsv goes")
    parser.add_argument("--max-candidates", type=int, default=None, help="override max_candidates_per_source")
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--dense", action="store_true", help="enable the optional dense (MiniLM + FAISS) pass")
    parser.add_argument("--no-parquet", action="store_true", help="skip the scored parquet output")
    args = parser.parse_args()

    cfg = BlockingConfig.load(args.config)
    if args.max_candidates:
        cfg.max_candidates_per_source = args.max_candidates
    if args.n_threads:
        cfg.n_threads = args.n_threads
    if args.n_jobs:
        cfg.n_jobs = args.n_jobs
    if args.dense:
        cfg.dense.enabled = True

    t0 = time.time()
    queries, gt = load_queries(args)
    target_split = "test" if args.split == "test" else "train"
    targets = {
        "S2": load_source(args.preprocessed_dir, target_split, 2),
        "S3": load_source(args.preprocessed_dir, target_split, 3),
    }
    n_targets = sum(len(t) for t in targets.values())
    print(f"Loaded {len(queries):,} S1 queries and {n_targets:,} S2/S3 targets in {time.time() - t0:.0f}s")
    print("Queries by country:", queries["country"].value_counts().to_dict())

    pipeline = BlockingPipeline(cfg)
    candidates = pipeline.run(queries, targets)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.split == "test":
        tsv_path = os.path.join(args.submission_dir, "candidate_pairs.tsv")
    else:
        tsv_path = os.path.join(args.out_dir, f"{args.split}_candidate_pairs.tsv")
    candidates.write_tsv(tsv_path)
    print(f"Wrote {tsv_path}")
    if not args.no_parquet:
        pq_path = os.path.join(args.out_dir, f"{args.split}_candidate_pairs.parquet")
        try:
            candidates.write_parquet(pq_path)
            print(f"Wrote {pq_path}")
        except OSError as exc:
            # The scored parquet is optional (Stage 3 convenience); never lose the run over it
            print(f"WARNING: could not write {pq_path} ({exc}); candidate TSV and report are unaffected")
            if os.path.isfile(pq_path):
                os.remove(pq_path)

    per_q = candidates.candidates_per_query()
    report = {
        "split": args.split,
        "queries": int(len(queries)),
        "targets": int(n_targets),
        "candidate_pairs": candidates.n_pairs,
        "candidates_per_query": {"mean": float(per_q.mean()), "p95": float(np.percentile(per_q, 95)),
                                 "max": int(per_q.max()), "zero": int((per_q == 0).sum())},
        "reduction_ratio": float(1.0 - candidates.n_pairs / (len(queries) * float(n_targets))),
        "seconds": time.time() - t0,
        "step_seconds": pd.DataFrame(pipeline.stats).groupby("step")["seconds"].sum().round(1).to_dict()
        if pipeline.stats else {},
        "config": cfg.to_dict(),
    }
    if gt is not None:
        metrics = evaluate_candidates(
            candidates.to_frame(), gt, n_targets=n_targets,
            query_country=pd.Series(queries["country"].to_numpy(), index=queries["entity_id"]),
        )
        report["metrics"] = metrics
        print("\n" + format_report(metrics))

    report_path = os.path.join(args.out_dir, f"{args.split}_blocking_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {report_path} ({report['seconds'] / 60:.1f} min total)")


if __name__ == "__main__":
    main()
