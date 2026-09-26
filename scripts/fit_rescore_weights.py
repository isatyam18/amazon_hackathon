"""
Fit the Stage 2 rescoring weights (BlockingConfig.rescore_weights) on labelled queries.

Runs the retrieval passes for the validation queries against the FULL train
S2/S3, computes exact similarities for every retrieved pair, fits a logistic
regression on half of the queries and reports recall per budget on the other
half (fitted vs current weights). Writes a config JSON usable with
`run_blocking.py --config`; the fitted weights are also printed so they can be
copied into src/blocking/config.py as the new defaults.

    python scripts/fit_rescore_weights.py
    python scripts/run_blocking.py --split val --config dataset/candidates/fitted_blocking_config.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.blocking import BlockingConfig, BlockingPipeline, load_source
from src.blocking.data import ground_truth_pairs, load_ground_truth
from src.blocking.tuning import (
    collect_labelled_union,
    fit_rescore_weights,
    recall_at_budgets,
    score_with_weights,
    split_queries,
)


def main():
    parser = argparse.ArgumentParser(description="Fit Stage 2 rescoring weights")
    parser.add_argument("--preprocessed-dir", default="dataset/preprocessed")
    parser.add_argument("--gt", default="dataset/val_split/val_ground_truth.tsv")
    parser.add_argument("--config", default=None, help="base config JSON (default: BlockingConfig())")
    parser.add_argument("--views", nargs="+", default=None,
                        help="similarity views to weight (default: the config's rescore views)")
    parser.add_argument("--budgets", nargs="+", type=int, default=[5, 10, 15, 20, 25, 30])
    parser.add_argument("--holdout-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="dataset/candidates/fitted_blocking_config.json")
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()

    cfg = BlockingConfig.load(args.config)
    if args.n_threads:
        cfg.n_threads = args.n_threads
    if args.n_jobs:
        cfg.n_jobs = args.n_jobs
    current = dict(cfg.rescore_weights)
    views = args.views or list(current)
    # Keep normalised target rows for every view we want to weight
    cfg.rescore_weights = {v: current.get(v, 1.0) for v in dict.fromkeys(views + list(current))}

    gt = load_ground_truth(args.gt)
    s1 = load_source(args.preprocessed_dir, "train", 1)
    queries = s1[s1["entity_id"].isin(set(gt["source1_entity_id"]))].reset_index(drop=True)
    targets = {"S2": load_source(args.preprocessed_dir, "train", 2),
               "S3": load_source(args.preprocessed_dir, "train", 3)}

    df = collect_labelled_union(BlockingPipeline(cfg), queries, targets, gt, list(cfg.rescore_weights))
    truth = ground_truth_pairs(gt)
    print(f"\nRetrieved {len(df):,} pairs; retrieval recall "
          f"{df['label'].sum() / max(len(truth), 1):.4f} of {len(truth):,} true pairs")

    fit_ids = set(split_queries(gt["source1_entity_id"], args.holdout_frac, args.seed))
    is_fit = df["source1_entity_id"].isin(fit_ids).to_numpy()
    held = df[~is_fit]
    n_true_held = int((~truth["source1_entity_id"].isin(fit_ids)).sum())

    fitted = fit_rescore_weights(df[is_fit], views)
    print(f"\nRecall per source budget on {len(set(held['source1_entity_id'])):,} held-out queries:")
    print("  current:", {b: round(r, 4) for b, r in recall_at_budgets(
        held, score_with_weights(held, current), n_true_held, args.budgets).items()}, current)
    print("  fitted :", {b: round(r, 4) for b, r in recall_at_budgets(
        held, score_with_weights(held, fitted), n_true_held, args.budgets).items()}, fitted)

    final = fit_rescore_weights(df, views)  # refit on all queries for the saved config
    cfg.rescore_weights = final
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cfg.save(args.out)
    print(f"\nWeights refitted on all queries: {json.dumps(final)}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
