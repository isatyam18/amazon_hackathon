"""
Stages 3-5 CLI: pair features, LightGBM matcher, decision rule, submission files.

Modes:
  train  Full-train universe (scripts/run_blocking.py --split train --all-train):
         2-fold out-of-fold training, then calibration + decision-rule tuning on
         fold 0 and an honest leaderboard-metric report on fold 1. Saves models/,
         models/oof/ (out-of-fold arrays) and models/matching_config.json.
         python scripts/run_matching.py --mode train

  tune   Re-run calibration, decision-rule tuning and the report from the saved OOF
         arrays (no retraining; seconds to minutes).
         python scripts/run_matching.py --mode tune

  stack  Train the second-stage model on the saved OOF probabilities (probability-context
         features: best competing S1 for the same record, other strong candidates of the
         same S1), re-tune and report with the same fold protocol. It is adopted
         (models/stack_fold*.txt, config "stack_models") only if it beats the first stage
         on fold 0.
         python scripts/run_matching.py --mode stack

  test   Scores the test candidates with both fold models, applies the tuned rule and
         writes output/matching_results.tsv and output/candidate_pairs.tsv (exactly the
         pairs the model scored), then runs the official validator.
         python scripts/run_matching.py --mode test

  all    train, then test (default).
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.blocking.data import load_ground_truth, read_id_set
from src.matching.decision import Calibrator, DecisionRule
from src.matching.model import MatcherModel
from src.matching.workflow import (
    PartitionResult,
    Universe,
    default_rule_grid,
    evaluate,
    orphan_drop_fraction,
    predict_and_write,
    query_folds,
    select_all,
    train_oof,
    tune_decision,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def fmt(report: dict) -> str:
    lines = []
    for k, v in report.items():
        if k == "macro_f05_by_country":
            lines.append(f"    {k:<26}: " + ", ".join(f"{c}={s:.4f}" for c, s in v.items()))
        elif isinstance(v, float):
            lines.append(f"    {k:<26}: {v:.5f}")
        else:
            lines.append(f"    {k:<26}: {v:,}")
    return "\n".join(lines)


def tune_and_report(args, universe: Universe, folds: np.ndarray, results) -> dict:
    """Calibration + rule on fold 0, honest report on fold 1."""
    tune_mask, report_mask = (folds == 0) & universe.active, (folds == 1) & universe.active
    labels = [r.labels() for r in results]
    fold0 = [tune_mask[r.q] for r in results]
    calibrator = Calibrator().fit(np.concatenate([r.prob[m] for r, m in zip(results, fold0)]),
                                  np.concatenate([l[m] for l, m in zip(labels, fold0)]))
    probs = [calibrator.transform(r.prob) for r in results]

    missed = 0
    for r, l in zip(results, labels):
        n_true = np.bincount(r.truth_q, minlength=len(universe))
        in_cands = np.bincount(r.q[l], minlength=len(universe))
        rows = r.query_rows[tune_mask[r.query_rows]]
        missed += int((n_true - in_cands)[rows].sum())
    missing_mass = missed / max(int(tune_mask.sum()), 1)
    log(f"\n[matching] true matches per S1 missed by blocking (fold 0): {missing_mass:.4f}")
    log("[matching] decision-rule grid on fold 0:")
    rule, grid = tune_decision(results, probs, universe, tune_mask, default_rule_grid(missing_mass), log=log)
    log(f"[matching] selected rule: {rule.to_dict()}")

    n = len(universe)
    selected = select_all(rule, results, probs)
    report = {"fold1": evaluate(results, selected, n, report_mask, universe.s1_country),
              "fold0_tuning": evaluate(results, selected, n, tune_mask, universe.s1_country)}
    baseline = DecisionRule("threshold", threshold=0.5, owner_normalize=False, exclusive=False)
    report["fold1_baseline_threshold_0.5"] = evaluate(
        results, select_all(baseline, results, probs), n, report_mask, universe.s1_country)
    if os.path.exists(args.val_gt):
        val_mask = np.isin(universe.s1_ids, list(read_id_set(args.val_gt))) & report_mask
        report["val_split_fold1"] = evaluate(results, selected, n, val_mask, universe.s1_country)
    for name in ("fold1", "fold1_baseline_threshold_0.5", "fold0_tuning", "val_split_fold1"):
        if name in report:
            log(f"\n[{name}]\n{fmt(report[name])}")
    return {"calibrator": calibrator.to_dict(), "decision_rule": rule.to_dict(), "rule_grid": grid,
            "missing_mass": missing_mass, "report": report}


def drop_fraction_for(args) -> float:
    if args.no_orphan_sim:
        return 0.0
    return orphan_drop_fraction(args.preprocessed_dir)


def run_train(args) -> None:
    log("=" * 70 + "\nSTAGES 3-4: OUT-OF-FOLD TRAINING, DECISION TUNING, EVALUATION\n" + "=" * 70)
    drop = drop_fraction_for(args)
    models, universe, folds, results = train_oof(
        args.train_candidates, args.preprocessed_dir, load_ground_truth(args.train_gt), "train",
        n_train_queries=args.train_queries, seed=args.seed, n_jobs=args.n_jobs,
        results_dir=os.path.join(args.model_dir, "oof"), log=log,
        drop_fraction=drop, num_boost_round=args.num_boost_round,
        cache_root=None if args.no_feature_cache else args.feature_cache,
    )
    importances = models[0].get_feature_importances()
    log("\nTop features (gain, fold-0 model):\n" + importances.head(25).to_string(index=False))

    os.makedirs(args.model_dir, exist_ok=True)
    for i, m in enumerate(models):
        m.save(os.path.join(args.model_dir, f"matcher_fold{i}.txt"))
    config = {
        "models": [f"matcher_fold{i}.txt" for i in range(len(models))],
        "feature_names": models[0].feature_names,
        "feature_importance": importances.to_dict(orient="records"),
        "universe_drop_fraction": drop,
        # features the models were trained on: float16 from the feature cache, float32 when computed live
        "feature_precision": "float32" if args.no_feature_cache else "float16",
    }
    config.update(tune_and_report(args, universe, folds, results))
    with open(os.path.join(args.model_dir, "matching_config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    log(f"\nSaved models, OOF arrays and {os.path.join(args.model_dir, 'matching_config.json')}")


def run_tune(args) -> None:
    log("=" * 70 + "\nSTAGE 4: DECISION TUNING FROM SAVED OOF ARRAYS\n" + "=" * 70)
    config_path = os.path.join(args.model_dir, "matching_config.json")
    config = json.load(open(config_path, encoding="utf-8")) if os.path.exists(config_path) else {}
    # Same universe the OOF arrays were produced on (test-like orphan simulation, if used)
    universe = Universe.load(args.preprocessed_dir, "train", drop_fraction=config.get("universe_drop_fraction", 0.0))
    folds = query_folds(universe.s1_ids)
    results = [PartitionResult.load(p) for p in sorted(glob.glob(os.path.join(args.model_dir, "oof", "oof_*.npz")))]
    if not results:
        sys.exit("No OOF arrays in models/oof/. Run --mode train first.")
    config.update(tune_and_report(args, universe, folds, results))
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    log(f"\nUpdated {config_path}")


def run_stack(args) -> None:
    from src.matching.stacking import train_stack

    log("=" * 70 + "\nSTAGE 4b: SECOND-STAGE (STACKED) MODEL ON OOF PROBABILITIES\n" + "=" * 70)
    config_path = os.path.join(args.model_dir, "matching_config.json")
    config = json.load(open(config_path, encoding="utf-8"))
    universe = Universe.load(args.preprocessed_dir, "train", drop_fraction=config.get("universe_drop_fraction", 0.0))
    folds = query_folds(universe.s1_ids)
    results = [PartitionResult.load(p) for p in sorted(glob.glob(os.path.join(args.model_dir, "oof", "oof_*.npz")))]
    first_stage = config["report"]["fold0_tuning"]["macro_f05"]

    extras = None
    if args.stack_support:
        from src.matching.stacking import support_features
        from src.matching.workflow import CandidateGraph, Partition, release_memory

        extras = []
        for r in results:
            graph = CandidateGraph.load(args.train_candidates, universe, r.country)
            if len(graph) != len(r.q) or not np.array_equal(graph.q, r.q):
                sys.exit(f"{r.country}: candidate graph does not match the saved OOF arrays; rerun --mode train")
            part = Partition.build(graph, args.preprocessed_dir, "train", args.n_jobs)
            store = part.store
            extras.append(support_features(r.q, r.prob, part.t_local, store))
            log(f"[stacking] {r.country}: support features for {len(r.q):,} pairs")
            del graph, part, store
            release_memory()

    models, stacked = train_stack(results, folds, log=log, extras=extras)
    for r, p in zip(results, stacked):
        r.prob = p
    outcome = tune_and_report(args, universe, folds, results)
    second_stage = outcome["report"]["fold0_tuning"]["macro_f05"]
    log(f"\n[stacking] fold-0 macro F0.5: first stage {first_stage:.5f} -> second stage {second_stage:.5f}")
    if second_stage > first_stage:
        for i, m in enumerate(models):
            m.save(os.path.join(args.model_dir, f"stack_fold{i}.txt"))
        config["previous_report"] = config["report"]
        config.update(outcome)
        config["stack_models"] = [f"stack_fold{i}.txt" for i in range(len(models))]
        config["stack_support"] = bool(args.stack_support)
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)
        log(f"[stacking] adopted: {config_path} updated")
    else:
        log("[stacking] not adopted (no gain on fold 0)")


def run_test(args) -> None:
    log("=" * 70 + "\nSTAGE 5: TEST INFERENCE AND SUBMISSION FILES\n" + "=" * 70)
    with open(os.path.join(args.model_dir, "matching_config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    models = [MatcherModel.load(os.path.join(args.model_dir, name)) for name in config["models"]]
    calibrator = Calibrator.from_dict(config["calibrator"])
    rule = DecisionRule(**config["decision_rule"])
    stack_models = [MatcherModel.load(os.path.join(args.model_dir, n)) for n in config.get("stack_models", [])]
    log(f"[matching] rule: {rule.to_dict()} | second stage: {bool(stack_models)}")

    matching_path = os.path.join(args.output_dir, "matching_results.tsv")
    candidate_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
    stats = predict_and_write(args.test_candidates, args.preprocessed_dir, "test", models, calibrator, rule,
                              matching_path, candidate_path, n_jobs=args.n_jobs, log=log,
                              stack_models=stack_models,
                              cache_root=None if args.no_feature_cache else args.feature_cache,
                              precision=config.get("feature_precision", "float32"),
                              stack_support=config.get("stack_support", False))
    log(f"[matching] wrote {matching_path} and {candidate_path}: {json.dumps(stats)}")
    with open(os.path.join(args.model_dir, "test_prediction_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    validator = os.path.join("student_resource", "utils", "validate_submission.py")
    if os.path.exists(validator):
        log("\n[matching] official validator:")
        cmd = [sys.executable, validator, "--matching", matching_path, "--candidate", candidate_path,
               "--test-dir", os.path.join("student_resource", "dataset", "test")]
        if args.check_ids:
            cmd.append("--check-ids")
        subprocess.run(cmd, check=False)

    if args.s3_bucket:
        try:
            import boto3

            boto3.client("s3").upload_file(matching_path, args.s3_bucket, "submissions/matching_results.tsv")
            log(f"[matching] uploaded to s3://{args.s3_bucket}/submissions/")
        except Exception as exc:  # optional convenience, never fail the run on it
            log(f"[matching] S3 upload skipped: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Stages 3-5: matching and submission")
    parser.add_argument("--mode", choices=["train", "tune", "stack", "test", "all"], default="all")
    parser.add_argument("--preprocessed-dir", default="dataset/preprocessed")
    parser.add_argument("--train-candidates", default="dataset/candidates/train_full_candidate_pairs.parquet")
    parser.add_argument("--test-candidates", default="dataset/candidates/test_candidate_pairs.parquet")
    parser.add_argument("--train-gt", default="student_resource/dataset/train/train_ground_truth.tsv")
    parser.add_argument("--val-gt", default="dataset/val_split/val_ground_truth.tsv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--train-queries", type=int, default=200_000,
                        help="S1 entities sampled per fold for training")
    parser.add_argument("--num-boost-round", type=int, default=3000,
                        help="maximum LightGBM rounds (early stopping decides)")
    parser.add_argument("--stack-support", action="store_true",
                        help="stack mode: add support features (overlap with the S1's other confident candidates)")
    parser.add_argument("--feature-cache", default="dataset/feature_cache",
                        help="directory for cached record-based pair features (built once per split/country)")
    parser.add_argument("--no-feature-cache", action="store_true",
                        help="always compute pair features live (no disk cache)")
    parser.add_argument("--no-orphan-sim", action="store_true",
                        help="train/evaluate on all train S1 instead of the test-like universe "
                             "(by default ~19%% of train S1 are dropped so their records become orphans, "
                             "matching test's records-per-S1 ratio)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=2, help="worker processes for record normalisation")
    parser.add_argument("--check-ids", action="store_true", help="validator: also check that every id exists")
    parser.add_argument("--s3-bucket", default=None, help="optional S3 bucket for a backup of the submission")
    args = parser.parse_args()

    if args.mode in ("train", "all"):
        run_train(args)
    if args.mode == "tune":
        run_tune(args)
    if args.mode == "stack":
        run_stack(args)
    if args.mode in ("test", "all"):
        run_test(args)


if __name__ == "__main__":
    main()
