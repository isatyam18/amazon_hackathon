"""
Stage 3 & 4 CLI: Feature Extraction, Model Training, and Submission Inference.

Modes:
  1. train:
     Extracts features, trains LightGBM, tunes decision threshold on validation for Macro F_0.5,
     and saves the model.
     python scripts/run_matching.py --mode train

  2. test:
     Extracts features on test candidates, scores them with the trained model,
     filters by optimal threshold, generates output/matching_results.tsv,
     and validates format.
     python scripts/run_matching.py --mode test

  3. all (default):
     Runs training, threshold tuning, and test inference end-to-end.
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.matching.features import extract_pair_features
from src.matching.model import MatcherModel, train_matcher
from src.matching.threshold import find_optimal_threshold, evaluate_f05
from src.matching.pipeline import generate_matching_results, write_submission


def load_preprocessed_lookup(preprocessed_dir: str, split: str = "train") -> dict:
    """Load preprocessed source files and build fast lookup mapping: entity_id -> dict."""
    print(f"Loading preprocessed lookup records for {split}...")
    lookup = {}
    sources = [1, 2, 3]
    for s in sources:
        tsv_path = os.path.join(preprocessed_dir, f"{split}_source{s}.tsv")
        if os.path.exists(tsv_path):
            df = pd.read_csv(tsv_path, sep="\t", dtype=str)
            for _, r in df.iterrows():
                eid = r.get("entity_id")
                if eid:
                    lookup[eid] = {
                        "clean_name": r.get("clean_name", ""),
                        "core_name": r.get("core_name", ""),
                        "clean_address": r.get("clean_address", ""),
                        "postal_code": r.get("postal_code", ""),
                    }
            print(f"  Loaded {split}_source{s}: {len(df):,} records")
        else:
            print(f"  Warning: {tsv_path} not found.")
    return lookup


def run_train_and_tune(args):
    print("\n" + "=" * 60)
    print("STAGE 3 & 4: MODEL TRAINING & THRESHOLD TUNING")
    print("=" * 60)

    # 1. Load preprocessed text lookup
    train_lookup = load_preprocessed_lookup(args.preprocessed_dir, split="train")

    # 2. Load candidate pairs
    print(f"\nLoading training candidates from: {args.train_candidates}")
    train_candidates = pd.read_parquet(args.train_candidates)
    print(f"Training pairs: {len(train_candidates):,}")

    # 3. Label training candidates using ground truth
    print(f"Loading ground truth: {args.train_gt}")
    gt_df = pd.read_csv(args.train_gt, sep="\t")
    gt_map = {}
    for _, r in gt_df.iterrows():
        qid = r["source1_entity_id"]
        m_str = str(r["matched_entity_ids"]) if pd.notna(r["matched_entity_ids"]) else ""
        gt_map[qid] = {m.strip() for m in m_str.split(",") if m.strip()}

    labels = []
    q_ids = train_candidates["query_id"].values
    c_ids = train_candidates["candidate_id"].values
    for qid, cid in zip(q_ids, c_ids):
        labels.append(1 if cid in gt_map.get(qid, set()) else 0)
    y_train = np.array(labels, dtype=int)
    print(f"Positive pairs: {y_train.sum():,} ({y_train.mean()*100:.2f}%), Negative pairs: {(1-y_train).sum():,}")

    # 4. Extract Stage 3 Features for Training
    print("\nExtracting fine-grained Stage 3 features for training pairs...")
    t0 = time.time()
    X_train = extract_pair_features(train_candidates, train_lookup)
    print(f"Extracted {X_train.shape[1]} features in {time.time()-t0:.1f}s.")

    # 5. Handle Validation Set
    X_val, y_val = None, None
    val_candidates = None
    if os.path.exists(args.val_candidates):
        print(f"\nLoading validation candidates from: {args.val_candidates}")
        val_candidates = pd.read_parquet(args.val_candidates)
        val_gt_df = pd.read_csv(args.val_gt, sep="\t") if os.path.exists(args.val_gt) else gt_df
        val_gt_map = {}
        for _, r in val_gt_df.iterrows():
            qid = r["source1_entity_id"]
            m_str = str(r["matched_entity_ids"]) if pd.notna(r["matched_entity_ids"]) else ""
            val_gt_map[qid] = {m.strip() for m in m_str.split(",") if m.strip()}

        val_labels = [1 if cid in val_gt_map.get(qid, set()) else 0
                      for qid, cid in zip(val_candidates["query_id"], val_candidates["candidate_id"])]
        y_val = np.array(val_labels, dtype=int)
        X_val = extract_pair_features(val_candidates, train_lookup)

    # 6. Train LightGBM Model
    print("\nTraining LightGBM Matcher...")
    model = train_matcher(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_boost_round=args.num_boost_round,
    )

    # Save model
    model.save(args.model_path)
    print(f"Model saved to: {args.model_path}")

    # Show top features
    print("\nTop 10 Feature Importances (Gain):")
    print(model.get_feature_importances().head(10).to_string(index=False))

    # 7. Tune Decision Threshold for Macro F_0.5
    optimal_threshold = args.threshold
    if val_candidates is not None and len(val_candidates) > 0:
        print("\nOptimizing Decision Threshold for Macro F_0.5 on Validation Split...")
        val_probs = model.predict_proba(X_val)
        best_t, best_score, score_curve = find_optimal_threshold(
            val_candidates, val_probs, val_gt_map, threshold_range=(0.40, 0.90, 0.02)
        )
        print(f"\nOptimal Decision Threshold: {best_t:.2f} -> Validation Macro F_0.5 = {best_score:.4f}")
        optimal_threshold = best_t

        # Save report
        report = {
            "best_threshold": best_t,
            "best_macro_f05": best_score,
            "threshold_curve": score_curve,
        }
        with open("dataset/candidates/threshold_tuning_report.json", "w") as f:
            json.dump(report, f, indent=2)

    return model, optimal_threshold


def run_test_inference(args, model, threshold):
    print("\n" + "=" * 60)
    print("STAGE 4: TEST INFERENCE & SUBMISSION GENERATION")
    print("=" * 60)

    # 1. Load test preprocessed lookup
    test_lookup = load_preprocessed_lookup(args.preprocessed_dir, split="test")

    # 2. Load test candidate pairs
    print(f"\nLoading test candidates from: {args.test_candidates}")
    test_candidates = pd.read_parquet(args.test_candidates)
    print(f"Total test candidate pairs to score: {len(test_candidates):,}")

    # 3. Load all test query IDs (every S1 query must appear in submission!)
    test_s1_path = os.path.join(args.preprocessed_dir, "test_source1.tsv")
    if not os.path.exists(test_s1_path):
        test_s1_path = "student_resource/dataset/test/test_source1.tsv"
    test_s1_df = pd.read_csv(test_s1_path, sep="\t", usecols=["entity_id"])
    all_query_ids = list(test_s1_df["entity_id"])
    print(f"Total test Source 1 queries: {len(all_query_ids):,}")

    # 4. Extract Stage 3 Features for Test
    print("\nExtracting features for test candidate pairs...")
    t0 = time.time()
    X_test = extract_pair_features(test_candidates, test_lookup)
    print(f"Extracted {X_test.shape[1]} features in {time.time()-t0:.1f}s.")

    # 5. Predict Probabilities & Generate Submission Table
    print(f"\nRunning model inference with decision threshold tau = {threshold:.2f}...")
    probs = model.predict_proba(X_test)
    submission_df = generate_matching_results(test_candidates, probs, all_query_ids, threshold=threshold)

    # Summary of matches
    matched_count = (submission_df["matched_entity_ids"] != "").sum()
    singleton_count = (submission_df["matched_entity_ids"] == "").sum()
    print(f"Predicted Matched Entities : {matched_count:,} ({matched_count/len(all_query_ids)*100:.1f}%)")
    print(f"Predicted Singletons (Empty): {singleton_count:,} ({singleton_count/len(all_query_ids)*100:.1f}%)")

    # Write output file
    write_submission(submission_df, args.out_file)

    # 6. Validate with official challenge validator
    validator_path = "student_resource/utils/validate_submission.py"
    if os.path.exists(validator_path):
        print("\nValidating submission against official competition rules...")
        test_dir = "student_resource/dataset/test"
        cand_arg = f"--candidate {args.test_candidate_tsv}" if os.path.exists(args.test_candidate_tsv) else ""
        cmd = f"python {validator_path} --matching {args.out_file} {cand_arg} --test-dir {test_dir}"
        ret = os.system(cmd)
        if ret == 0:
            print("\nSUCCESS: Submission passed all validation checks!")
        else:
            print("\nWARNING: Validator reported issues. Check terminal output above.")

    # 7. Optional S3 Upload
    if args.s3_bucket:
        try:
            import boto3
            print(f"\nUploading artifacts to S3 bucket: {args.s3_bucket}...")
            s3 = boto3.client("s3")
            s3.upload_file(args.out_file, args.s3_bucket, f"submissions/{os.path.basename(args.out_file)}")
            if os.path.exists(args.model_path):
                s3.upload_file(args.model_path, args.s3_bucket, f"models/{os.path.basename(args.model_path)}")
            print(f"S3 upload complete! Accessible at s3://{args.s3_bucket}/")
        except Exception as e:
            print(f"S3 upload skipped or failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Stage 3 & 4 Matcher Pipeline")
    parser.add_argument("--mode", choices=["train", "test", "all"], default="all")
    parser.add_argument("--preprocessed-dir", default="dataset/preprocessed")
    parser.add_argument("--train-candidates", default="dataset/candidates/train_candidate_pairs.parquet")
    parser.add_argument("--val-candidates", default="dataset/candidates/val_candidate_pairs.parquet")
    parser.add_argument("--test-candidates", default="dataset/candidates/test_candidate_pairs.parquet")
    parser.add_argument("--test-candidate-tsv", default="output/candidate_pairs.tsv")
    parser.add_argument("--train-gt", default="student_resource/dataset/train/train_ground_truth.tsv")
    parser.add_argument("--val-gt", default="dataset/val_split/val_ground_truth.tsv")
    parser.add_argument("--model-path", default="models/lightgbm_matcher.txt")
    parser.add_argument("--out-file", default="output/matching_results.tsv")
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument("--num-boost-round", type=int, default=300)
    parser.add_argument("--s3-bucket", default=None, help="Optional S3 bucket name to backup results")
    args = parser.parse_args()

    model = None
    threshold = args.threshold

    if args.mode in ("train", "all"):
        model, threshold = run_train_and_tune(args)

    if args.mode in ("test", "all"):
        if model is None:
            if not os.path.exists(args.model_path):
                sys.exit(f"Error: Model file {args.model_path} not found. Run with --mode train first.")
            print(f"Loading existing model from {args.model_path}...")
            model = MatcherModel.load(args.model_path)
        run_test_inference(args, model, threshold)


if __name__ == "__main__":
    main()
