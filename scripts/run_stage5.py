"""
Stage 5 CLI: Chunked/Patched Inference, Conflict Resolution, Candidate Budget Optimization,
and Submission Packaging.

Processes the complete 86,627,063 test candidate pairs in memory-safe patches (chunks),
allowing the entire dataset to be scored on an 8 GB laptop without running out of RAM.
"""

import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.matching.features import compute_pair_features
from src.matching.model import MatcherModel
from src.postprocessing.consistency import (
    filter_by_country_consistency,
    resolve_source_conflicts,
    build_final_matches_map,
)
from src.postprocessing.candidate_optimizer import (
    optimize_candidate_set,
    compute_candidate_efficiency_metrics,
)
from src.postprocessing.submission import (
    run_official_validator,
    package_submission_zip,
)


def load_preprocessed_lookup(preprocessed_dir: str, split: str = "test") -> dict:
    """Load preprocessed text into memory for fast Stage 3 feature extraction."""
    print(f"Loading {split} preprocessed text records into memory...")
    t0 = time.time()
    lookup = {}
    for s in (1, 2, 3):
        p_path = os.path.join(preprocessed_dir, f"{split}_source{s}.parquet")
        t_path = os.path.join(preprocessed_dir, f"{split}_source{s}.tsv")
        if os.path.exists(p_path):
            df = pd.read_parquet(p_path, columns=["entity_id", "clean_name", "clean_address", "country"])
        elif os.path.exists(t_path):
            df = pd.read_csv(t_path, sep="\t", dtype=str, usecols=["entity_id", "clean_name", "clean_address", "country"])
        else:
            continue
        for eid, name, addr, c in zip(df["entity_id"], df["clean_name"], df["clean_address"], df["country"]):
            lookup[eid] = {
                "name": str(name or ""),
                "addr": str(addr or ""),
                "country": str(c or "").strip().lower(),
            }
        print(f"  Loaded {split}_source{s}: {len(df):,} records")
    print(f"Lookup dictionary ready ({len(lookup):,} entities in {time.time()-t0:.1f}s)")
    return lookup


def extract_patch_features(
    batch_df: pd.DataFrame,
    lookup: dict,
    feature_names: List[str],
) -> np.ndarray:
    """Fast vectorized Stage 3 feature extraction on a single patch of pairs."""
    q_ids = batch_df["source1_entity_id"].values
    c_ids = batch_df["candidate_entity_id"].values

    n = len(batch_df)
    from concurrent.futures import ThreadPoolExecutor

    def extract_slice(indices):
        sub_rows = []
        for idx in indices:
            qid, cid = q_ids[idx], c_ids[idx]
            qr = lookup.get(qid, {})
            cr = lookup.get(cid, {})
            sub_rows.append(compute_pair_features(
                q_name=qr.get("name", ""),
                q_core=qr.get("name", ""),
                q_addr=qr.get("addr", ""),
                q_zip="",
                c_name=cr.get("name", ""),
                c_core=cr.get("name", ""),
                c_addr=cr.get("addr", ""),
                c_zip="",
            ))
        return sub_rows

    n_threads = 8
    step = (n + n_threads - 1) // n_threads
    slices = [range(i * step, min((i + 1) * step, n)) for i in range(n_threads) if i * step < n]

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = list(executor.map(extract_slice, slices))

    feature_rows = [row for sub in results for row in sub]
    feat_df = pd.DataFrame(feature_rows)

    # Attach precomputed Stage 2 features
    for col in ["candidate_score", "source_rank", "score_name", "score_addr", "score_name_addr",
                "sim_name_char", "sim_name_phon", "sim_addr_word", "sim_addr_bigram", "sim_name_addr"]:
        if col in batch_df.columns:
            feat_df[col] = batch_df[col].astype(float).values
        else:
            feat_df[col] = 0.0

    feat_df["is_s2"] = batch_df["source"].astype(str).str.startswith("S2").astype(float).values

    # Reorder columns exactly as model expects
    for col in feature_names:
        if col not in feat_df.columns:
            feat_df[col] = 0.0

    return feat_df[feature_names].values


def main():
    parser = argparse.ArgumentParser(description="Stage 5: Patched/Chunked Inference & Submission Packaging")
    parser.add_argument("--candidates", default="dataset/candidates/test_candidate_pairs.parquet",
                        help="Path to test candidate pairs parquet from Stage 2")
    parser.add_argument("--model-path", default="models/lightgbm_matcher.txt",
                        help="Path to trained LightGBM model from Stage 4")
    parser.add_argument("--test-dir", default="student_resource/dataset/test",
                        help="Path to raw test directory")
    parser.add_argument("--preprocessed-dir", default="dataset/preprocessed",
                        help="Path to preprocessed files")
    parser.add_argument("--threshold", type=float, default=0.66,
                        help="Decision threshold tau (default 0.66 based on tuning report)")
    parser.add_argument("--patch-size", type=int, default=250000,
                        help="Number of candidate pairs per patch/chunk to process in memory")
    parser.add_argument("--prefilter-score", type=float, default=0.20,
                        help="Pre-filter candidate_score below this threshold (they cannot pass tau)")
    parser.add_argument("--max-candidates-per-source", type=int, default=15,
                        help="Target candidate budget per source to maximize scaling rank bonus")
    parser.add_argument("--out-dir", default="output",
                        help="Output directory for submission files")
    parser.add_argument("--no-zip", action="store_true", help="Skip creating submission.zip")
    args = parser.parse_args()

    t_start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "=" * 75)
    print("STAGE 5: FULL-DATASET PATCHED INFERENCE & SUBMISSION PACKAGING")
    print("=" * 75)

    # 1. Load required test query IDs
    test_s1_path = os.path.join(args.test_dir, "test_source1.tsv")
    if not os.path.exists(test_s1_path):
        test_s1_path = os.path.join(args.preprocessed_dir, "test_source1.parquet")
        if not os.path.exists(test_s1_path):
            test_s1_path = os.path.join(args.preprocessed_dir, "test_source1.tsv")

    print(f"1. Reading required test query IDs from: {test_s1_path}")
    if test_s1_path.endswith(".parquet"):
        s1_df = pd.read_parquet(test_s1_path, columns=["entity_id"])
    else:
        s1_df = pd.read_csv(test_s1_path, sep="\t", dtype=str, usecols=["entity_id"])
    all_query_ids = s1_df["entity_id"].tolist()
    print(f"   Total required Source 1 queries: {len(all_query_ids):,}")

    # 2. Check candidate file
    if not os.path.exists(args.candidates):
        sys.exit(f"Error: Candidate file {args.candidates} not found. Run Stage 2 first.")

    pfile = pq.ParquetFile(args.candidates)
    total_pairs = pfile.metadata.num_rows
    total_batches = (total_pairs + args.patch_size - 1) // args.patch_size
    print(f"\n2. Complete Test Dataset candidate pairs: {total_pairs:,}")
    print(f"   Processing in {total_batches:,} patches of {args.patch_size:,} pairs each.")

    # 3. Load trained model
    print(f"\n3. Loading trained LightGBM model: {args.model_path}")
    if not os.path.exists(args.model_path):
        sys.exit(f"Error: Model {args.model_path} not found. Run training first.")
    model = MatcherModel.load(args.model_path)
    feature_names = model.feature_names
    print(f"   Model feature count: {len(feature_names)}")

    # 4. Load preprocessed text lookup
    test_lookup = load_preprocessed_lookup(args.preprocessed_dir, split="test")

    # 5. Process patches
    print(f"\n4. Scoring {total_pairs:,} candidate pairs in patches (tau = {args.threshold:.2f})...")
    passing_pairs_list = []
    top_candidates_for_bonus = []
    scored_total = 0
    t_loop_start = time.time()

    for batch_idx, batch in enumerate(pfile.iter_batches(batch_size=args.patch_size), 1):
        batch_df = batch.to_pandas()
        scored_total += len(batch_df)

        # Country check (skip cross-country pairs immediately)
        q_countries = [test_lookup.get(q, {}).get("country", "") for q in batch_df["source1_entity_id"]]
        c_countries = [test_lookup.get(c, {}).get("country", "") for c in batch_df["candidate_entity_id"]]
        same_country = [(qc == cc or not qc or not cc) for qc, cc in zip(q_countries, c_countries)]
        batch_df = batch_df[same_country].copy()

        # Pre-filter candidate_score below threshold (saving 80%+ of CPU RapidFuzz calculations)
        promising_mask = batch_df["candidate_score"] >= args.prefilter_score
        promising_df = batch_df[promising_mask].copy()

        if not promising_df.empty:
            X_batch = extract_patch_features(promising_df, test_lookup, feature_names)
            probs = model.predict_proba(X_batch)
            promising_df["probability"] = probs

            # Filter pairs meeting threshold tau
            passed = promising_df[promising_df["probability"] >= args.threshold][
                ["source1_entity_id", "candidate_entity_id", "probability"]
            ]
            if not passed.empty:
                passing_pairs_list.append(passed)

            # Keep top candidates per query for candidate set optimization
            keep_cols = ["source1_entity_id", "candidate_entity_id", "candidate_score", "source_rank"]
            available_cols = [c for c in keep_cols if c in promising_df.columns]
            top_candidates_for_bonus.append(promising_df[available_cols])

        # Progress update every 10 patches
        if batch_idx % 10 == 0 or batch_idx == total_batches:
            elapsed = time.time() - t_loop_start
            rate = scored_total / elapsed if elapsed > 0 else 0
            eta = (total_pairs - scored_total) / rate if rate > 0 else 0
            n_passed = sum(len(p) for p in passing_pairs_list)
            print(f"   Patch {batch_idx:>3}/{total_batches} ({scored_total/total_pairs*100:>5.1f}%) | "
                  f"Passed: {n_passed:,} | Rate: {rate:,.0f} pairs/s | ETA: {eta/60:.1f}m")

        del batch_df, promising_df
        gc.collect()

    print(f"\n   Completed full-dataset scoring in {(time.time()-t_loop_start)/60:.1f} minutes!")

    # 6. Resolve Graph Conflicts (at most 1 match per source partition)
    print("\n5. Applying Stage 5 Conflict Resolution (Max 1 match in S2, 1 in S3)...")
    if passing_pairs_list:
        all_passing = pd.concat(passing_pairs_list, ignore_index=True)
        resolved_matches = resolve_source_conflicts(all_passing, threshold=args.threshold, max_per_source=1)
    else:
        resolved_matches = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "probability"])

    print(f"   Final accepted match pairs: {len(resolved_matches):,}")
    matches_map = build_final_matches_map(resolved_matches, all_query_ids)

    # 7. Write output/matching_results.tsv
    matching_tsv = os.path.join(args.out_dir, "matching_results.tsv")
    print(f"\n6. Writing final matching results: {matching_tsv}")
    matching_rows = [{"source1_entity_id": qid, "matched_entity_ids": matches_map[qid]} for qid in all_query_ids]
    matching_df = pd.DataFrame(matching_rows)
    matching_df.to_csv(matching_tsv, sep="\t", index=False)

    num_singletons = sum(1 for v in matches_map.values() if not v)
    print(f"   Total rows: {len(matching_df):,}")
    print(f"   Singletons: {num_singletons:,} ({num_singletons/len(all_query_ids)*100:.2f}%)")
    print(f"   Matches   : {len(all_query_ids) - num_singletons:,} ({(1 - num_singletons/len(all_query_ids))*100:.2f}%)")

    # 8. Optimize candidate_pairs.tsv for scaling rank bonus
    candidate_tsv = os.path.join(args.out_dir, "candidate_pairs.tsv")
    print(f"\n7. Optimizing candidate set for Scaling Rank Bonus: {candidate_tsv}")
    if top_candidates_for_bonus:
        top_cands_df = pd.concat(top_candidates_for_bonus, ignore_index=True)
        opt_candidates_df = optimize_candidate_set(
            top_cands_df,
            matches_map,
            all_query_ids,
            max_candidates_per_source=args.max_candidates_per_source,
        )
    else:
        # Fallback to matches only
        opt_candidates_df = pd.DataFrame([
            {"source1_entity_id": qid, "candidate_entity_ids": matches_map[qid]} for qid in all_query_ids
        ])

    opt_candidates_df.to_csv(candidate_tsv, sep="\t", index=False)
    metrics = compute_candidate_efficiency_metrics(opt_candidates_df, matching_df)
    print(f"   Mean candidates per S1 entity: {metrics['mean_candidates_per_query']:.2f}")
    print(f"   Subset rule violations: {metrics['subset_rule_violations']} (0 required)")

    # 9. Run official validator
    print("\n8. Executing Official Submission Validator:")
    valid_ok, valid_lines = run_official_validator(
        matching_tsv,
        candidate_path=candidate_tsv,
        test_dir=args.test_dir,
    )
    for line in valid_lines:
        print(f"   [Validator] {line}")

    if valid_ok:
        print("\n   >>> VALIDATION PASSED! Both TSVs comply 100% with official rules! <<<")

    # 10. Package submission zip
    if not args.no_zip:
        print("\n9. Packaging Final Submission Zip:")
        zip_path = os.path.join(args.out_dir, "submission.zip")
        manifest = package_submission_zip(matching_tsv, candidate_tsv, output_zip_path=zip_path)
        print(f"   Archive Path : {zip_path} ({manifest['zip_size_mb']} MB)")
        print(f"   Archive SHA256: {manifest['zip_sha256']}")

    print("\n" + "=" * 75)
    print(f"STAGE 5 FULL-DATASET INFERENCE COMPLETE in {(time.time()-t_start)/60:.1f} minutes!")
    print("=" * 75)


if __name__ == "__main__":
    main()
