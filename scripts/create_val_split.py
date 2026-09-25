"""
Create a stratified validation split for rapid offline experimentation.

Extracts:
- N Source 1 entities (preserving the ~5.6% singleton ratio)
- All corresponding true matches from train_source2.tsv and train_source3.tsv
- A configurable number of distractor records from Source 2 and 3
Outputs saved to dataset/val_split/.
"""

import argparse
import os
import random
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def create_val_split(
    data_dir: str = "student_resource/dataset/train",
    out_dir: str = "dataset/val_split",
    num_s1: int = 20000,
    num_distractors: int = 50000,
    random_seed: int = 42,
):
    random.seed(random_seed)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading ground truth from {data_dir}/train_ground_truth.tsv ...")
    gt_df = pd.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), sep="\t")

    # Identify singletons vs matched
    gt_df["matched_str"] = gt_df["matched_entity_ids"].fillna("").astype(str).str.strip()
    is_singleton = gt_df["matched_str"] == ""

    singletons = gt_df[is_singleton]
    matched = gt_df[~is_singleton]

    # Target 5.6% singletons
    n_singletons = int(num_s1 * 0.056)
    n_matched = num_s1 - n_singletons

    sampled_singletons = singletons.sample(n=min(n_singletons, len(singletons)), random_state=random_seed)
    sampled_matched = matched.sample(n=min(n_matched, len(matched)), random_state=random_seed)

    val_gt = pd.concat([sampled_singletons, sampled_matched]).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    val_s1_ids = set(val_gt["source1_entity_id"])

    # Collect all true target IDs
    true_s2_ids = set()
    true_s3_ids = set()
    for targets in val_gt["matched_str"]:
        if targets:
            for tid in targets.split(","):
                tid = tid.strip()
                if tid.startswith("S2-"):
                    true_s2_ids.add(tid)
                elif tid.startswith("S3-"):
                    true_s3_ids.add(tid)

    print(f"Selected {len(val_s1_ids):,} Source 1 entities.")
    print(f"Ground truth requires {len(true_s2_ids):,} S2 matches and {len(true_s3_ids):,} S3 matches.")

    # Save validation ground truth
    val_gt_clean = val_gt[["source1_entity_id", "matched_entity_ids"]]
    val_gt_clean.to_csv(os.path.join(out_dir, "val_ground_truth.tsv"), sep="\t", index=False)
    print("Saved val_ground_truth.tsv")

    # Filter Source 1
    print("Filtering Source 1...")
    s1_rows = []
    for chunk in pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t", chunksize=100000):
        sub = chunk[chunk["entity_id"].isin(val_s1_ids)]
        if not sub.empty:
            s1_rows.append(sub)
        if sum(len(x) for x in s1_rows) >= len(val_s1_ids):
            break
    val_s1_df = pd.concat(s1_rows, ignore_index=True)
    val_s1_df.to_csv(os.path.join(out_dir, "val_source1.tsv"), sep="\t", index=False)
    print(f"Saved val_source1.tsv with {len(val_s1_df):,} rows.")

    # Filter Source 2 (true matches + distractors)
    print("Filtering Source 2...")
    s2_rows = []
    distractors_s2 = []
    for chunk in pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t", chunksize=150000):
        sub_true = chunk[chunk["entity_id"].isin(true_s2_ids)]
        if not sub_true.empty:
            s2_rows.append(sub_true)
        if len(distractors_s2) < num_distractors // 2:
            rem = chunk[~chunk["entity_id"].isin(true_s2_ids)]
            sample_size = min(len(rem), (num_distractors // 2) - len(distractors_s2))
            distractors_s2.append(rem.sample(n=sample_size, random_state=random_seed))
    val_s2_df = pd.concat(s2_rows + distractors_s2, ignore_index=True).drop_duplicates(subset=["entity_id"])
    val_s2_df.to_csv(os.path.join(out_dir, "val_source2.tsv"), sep="\t", index=False)
    print(f"Saved val_source2.tsv with {len(val_s2_df):,} rows.")

    # Filter Source 3 (true matches + distractors)
    print("Filtering Source 3...")
    s3_rows = []
    distractors_s3 = []
    for chunk in pd.read_csv(os.path.join(data_dir, "train_source3.tsv"), sep="\t", chunksize=150000):
        sub_true = chunk[chunk["entity_id"].isin(true_s3_ids)]
        if not sub_true.empty:
            s3_rows.append(sub_true)
        if len(distractors_s3) < num_distractors // 2:
            rem = chunk[~chunk["entity_id"].isin(true_s3_ids)]
            sample_size = min(len(rem), (num_distractors // 2) - len(distractors_s3))
            distractors_s3.append(rem.sample(n=sample_size, random_state=random_seed))
    val_s3_df = pd.concat(s3_rows + distractors_s3, ignore_index=True).drop_duplicates(subset=["entity_id"])
    val_s3_df.to_csv(os.path.join(out_dir, "val_source3.tsv"), sep="\t", index=False)
    print(f"Saved val_source3.tsv with {len(val_s3_df):,} rows.")

    print("\nValidation split creation complete in:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create validation split")
    parser.add_argument("--data-dir", default="student_resource/dataset/train")
    parser.add_argument("--out-dir", default="dataset/val_split")
    parser.add_argument("--num-s1", type=int, default=10000, help="Number of Source 1 entities")
    parser.add_argument("--num-distractors", type=int, default=30000, help="Number of distractor candidates")
    args = parser.parse_args()

    create_val_split(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        num_s1=args.num_s1,
        num_distractors=args.num_distractors,
    )
