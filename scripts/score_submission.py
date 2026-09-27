"""
Score a matching_results.tsv against a ground-truth TSV with the leaderboard metric.

Independent of the pipeline's vectorised evaluation: it parses both files and
applies the reference per-entity F0.5 (src/matching/threshold.compute_entity_f05),
averaged over every S1 entity of the ground truth (singletons: 1.0 only when
predicted empty). S1 entities missing from the submission count as empty.

    python scripts/score_submission.py --matching output/matching_results_fold1.tsv \
        --gt student_resource/dataset/train/train_ground_truth.tsv --only-submitted
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.matching.threshold import compute_entity_f05


def read_lists(path: str) -> dict:
    out = {}
    with open(path, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            s1, _, rest = line.rstrip("\n").partition("\t")
            out[s1] = {x for x in rest.split(",") if x}
    return out


def main():
    parser = argparse.ArgumentParser(description="Leaderboard metric (macro F0.5) for a matching file")
    parser.add_argument("--matching", required=True)
    parser.add_argument("--gt", required=True, help="TSV with source1_entity_id, matched_entity_ids")
    parser.add_argument("--only-submitted", action="store_true",
                        help="average only over S1 entities present in the matching file (e.g. a fold)")
    args = parser.parse_args()

    pred = read_lists(args.matching)
    truth = read_lists(args.gt)
    ids = [q for q in truth if q in pred] if args.only_submitted else list(truth)
    scores = [compute_entity_f05(pred.get(q, set()), truth[q]) for q in ids]
    singles = [s for q, s in zip(ids, scores) if not truth[q]]
    print(f"S1 entities scored : {len(ids):,}")
    print(f"macro F0.5         : {sum(scores) / max(len(scores), 1):.5f}")
    print(f"singleton accuracy : {sum(singles) / max(len(singles), 1):.5f} ({len(singles):,} singletons)")


if __name__ == "__main__":
    main()
