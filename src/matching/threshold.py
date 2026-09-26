"""
Stage 4: Threshold optimization and Macro F_0.5 evaluation.

Enforces the exact competition evaluation formula:
- Macro-averaged F_0.5 across all Source 1 queries
- F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
- Singletons (no true match) score 1.0 if predicted empty, 0.0 otherwise
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


def compute_entity_f05(predicted_ids: Set[str], ground_truth_ids: Set[str]) -> float:
    """Compute F_0.5 score for a single Source 1 entity."""
    # Singleton case: true entity has no matches
    if not ground_truth_ids:
        return 1.0 if not predicted_ids else 0.0

    # Entity has matches, but model predicted nothing
    if not predicted_ids:
        return 0.0

    tp = len(predicted_ids & ground_truth_ids)
    if tp == 0:
        return 0.0

    precision = tp / len(predicted_ids)
    recall = tp / len(ground_truth_ids)

    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0

    return (1.25 * precision * recall) / denom


def evaluate_f05(
    predictions_by_query: Dict[str, List[str]],
    ground_truth_by_query: Dict[str, Set[str]],
) -> Dict[str, float]:
    """Compute overall Macro F_0.5 and sub-metrics across all queries."""
    scores = []
    tp_total = 0
    fp_total = 0
    fn_total = 0

    all_queries = list(ground_truth_by_query.keys())
    for qid in all_queries:
        gt_set = ground_truth_by_query.get(qid, set())
        pred_set = set(predictions_by_query.get(qid, []))

        s = compute_entity_f05(pred_set, gt_set)
        scores.append(s)

        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        tp_total += tp
        fp_total += fp
        fn_total += fn

    micro_prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0

    macro_f05 = float(np.mean(scores)) if scores else 0.0

    return {
        "macro_f05": macro_f05,
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "num_queries": len(all_queries),
    }


def find_optimal_threshold(
    candidates_df: pd.DataFrame,
    probabilities: np.ndarray,
    ground_truth_by_query: Dict[str, Set[str]],
    threshold_range: Tuple[float, float, float] = (0.30, 0.90, 0.02),
) -> Tuple[float, float, Dict[float, float]]:
    """Sweep decision thresholds and return the threshold maximizing Macro F_0.5.

    Returns:
        best_threshold: float
        best_score: float
        scores_by_threshold: dict mapping threshold -> macro_f05
    """
    df = pd.DataFrame({
        "query_id": candidates_df["query_id"].values,
        "candidate_id": candidates_df["candidate_id"].values,
        "prob": probabilities,
    })

    # Group by query for fast thresholding
    grouped = df.groupby("query_id")
    query_groups = {qid: group for qid, group in grouped}

    all_queries = list(ground_truth_by_query.keys())

    start, stop, step = threshold_range
    thresholds = np.arange(start, stop + step / 2, step)

    best_threshold = 0.50
    best_score = -1.0
    scores_by_threshold = {}

    for t in thresholds:
        preds = {}
        for qid in all_queries:
            if qid in query_groups:
                g = query_groups[qid]
                matches = g[g["prob"] >= t].sort_values("prob", ascending=False)["candidate_id"].tolist()
                preds[qid] = matches
            else:
                preds[qid] = []

        metrics = evaluate_f05(preds, ground_truth_by_query)
        score = metrics["macro_f05"]
        scores_by_threshold[round(float(t), 3)] = score

        if score > best_score:
            best_score = score
            best_threshold = float(t)

    return best_threshold, best_score, scores_by_threshold
