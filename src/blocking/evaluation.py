"""
Blocking-quality metrics against ground truth.

- pair_recall (pair completeness): fraction of true (S1, S2/S3) pairs present
  among the candidates, i.e. the recall ceiling of any Stage 3 matcher.
- f05_ceiling: the leaderboard metric (macro F0.5 over all S1 entities, with
  singletons scoring 1.0 when predicted empty) that a *perfect* matcher would
  reach on these candidates. This is the real upper bound blocking imposes.
- reduction_ratio: 1 - |candidates| / (|S1| x |S2 u S3|).
- recall@k per source, from the fused per-source rank (sizes the budget).
- per-pass recall and unique recall (true pairs found only by that pass).
"""

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from src.blocking.data import ground_truth_pairs


def f_beta(precision: np.ndarray, recall: np.ndarray, beta: float = 0.5) -> np.ndarray:
    b2 = beta * beta
    denom = b2 * precision + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (1 + b2) * precision * recall / denom
    return np.where(denom > 0, f, 0.0)


def macro_f05(pred: pd.DataFrame, gt: pd.DataFrame, pred_col: str = "candidate_entity_id") -> float:
    """Leaderboard metric for exploded predictions (source1_entity_id, pred_col) vs ground truth."""
    truth = ground_truth_pairs(gt)
    n_true = truth.groupby("source1_entity_id").size()
    n_pred = pred.groupby("source1_entity_id").size()
    hits = pred.merge(truth, left_on=["source1_entity_id", pred_col],
                      right_on=["source1_entity_id", "candidate_entity_id"])
    n_hit = hits.groupby("source1_entity_id").size()
    ids = gt["source1_entity_id"]
    t = n_true.reindex(ids, fill_value=0).to_numpy(float)
    p = n_pred.reindex(ids, fill_value=0).to_numpy(float)
    h = n_hit.reindex(ids, fill_value=0).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(p > 0, h / p, 0.0)
        rec = np.where(t > 0, h / t, 0.0)
    score = f_beta(prec, rec)
    score = np.where((t == 0) & (p == 0), 1.0, score)  # correctly predicted singleton
    return float(score.mean())


def evaluate_candidates(
    pairs: pd.DataFrame,
    gt: pd.DataFrame,
    n_targets: Optional[int] = None,
    query_country: Optional[pd.Series] = None,
    ks: Sequence[int] = (1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50),
) -> Dict:
    """Evaluate candidate pairs (source1_entity_id, candidate_entity_id[, source_rank, score_*]).

    `gt` defines the evaluated query set: every S1 id in it counts, including
    those without candidates.
    """
    queries = gt["source1_entity_id"]
    n_q = len(queries)
    pairs = pairs[pairs["source1_entity_id"].isin(set(queries))]
    truth = ground_truth_pairs(gt)
    truth["source"] = truth["candidate_entity_id"].str[:2]

    keep_cols = ["source1_entity_id", "candidate_entity_id"] + [
        c for c in pairs.columns if c == "source_rank" or c.startswith("score_")
    ]
    hit = truth.merge(pairs[keep_cols], on=["source1_entity_id", "candidate_entity_id"], how="left",
                      indicator=True)
    hit["found"] = hit["_merge"] == "both"

    per_q_true = truth.groupby("source1_entity_id").size().reindex(queries, fill_value=0)
    per_q_found = hit[hit["found"]].groupby("source1_entity_id").size().reindex(queries, fill_value=0)
    per_q_cands = pairs.groupby("source1_entity_id").size().reindex(queries, fill_value=0)

    t = per_q_true.to_numpy(float)
    f = per_q_found.to_numpy(float)
    matched = t > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(matched, f / t, 0.0)
    ceiling = np.where(matched, f_beta(np.ones_like(r), r), 1.0)

    report = {
        "queries": int(n_q),
        "singleton_queries": int((~matched).sum()),
        "true_pairs": int(len(truth)),
        "candidate_pairs": int(len(pairs)),
        "candidates_per_query": {
            "mean": float(per_q_cands.mean()),
            "median": float(per_q_cands.median()),
            "p95": float(per_q_cands.quantile(0.95)),
            "max": int(per_q_cands.max()) if n_q else 0,
            "zero_candidate_queries": int((per_q_cands == 0).sum()),
        },
        "pair_recall": float(hit["found"].mean()) if len(hit) else 1.0,
        "query_full_recall": float((f[matched] == t[matched]).mean()) if matched.any() else 1.0,
        "query_zero_recall": float((f[matched] == 0).mean()) if matched.any() else 0.0,
        "f05_ceiling": float(ceiling.mean()) if n_q else 1.0,
        "recall_by_source": hit.groupby("source")["found"].mean().round(5).to_dict(),
    }
    if n_targets:
        report["reduction_ratio"] = float(1.0 - len(pairs) / (n_q * float(n_targets)))
    if query_country is not None:
        hit["country"] = hit["source1_entity_id"].map(query_country)
        report["recall_by_country"] = hit.groupby("country")["found"].mean().round(5).to_dict()

    if "source_rank" in hit.columns:
        rank = hit["source_rank"].to_numpy(float)
        report["recall_at_k_per_source"] = {
            int(k): float(np.nanmean(np.where(hit["found"], rank < k, False))) for k in ks
        }

    score_cols = [c for c in hit.columns if c.startswith("score_")]
    if score_cols:
        found_by = hit[score_cols].fillna(0).to_numpy() > 0
        n_by = found_by.sum(axis=1)
        report["pass_recall"] = {
            c[len("score_"):]: {
                "recall": float(found_by[:, i].mean()),
                "unique_recall": float((found_by[:, i] & (n_by == 1)).mean()),
            }
            for i, c in enumerate(score_cols)
        }
    return report


def format_report(report: Dict) -> str:
    """Human-readable summary of `evaluate_candidates` output."""
    c = report["candidates_per_query"]
    lines = [
        f"queries             : {report['queries']:,} ({report['singleton_queries']:,} singletons)",
        f"true pairs          : {report['true_pairs']:,}",
        f"candidate pairs     : {report['candidate_pairs']:,}  "
        f"(mean {c['mean']:.1f} / median {c['median']:.0f} / p95 {c['p95']:.0f} / max {c['max']} per query, "
        f"{c['zero_candidate_queries']:,} queries with none)",
        f"pair recall         : {report['pair_recall']:.4f}   by source {report['recall_by_source']}",
        f"F0.5 ceiling        : {report['f05_ceiling']:.4f}",
        f"queries full recall : {report['query_full_recall']:.4f}   zero recall: {report['query_zero_recall']:.4f}",
    ]
    if "reduction_ratio" in report:
        lines.append(f"reduction ratio     : {report['reduction_ratio']:.8f}")
    if "recall_by_country" in report:
        lines.append(f"recall by country   : {report['recall_by_country']}")
    if "recall_at_k_per_source" in report:
        lines.append("recall@k/source     : " + ", ".join(
            f"{k}:{v:.4f}" for k, v in report["recall_at_k_per_source"].items()))
    for name, v in report.get("pass_recall", {}).items():
        lines.append(f"  pass {name:<12}: recall {v['recall']:.4f}  unique {v['unique_recall']:.4f}")
    return "\n".join(lines)
