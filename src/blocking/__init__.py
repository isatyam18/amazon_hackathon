"""
Stage 2: Scalable candidate generation (blocking).

Typical use:

    from src.blocking import BlockingConfig, BlockingPipeline, load_source

    cfg = BlockingConfig()
    queries = load_source("dataset/preprocessed", "test", 1)
    targets = {"S2": load_source("dataset/preprocessed", "test", 2),
               "S3": load_source("dataset/preprocessed", "test", 3)}
    candidates = BlockingPipeline(cfg).run(queries, targets)
    candidates.write_tsv("output/candidate_pairs.tsv")
"""

from src.blocking.candidates import CandidateSet
from src.blocking.config import BlockingConfig, DenseConfig, PassConfig
from src.blocking.data import load_ground_truth, load_source
from src.blocking.evaluation import evaluate_candidates, format_report
from src.blocking.pipeline import BlockingPipeline

__all__ = [
    "BlockingConfig",
    "BlockingPipeline",
    "CandidateSet",
    "DenseConfig",
    "PassConfig",
    "evaluate_candidates",
    "format_report",
    "load_ground_truth",
    "load_source",
]
