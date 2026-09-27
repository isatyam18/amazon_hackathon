"""
Stages 3-5: pair features, LightGBM matcher, decision rule and submission output.

- records.py   RecordStore: per-record normalised fields and token matrices
- features.py  bulk pair features (rapidfuzz cpdist, token overlaps, house numbers)
- context.py   query-side and target-side competition features
- model.py     LightGBM matcher (MatcherModel, train_matcher)
- decision.py  calibration, expected-F0.5 set selection, one-S1-per-record exclusivity
- threshold.py exact leaderboard metric (per-S1 F0.5, singletons) and threshold sweep
- workflow.py  out-of-fold training, decision tuning, evaluation, output writers
- pipeline.py  simple threshold-based submission helpers
"""

from .decision import Calibrator, DecisionRule, apply_rule
from .features import pair_features
from .model import MatcherModel, train_matcher
from .pipeline import MatchingPipeline, generate_matching_results, write_submission
from .records import RecordStore
from .threshold import compute_entity_f05, evaluate_f05, find_optimal_threshold, macro_f05_arrays

__all__ = [
    "Calibrator",
    "DecisionRule",
    "MatcherModel",
    "MatchingPipeline",
    "RecordStore",
    "apply_rule",
    "compute_entity_f05",
    "evaluate_f05",
    "find_optimal_threshold",
    "generate_matching_results",
    "macro_f05_arrays",
    "pair_features",
    "train_matcher",
    "write_submission",
]
