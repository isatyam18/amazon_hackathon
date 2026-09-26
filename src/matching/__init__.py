"""
Stage 3 & 4: Business Entity Resolution Matching Package.

Includes:
- Feature extraction (RapidFuzz string similarities, numeric matches, address ratios)
- Model training & inference with LightGBM
- Threshold optimization specifically for Macro F_0.5 and singletons
- End-to-end pipeline generating output/matching_results.tsv
"""

from .features import extract_pair_features, extract_features_dataframe
from .model import MatcherModel, train_matcher
from .threshold import evaluate_f05, find_optimal_threshold
from .pipeline import MatchingPipeline

__all__ = [
    "extract_pair_features",
    "extract_features_dataframe",
    "MatcherModel",
    "train_matcher",
    "evaluate_f05",
    "find_optimal_threshold",
    "MatchingPipeline",
]
