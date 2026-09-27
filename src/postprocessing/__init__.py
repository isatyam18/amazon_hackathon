"""
Stage 5: Post-Processing, Threshold Tuning, Constraint Enforcement, and Submission.
"""

from .consistency import (
    filter_by_country_consistency,
    resolve_source_conflicts,
    build_final_matches_map,
)
from .candidate_optimizer import (
    optimize_candidate_set,
    compute_candidate_efficiency_metrics,
)
from .submission import (
    compute_sha256,
    run_official_validator,
    package_submission_zip,
)

__all__ = [
    "filter_by_country_consistency",
    "resolve_source_conflicts",
    "build_final_matches_map",
    "optimize_candidate_set",
    "compute_candidate_efficiency_metrics",
    "compute_sha256",
    "run_official_validator",
    "package_submission_zip",
]
