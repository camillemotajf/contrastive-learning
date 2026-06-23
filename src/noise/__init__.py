"""Label-noise tooling: synthetic noise, KNN/CL scores, heuristics, ensemble.

Honest naming throughout: these produce *suspicion scores* / *candidates for
review*, NOT a "real probability of noise". We only have a ground truth for
SYNTHETIC noise; on real data the scores are a ranking, not a verdict.
"""
from .synthetic_noise import inject_synthetic_label_noise
from .knn_noise import knn_label_disagreement_score, knn_weighted_disagreement_score
from .heuristic_rules import heuristic_flags, heuristic_bot_flag
from .confident_learning import oof_probabilities, confident_learning_scores
from .scoring import (
    centroid_distance_scores, cluster_entropy_score, cluster_label_mismatch_score,
    normalize01, ensemble_score,
)

__all__ = [
    "inject_synthetic_label_noise",
    "knn_label_disagreement_score",
    "knn_weighted_disagreement_score",
    "heuristic_flags",
    "heuristic_bot_flag",
    "oof_probabilities",
    "confident_learning_scores",
    "centroid_distance_scores",
    "cluster_entropy_score",
    "cluster_label_mismatch_score",
    "normalize01",
    "ensemble_score",
]
