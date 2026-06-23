"""Backward-compatibility shim — noise tooling moved to :mod:`src.noise`.

Keeps the original function names/signatures used by older scripts/notebooks.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.preprocessing import manual_features as _manual_features  # noqa: F401
from src.noise.confident_learning import (  # noqa: F401
    oof_probabilities,
    confident_learning_scores,
)
from src.noise.heuristic_rules import heuristic_flags


def build_full_features(headers, requests, seed=42, svd_dim=64):
    """Label-free features on the FULL corpus (no labels used, so fitting on all
    rows leaks nothing). Same behaviour as the original implementation."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=5000)
    X = TruncatedSVD(n_components=svd_dim, random_state=seed).fit_transform(
        vec.fit_transform(headers)
    )
    X = np.hstack([X, _manual_features(headers, requests)])
    mean, std = X.mean(0), X.std(0) + 1e-8
    return ((X - mean) / std).astype(np.float32)


def confident_learning(y, proba):
    """Legacy order: returns ``(flags, suspicion, thresholds)``."""
    score, flags, thresholds = confident_learning_scores(y, proba)
    return flags, score, thresholds


def heuristic_labels(headers, requests):
    """Legacy weak labels: 1 = almost-certainly bot, -1 = no confident heuristic."""
    flags = heuristic_flags(headers, requests)["heuristic_bot_flag"]
    out = np.full(len(headers), -1, dtype=np.int64)
    out[flags] = 1
    return out


__all__ = [
    "build_full_features", "oof_probabilities", "confident_learning",
    "heuristic_labels", "_manual_features",
]
