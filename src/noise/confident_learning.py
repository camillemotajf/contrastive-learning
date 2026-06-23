"""Confident Learning (Northcutt et al., 2021) as a SUSPICION SCORE.

Out-of-fold predicted probabilities give each sample a "second opinion" from a
model that never trained on it. Per-class confidence thresholds (the confident
joint) decide when that opinion confidently disagrees with the given label.

We expose both:
  * a continuous score  = P(other class) - P(given label)  (for ranking/AUC)
  * a boolean flag       = confident-joint disagreement     (for counts)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def oof_probabilities(X, y, seed: int = 42, n_splits: int = 5,
                      n_estimators: int = 300) -> np.ndarray:
    """Out-of-fold predicted probabilities from a strong RF (no leakage:
    every sample is scored by a fold that excluded it)."""
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(clf, X, y, cv=cv, method="predict_proba", n_jobs=-1)


def confident_learning_scores(y, proba) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(score, flags, thresholds)``.

    score      : P(most-confident other class) - P(given label)  (continuous)
    flags      : confident-joint disagreement (bool)
    thresholds : per-class mean self-confidence
    """
    y = np.asarray(y)
    proba = np.asarray(proba)
    classes = np.unique(y)
    thresholds = np.array([proba[y == c, c].mean() for c in classes])

    eligible = proba >= thresholds[None, :]
    masked = np.where(eligible, proba, -1.0)
    confident_class = masked.argmax(axis=1)
    has_conf = eligible.any(axis=1)
    flags = has_conf & (confident_class != y)

    given_conf = proba[np.arange(len(y)), y]
    other_conf = proba.max(axis=1)
    score = other_conf - given_conf
    return score, flags, thresholds
