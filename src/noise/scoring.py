"""CC-derived suspicion scores + normalisation + ensemble.

All scores follow the convention "higher = more suspicious". None of them is a
probability of noise — they are ranking signals to be combined and reviewed.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def normalize01(score: np.ndarray) -> np.ndarray:
    """Min-max to [0, 1]; flat vectors map to zeros."""
    s = np.asarray(score, dtype=float)
    lo, hi = np.nanmin(s), np.nanmax(s)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)


def cluster_entropy_score(cluster_probs: np.ndarray) -> np.ndarray:
    """Per-sample entropy of the cluster distribution. High entropy = the model
    is unsure which cluster the sample belongs to."""
    p = np.asarray(cluster_probs, dtype=float)
    return -(p * np.log(p + 1e-12)).sum(axis=1)


def cluster_label_mismatch_score(cluster_assignments: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Map each cluster to its majority observed label; score 1 if the sample's
    observed label differs from its cluster's majority label, else 0.

    Soft version: 1 - (fraction of the sample's cluster that shares its label)."""
    cluster_assignments = np.asarray(cluster_assignments)
    y = np.asarray(y)
    score = np.zeros(len(y), dtype=float)
    for c in np.unique(cluster_assignments):
        members = cluster_assignments == c
        labels = y[members]
        if len(labels) == 0:
            continue
        # fraction of this cluster sharing each sample's own label
        for lab in np.unique(labels):
            same = (y == lab) & members
            frac_same = (labels == lab).mean()
            score[same] = 1.0 - frac_same
    return score


def centroid_distance_scores(emb: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """Distances in embedding space relative to class centroids.

    own_distance      : distance to the centroid of the sample's OWN class.
    relative_distance : own_distance - distance to the OPPOSITE class centroid.
                        Positive = closer to the other class (suspicious).
    """
    emb = np.asarray(emb, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)
    centroids = {c: emb[y == c].mean(axis=0) for c in classes}

    own = np.zeros(len(y))
    other = np.zeros(len(y))
    for i in range(len(y)):
        yi = y[i]
        own[i] = np.linalg.norm(emb[i] - centroids[yi])
        others = [np.linalg.norm(emb[i] - centroids[c]) for c in classes if c != yi]
        other[i] = min(others) if others else own[i]
    return {
        "own_distance": own,
        "opposite_distance": other,
        "relative_distance": own - other,  # higher = closer to the wrong class
    }


def ensemble_score(scores: Sequence[np.ndarray], weights: Sequence[float] | None = None) -> np.ndarray:
    """Average of min-max-normalised scores (optionally weighted)."""
    norm = [normalize01(s) for s in scores]
    stack = np.vstack(norm)
    if weights is None:
        return stack.mean(axis=0)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return (stack * w[:, None]).sum(axis=0)
