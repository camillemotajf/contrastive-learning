"""KNN-based local label-inconsistency scores.

A sample sitting in a neighbourhood dominated by the OTHER label is locally
inconsistent — a candidate for review. This is NOT a real probability of noise:
a high score can also mean a hard sample, an outlier, or a class-boundary case.
We therefore call it a *KNN noise-suspicion score* / *local inconsistency score*.

Two variants:
  knn_label_disagreement_score          — fraction of neighbours with a
                                          different label (unweighted).
  knn_weighted_disagreement_score       — same, weighted by similarity so closer
                                          neighbours count more.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _neighbours(X, k, metric):
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    dist, idx = nn.kneighbors(X)
    return dist[:, 1:], idx[:, 1:]  # drop self


def knn_label_disagreement_score(X, y, k: int = 20, metric: str = "cosine") -> np.ndarray:
    """Proportion of the k nearest neighbours whose label differs from y[i].

    Example: a bot with 18 human and 2 bot neighbours -> 0.90.
    """
    y = np.asarray(y)
    _, idx = _neighbours(X, k, metric)
    neigh_labels = y[idx]                       # (n, k)
    return (neigh_labels != y[:, None]).mean(axis=1)


def knn_weighted_disagreement_score(X, y, k: int = 20, metric: str = "cosine",
                                    temperature: float | None = None) -> np.ndarray:
    """Similarity-weighted disagreement. Weight = exp(-dist/temperature); when
    ``temperature`` is None it is set to the median neighbour distance."""
    y = np.asarray(y)
    dist, idx = _neighbours(X, k, metric)
    if temperature is None:
        temperature = float(np.median(dist)) + 1e-9
    w = np.exp(-dist / temperature)             # (n, k)
    disagree = (y[idx] != y[:, None]).astype(float)
    return (w * disagree).sum(axis=1) / (w.sum(axis=1) + 1e-12)
