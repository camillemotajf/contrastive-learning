"""Clustering metrics: ARI, NMI, Hungarian accuracy, size/entropy of clusters."""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def clustering_accuracy(y_true, y_pred) -> float:
    """Accuracy under optimal (Hungarian) cluster->class assignment."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = max(int(y_true.max()), int(y_pred.max())) + 1
    cost = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cost[p, t] += 1
    row, col = linear_sum_assignment(-cost)
    mapping = {r: c for r, c in zip(row, col)}
    return float(sum(cost[r, mapping[r]] for r in row) / len(y_true))


def cluster_distribution(y_pred) -> Dict[str, object]:
    """Cluster size distribution + Shannon entropy (collapse diagnostic)."""
    y_pred = np.asarray(y_pred)
    _, counts = np.unique(y_pred, return_counts=True)
    p = counts / counts.sum()
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    return {
        "sizes": counts.tolist(),
        "fractions": p.tolist(),
        "entropy": entropy,
        "n_nonempty": int((counts > 0).sum()),
    }


def clustering_metrics(y_true, y_pred) -> Dict[str, object]:
    return {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "cluster_acc": clustering_accuracy(y_true, y_pred),
        "distribution": cluster_distribution(y_pred),
    }
