"""Backward-compatibility shim — metrics moved to :mod:`src.evaluation`."""
from src.evaluation.classification import linear_probe_metrics  # noqa: F401
from src.evaluation.clustering import (  # noqa: F401
    clustering_accuracy,
    clustering_metrics as _clustering_metrics_full,
)


def clustering_metrics(y_true, y_pred):
    """Original 3-key shape (ari/nmi/cluster_acc) for legacy callers."""
    m = _clustering_metrics_full(y_true, y_pred)
    return {"ari": m["ari"], "nmi": m["nmi"], "cluster_acc": m["cluster_acc"]}


__all__ = ["linear_probe_metrics", "clustering_accuracy", "clustering_metrics"]
