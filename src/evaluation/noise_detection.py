"""Evaluate noise-suspicion SCORES against a known noise mask (synthetic noise).

A score is a per-sample real number where higher = more suspected of being
mislabeled. With a ground-truth ``noise_mask`` (only available under synthetic
noise) we measure how well the ranking surfaces the corrupted samples.

Ranking quality, not calibrated probability: we report ROC-AUC, Average
Precision (AUPRC), precision@k and recall@k.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_at_k(scores, mask, k: int) -> float:
    """Fraction of the top-k highest-scoring samples that are truly noisy."""
    order = np.argsort(scores)[::-1][:k]
    return float(np.asarray(mask)[order].mean()) if k > 0 else 0.0


def recall_at_k(scores, mask, k: int) -> float:
    """Fraction of all truly-noisy samples captured in the top-k."""
    mask = np.asarray(mask).astype(bool)
    total = int(mask.sum())
    if total == 0:
        return 0.0
    order = np.argsort(scores)[::-1][:k]
    return float(mask[order].sum() / total)


def topk_overlap(scores_a, scores_b, k: int) -> Dict[str, float]:
    """Overlap between the top-k of two score vectors (Jaccard + count)."""
    a = set(np.argsort(scores_a)[::-1][:k].tolist())
    b = set(np.argsort(scores_b)[::-1][:k].tolist())
    inter = len(a & b)
    union = len(a | b)
    return {"intersection": inter, "jaccard": float(inter / union) if union else 0.0}


def noise_detection_metrics(scores, mask, ks: Sequence[int] = (50, 100, 200)) -> Dict[str, float]:
    """ROC-AUC, AUPRC + precision@k / recall@k for the given ks."""
    scores = np.asarray(scores, dtype=float)
    mask = np.asarray(mask).astype(int)
    out: Dict[str, float] = {}
    if mask.sum() == 0 or mask.sum() == len(mask):
        out["roc_auc"] = float("nan")
        out["auprc"] = float("nan")
    else:
        out["roc_auc"] = float(roc_auc_score(mask, scores))
        out["auprc"] = float(average_precision_score(mask, scores))
    out["base_rate"] = float(mask.mean())
    for k in ks:
        kk = min(k, len(scores))
        out[f"precision@{k}"] = precision_at_k(scores, mask, kk)
        out[f"recall@{k}"] = recall_at_k(scores, mask, kk)
    return out
