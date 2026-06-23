"""Supervised metrics + linear-probe (linear evaluation protocol)."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)


def classification_metrics(y_true, proba, threshold: float = 0.5) -> Dict[str, float]:
    """F1, ROC-AUC, accuracy, precision, recall from positive-class scores."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, proba)),
        "acc": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }


def linear_probe_metrics(emb_tr, y_tr, emb_te, y_te, seed: int = 42) -> Dict[str, float]:
    """Freeze embeddings, fit a logistic probe on train, evaluate on test."""
    clf = LogisticRegression(max_iter=2000, random_state=seed).fit(emb_tr, y_tr)
    proba = clf.predict_proba(emb_te)[:, 1]
    return classification_metrics(y_te, proba)
