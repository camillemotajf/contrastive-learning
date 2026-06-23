"""Label-efficiency helper: subsample TRAIN labels at a fraction, stratified."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def label_subset(y_tr: np.ndarray, fraction: float, seed: int = 42) -> np.ndarray:
    """Indices of a stratified ``fraction`` of the training set (or all if 1.0)."""
    y_tr = np.asarray(y_tr)
    if fraction >= 1.0:
        return np.arange(len(y_tr))
    idx, _ = train_test_split(
        np.arange(len(y_tr)), train_size=fraction, random_state=seed, stratify=y_tr
    )
    return idx
