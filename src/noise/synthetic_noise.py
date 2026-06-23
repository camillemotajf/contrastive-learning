"""Inject controlled label noise so we have a GROUND TRUTH for noise detection.

On the real dataset we do not know which labels are wrong, so we cannot measure
whether a suspicion score actually finds noise. Synthetic noise solves this: we
flip a known set of labels and keep a ``noise_mask``; any detector is then graded
against that mask (ROC-AUC, precision@k, ...).

Modes
-----
symmetric        : flip a random ``noise_rate`` fraction across all classes.
bot_to_human     : only flip true bots (1) -> human (0).
human_to_bot     : only flip true humans (0) -> bot (1).
class_conditional: asymmetric rates per class (defaults skew towards bot->human,
                   the realistic direction — a missed bot looks human).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def inject_synthetic_label_noise(
    y: np.ndarray,
    noise_rate: float,
    mode: str = "symmetric",
    random_state: int = 42,
    class_rates: Optional[Dict[int, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(y_noisy, noise_mask)`` where ``noise_mask[i]`` is True iff the
    label of sample ``i`` was artificially changed.

    Binary labels assumed (0 = human/unsafe, 1 = bot).
    """
    rng = np.random.default_rng(random_state)
    y = np.asarray(y).copy()
    y_noisy = y.copy()
    mask = np.zeros(len(y), dtype=bool)

    def flip_subset(candidate_idx: np.ndarray, rate: float) -> None:
        if len(candidate_idx) == 0 or rate <= 0:
            return
        n_flip = int(round(rate * len(candidate_idx)))
        if n_flip == 0:
            return
        chosen = rng.choice(candidate_idx, size=n_flip, replace=False)
        y_noisy[chosen] = 1 - y_noisy[chosen]  # binary flip
        mask[chosen] = True

    if mode == "symmetric":
        flip_subset(np.arange(len(y)), noise_rate)
    elif mode == "bot_to_human":
        flip_subset(np.where(y == 1)[0], noise_rate)
    elif mode == "human_to_bot":
        flip_subset(np.where(y == 0)[0], noise_rate)
    elif mode == "class_conditional":
        rates = class_rates or {1: noise_rate, 0: noise_rate * 0.5}
        for cls, rate in rates.items():
            flip_subset(np.where(y == cls)[0], rate)
    else:
        raise ValueError(f"Unknown noise mode: {mode}")

    return y_noisy, mask
