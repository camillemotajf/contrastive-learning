"""Reproducible seeding across random, numpy and torch."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic_torch: bool = False) -> None:
    """Seed every RNG we touch. Call at the top of each experiment.

    ``deterministic_torch`` trades a little speed for bitwise reproducibility on
    CUDA; off by default because it is not needed for the conclusions here.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def new_rng(seed: int) -> np.random.Generator:
    """A fresh NumPy Generator — preferred over the legacy global state for
    augmentations and noise injection (independent, picklable, seedable)."""
    return np.random.default_rng(seed)
