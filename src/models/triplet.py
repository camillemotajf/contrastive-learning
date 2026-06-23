"""Supervised contrastive baseline — TripletMarginLoss over the encoder."""
from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from src.models.encoder import TrafficEncoder


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_triplet(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
                  embed_dim: int = 32, epochs: int = 20, batch_size: int = 256,
                  lr: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """Train an encoder with triplets sampled from TRAIN labels; return
    (train_embeddings, test_embeddings). Operates on already-normalised features.
    """
    device = _device()
    Xtr_t = torch.tensor(Xtr, device=device)
    Xte_t = torch.tensor(Xte, device=device)
    ytr = np.asarray(ytr)
    cls_idx = {c: np.where(ytr == c)[0] for c in np.unique(ytr)}
    classes = list(cls_idx)
    n = len(ytr)
    if len(classes) < 2:
        raise ValueError("triplet training needs >=2 classes")

    model = TrafficEncoder(Xtr.shape[1], embed_dim=embed_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.TripletMarginLoss(margin=1.0, p=2)

    model.train()
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n - batch_size + 1, batch_size):
            a = perm[i:i + batch_size]
            al = ytr[a]
            p = np.array([np.random.choice(cls_idx[l]) for l in al])
            nl = np.array([random.choice([c for c in classes if c != l]) for l in al])
            ng = np.array([np.random.choice(cls_idx[l]) for l in nl])
            loss = crit(model(Xtr_t[a]), model(Xtr_t[p]), model(Xtr_t[ng]))
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        return model(Xtr_t).cpu().numpy(), model(Xte_t).cpu().numpy()
