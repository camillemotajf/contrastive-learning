"""Self-supervised contrastive pretraining (NT-Xent / InfoNCE, NO labels).

Frozen backbone + linear probe is the standard linear-evaluation protocol. Used
both as a label-efficiency baseline and as a representation-quality reference.
This variant augments in SVD space (mask + Gaussian) to stay comparable with the
original SSL experiment; the HTTP-text augmentations live in the CC trainer.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import TrafficEncoder
from src.losses.contrastive_losses import InstanceLoss


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(),
                                 nn.Linear(in_dim, out_dim))

    def forward(self, h):
        return F.normalize(self.net(h), dim=1)


def _augment(x, mask_prob=0.15, noise_std=0.05):
    mask = (torch.rand_like(x) > mask_prob).float()
    return x * mask + torch.randn_like(x) * noise_std


def ssl_pretrain(Xtr: np.ndarray, embed_dim: int = 64, epochs: int = 40,
                 batch_size: int = 256, lr: float = 1e-3):
    """Return (frozen backbone, train embeddings)."""
    device = _device()
    Xtr_t = torch.tensor(Xtr, device=device)
    n = Xtr.shape[0]
    backbone = TrafficEncoder(Xtr.shape[1], embed_dim=embed_dim).to(device)
    proj = ProjectionHead(embed_dim, 32).to(device)
    opt = torch.optim.Adam(list(backbone.parameters()) + list(proj.parameters()), lr=lr)
    crit = InstanceLoss(temperature=0.5).to(device)

    backbone.train(); proj.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n - batch_size + 1, batch_size):
            xb = Xtr_t[perm[i:i + batch_size]]
            loss = crit(proj(backbone(_augment(xb))), proj(backbone(_augment(xb))))
            opt.zero_grad(); loss.backward(); opt.step()

    backbone.eval()
    with torch.no_grad():
        emb_tr = backbone(Xtr_t).cpu().numpy()
    return backbone, emb_tr


def backbone_embed(backbone, X: np.ndarray) -> np.ndarray:
    device = _device()
    with torch.no_grad():
        return backbone(torch.tensor(X, device=device)).cpu().numpy()
