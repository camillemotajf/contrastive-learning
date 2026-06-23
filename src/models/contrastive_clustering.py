"""Contrastive Clustering — configurable net + two-view (HTTP-text) trainer.

Key change vs. the original: the two contrastive views are produced by HTTP-level
augmentations on the raw headers/request text (see
:mod:`src.data.http_augmentations`) and then pushed through the SAME fitted
:class:`~src.data.preprocessing.Preprocessor` (transform-only — no leakage),
instead of perturbing the SVD vector directly.

Everything is configurable via :class:`CCConfig`: embedding/projection dims,
num_clusters (2/4/8/16), loss weights, temperatures, optimiser, augmentation
type. The trainer returns embeddings, cluster probabilities and assignments for
both splits so the three evaluation axes (clustering / representation / noise
auditing) can all be computed downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.http_augmentations import augment_headers_text, augment_request_text, get_config
from src.data.preprocessing import Preprocessor
from src.losses.contrastive_losses import ClusterLoss, InstanceLoss
from src.utils.seeds import new_rng


@dataclass
class CCConfig:
    embedding_dim: int = 64
    projection_dim: int = 16
    num_clusters: int = 2
    lambda_instance: float = 1.0
    lambda_cluster: float = 1.0
    lambda_entropy: float = 1.0
    temperature_instance: float = 0.5
    temperature_cluster: float = 1.0
    batch_size: int = 256
    epochs: int = 30
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    augmentation_type: str = "medium_http_aug"
    seed: int = 42

    def to_dict(self):
        return asdict(self)


class ContrastiveClusteringNet(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 64,
                 projection_dim: int = 16, num_clusters: int = 2):
        super().__init__()
        from src.models.encoder import TrafficEncoder

        self.encoder = TrafficEncoder(input_dim, embed_dim=embedding_dim)
        self.instance_head = nn.Sequential(
            nn.Linear(embedding_dim, 32), nn.ReLU(), nn.Linear(32, projection_dim)
        )
        self.cluster_head = nn.Sequential(
            nn.Linear(embedding_dim, 32), nn.ReLU(),
            nn.Linear(32, num_clusters), nn.Softmax(dim=1)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = F.normalize(self.instance_head(h), dim=1)
        p = self.cluster_head(h)
        return z, p

    @torch.no_grad()
    def embed(self, x):
        """Backbone embedding (the representation used for probes / KNN)."""
        return self.encoder(x)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_cc(h_tr: Sequence[str], r_tr: Sequence[str], pre: Preprocessor,
             cfg: CCConfig, h_te=None, r_te=None, verbose: bool = False):
    """Train CC with two HTTP-text views. Returns a dict of arrays + the model.

    Parameters
    ----------
    h_tr, r_tr : raw training headers/request strings (NOT yet vectorised)
    pre        : a Preprocessor ALREADY fitted on the training split
    cfg        : CCConfig
    h_te, r_te : optional raw test strings for test-side outputs
    """
    device = _device()
    rng = new_rng(cfg.seed)
    aug = get_config(cfg.augmentation_type)

    Xtr_clean = pre.transform(h_tr, r_tr)        # for embeddings/assignments
    input_dim = Xtr_clean.shape[1]
    n = len(h_tr)

    model = ContrastiveClusteringNet(input_dim, cfg.embedding_dim,
                                     cfg.projection_dim, cfg.num_clusters).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)
    ins = InstanceLoss(temperature=cfg.temperature_instance).to(device)
    clu = ClusterLoss(temperature=cfg.temperature_cluster,
                      entropy_weight=cfg.lambda_entropy).to(device)

    h_tr = list(h_tr); r_tr = list(r_tr)
    model.train()
    for epoch in range(cfg.epochs):
        perm = rng.permutation(n)
        total = 0.0
        for i in range(0, n - cfg.batch_size + 1, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            hb = [h_tr[j] for j in idx]
            rb = [r_tr[j] for j in idx]
            # two HTTP-text views, transformed through the FITTED preprocessor
            h1 = [augment_headers_text(x, rng, aug) for x in hb]
            r1 = [augment_request_text(x, rng, aug) for x in rb]
            h2 = [augment_headers_text(x, rng, aug) for x in hb]
            r2 = [augment_request_text(x, rng, aug) for x in rb]
            x1 = torch.tensor(pre.transform(h1, r1), device=device)
            x2 = torch.tensor(pre.transform(h2, r2), device=device)

            z1, p1 = model(x1)
            z2, p2 = model(x2)
            loss = cfg.lambda_instance * ins(z1, z2) + cfg.lambda_cluster * clu(p1, p2)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach())
        if verbose:
            print(f"  epoch {epoch + 1:>3}/{cfg.epochs}  loss={total:.3f}")

    model.eval()
    out = {"model": model, "config": cfg.to_dict()}
    with torch.no_grad():
        xt = torch.tensor(Xtr_clean, device=device)
        z_tr = model.encoder(xt).cpu().numpy()
        _, p_tr = model(xt)
        p_tr = p_tr.cpu().numpy()
        out["embeddings_train"] = z_tr
        out["cluster_probs_train"] = p_tr
        out["cluster_assignments_train"] = p_tr.argmax(1)
        if h_te is not None:
            Xte = pre.transform(h_te, r_te)
            xte = torch.tensor(Xte, device=device)
            z_te = model.encoder(xte).cpu().numpy()
            _, p_te = model(xte)
            p_te = p_te.cpu().numpy()
            out["embeddings_test"] = z_te
            out["cluster_probs_test"] = p_te
            out["cluster_assignments_test"] = p_te.argmax(1)
    return out


def view_instability(h_raw: Sequence[str], r_raw: Sequence[str], pre: Preprocessor,
                     model: ContrastiveClusteringNet, aug_type: str,
                     n_views: int = 5, seed: int = 0) -> np.ndarray:
    """Per-sample instability: mean variance of cluster-prob vectors across
    ``n_views`` augmented views. High = the sample's cluster assignment is not
    robust to plausible HTTP variations (a noise-suspicion signal)."""
    device = _device()
    rng = new_rng(seed)
    aug = get_config(aug_type)
    probs = []
    model.eval()
    with torch.no_grad():
        for _ in range(n_views):
            h = [augment_headers_text(x, rng, aug) for x in h_raw]
            r = [augment_request_text(x, rng, aug) for x in r_raw]
            x = torch.tensor(pre.transform(h, r), device=device)
            _, p = model(x)
            probs.append(p.cpu().numpy())
    stacked = np.stack(probs, axis=0)          # (views, n, clusters)
    return stacked.var(axis=0).mean(axis=1)    # (n,)
