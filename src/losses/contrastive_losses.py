"""Contrastive Clustering losses (Li et al., 2021), with explicit entropy term.

InstanceLoss  — NT-Xent over the instance projections (rows = samples).
ClusterLoss   — NT-Xent over cluster columns + an entropy regulariser on the
                marginal cluster distribution that prevents the degenerate
                "everything in one cluster" collapse.

Both return scalar tensors and are unchanged in behaviour from the original
implementation — only relocated and documented.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.shape[0]
        z = torch.cat((z_i, z_j), dim=0)
        sim = torch.matmul(z, z.T) / self.temperature
        sim.fill_diagonal_(-1e9)
        positives = torch.cat(
            [torch.arange(batch_size, 2 * batch_size), torch.arange(batch_size)]
        ).to(z.device)
        return self.criterion(sim, positives)


class ClusterLoss(nn.Module):
    """Cluster-level contrast + entropy regulariser.

    ``entropy_weight`` scales ``-H(P)`` (added to the loss), so a larger weight
    pushes harder towards a balanced use of clusters. Returns the contrastive
    term and the entropy term separately is not needed by callers, so we return
    their sum; the experiment scripts log lambda weights via the trainer.
    """

    def __init__(self, temperature: float = 1.0, entropy_weight: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, p_i: torch.Tensor, p_j: torch.Tensor) -> torch.Tensor:
        eps = 1e-9
        p_i_mean = p_i.mean(dim=0)
        p_j_mean = p_j.mean(dim=0)
        ne = (p_i_mean * torch.log(p_i_mean + eps)).sum() \
            + (p_j_mean * torch.log(p_j_mean + eps)).sum()
        entropy_loss = self.entropy_weight * ne  # = -entropy_weight * (H(P_i)+H(P_j))

        pi = F.normalize(p_i.T, dim=1, p=2)
        pj = F.normalize(p_j.T, dim=1, p=2)
        k = pi.shape[0]
        p = torch.cat((pi, pj), dim=0)
        sim = torch.matmul(p, p.T) / self.temperature
        sim.fill_diagonal_(-1e9)
        positives = torch.cat([torch.arange(k, 2 * k), torch.arange(k)]).to(p.device)
        return self.criterion(sim, positives) + entropy_loss
