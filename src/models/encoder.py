"""Dense traffic encoder — MLP with L2-normalised output (hypersphere)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrafficEncoder(nn.Module):
    """input_dim -> 128 -> 64 -> embed_dim, L2-normalised.

    L2 normalisation places every embedding on the unit hypersphere, which is
    what makes cosine-similarity contrastive losses well-behaved.
    """

    def __init__(self, input_dim: int, embed_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return F.normalize(x, p=2, dim=1)
