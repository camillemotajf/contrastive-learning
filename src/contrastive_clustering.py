"""Backward-compatibility shim.

The model moved to :mod:`src.models.contrastive_clustering` and the losses to
:mod:`src.losses.contrastive_losses`. The constructor signature is unchanged for
legacy keyword calls: ``ContrastiveClusteringNet(input_dim=..., num_clusters=...)``.
"""
from src.models.contrastive_clustering import ContrastiveClusteringNet  # noqa: F401
from src.losses.contrastive_losses import InstanceLoss, ClusterLoss  # noqa: F401

__all__ = ["ContrastiveClusteringNet", "InstanceLoss", "ClusterLoss"]
