"""Contrastive losses (instance-level NT-Xent + cluster-level with entropy)."""
from .contrastive_losses import InstanceLoss, ClusterLoss

__all__ = ["InstanceLoss", "ClusterLoss"]
