"""Backward-compatibility shim — TrafficEncoder moved to :mod:`src.models.encoder`."""
from src.models.encoder import TrafficEncoder  # noqa: F401

__all__ = ["TrafficEncoder"]
