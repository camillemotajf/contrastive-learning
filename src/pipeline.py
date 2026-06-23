"""Backward-compatibility shim.

The pipeline moved to :mod:`src.data.loading` + :mod:`src.data.preprocessing`.
This module re-exports the original public names so existing experiments
(`from src.pipeline import build_features`, etc.) keep working unchanged.
"""
from src.data.loading import load_raw, split_raw  # noqa: F401
from src.data.preprocessing import (  # noqa: F401
    build_features,
    canonicalize as _canonicalize,
    manual_features as _manual_features,
)

__all__ = ["load_raw", "split_raw", "build_features", "_canonicalize", "_manual_features"]
