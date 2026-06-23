"""Shared scaffolding for the experiment runners.

Centralises: config loading (YAML + CLI overrides), the leakage-free data split,
and fitting the Preprocessor on TRAIN only. Every runner goes through here so the
"split before fit" rule is enforced in exactly one place.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.data.loading import load_source, split_raw
from src.data.preprocessing import Preprocessor
from src.utils.config import Config, load_config
from src.utils.logging import get_logger

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

log = get_logger("exp")


@dataclass
class Split:
    h_tr: List[str]
    h_te: List[str]
    r_tr: List[str]
    r_te: List[str]
    y_tr: np.ndarray
    y_te: np.ndarray
    idx_tr: np.ndarray
    idx_te: np.ndarray
    pre: Preprocessor
    Xtr: np.ndarray
    Xte: np.ndarray


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--source", default=None, help="traffic source (e.g. outbrain)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--svd-dim", type=int, default=None)
    parser.add_argument("--subsample", type=int, default=None,
                        help="cap total samples (smoke tests)")
    parser.add_argument("--out", default=None, help="output directory")


def resolve_config(args: argparse.Namespace, defaults: dict) -> Config:
    """defaults < YAML file < explicit CLI args (CLI wins)."""
    cfg = dict(defaults)
    if getattr(args, "config", None):
        cfg.update(load_config(args.config))
    for key in ("source", "seed", "test_size", "svd_dim", "subsample", "out"):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    return Config(cfg)


def make_split(cfg: Config, fit_features: bool = True) -> Split:
    """Load -> stratified split on raw text -> fit Preprocessor on TRAIN only."""
    source = cfg.get("source", "outbrain")
    seed = int(cfg.get("seed", 42))
    test_size = float(cfg.get("test_size", 0.3))
    svd_dim = int(cfg.get("svd_dim", 64))
    subsample = cfg.get("subsample", None)

    h, r, y = load_source(source)
    if subsample:
        sel = np.random.RandomState(seed).choice(len(y), int(subsample), replace=False)
        h = [h[i] for i in sel]; r = [r[i] for i in sel]; y = y[sel]

    h_tr, h_te, r_tr, r_te, y_tr, y_te, idx_tr, idx_te = split_raw(
        h, r, y, test_size=test_size, seed=seed
    )
    pre = Preprocessor(svd_dim=svd_dim,
                       use_manual=bool(cfg.get("use_manual", True)),
                       canonicalize_headers=bool(cfg.get("canonicalize", False)),
                       seed=seed)
    Xtr = Xte = None
    if fit_features:
        pre.fit(h_tr, r_tr)
        Xtr = pre.transform(h_tr, r_tr)
        Xte = pre.transform(h_te, r_te)
    return Split(h_tr, h_te, r_tr, r_te, y_tr, y_te, idx_tr, idx_te, pre, Xtr, Xte)


def out_dir(cfg: Config, default_subdir: str) -> str:
    path = cfg.get("out") or os.path.join(RESULTS_DIR, default_subdir)
    os.makedirs(path, exist_ok=True)
    return path
