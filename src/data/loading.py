"""Raw data loading and the (label-blind) train/test split.

label 0 = unsafe (human), label 1 = bots — same convention as the original
``src/datasets.py`` / ``src/pipeline.py`` so existing experiments stay valid.

The split happens here, on the raw text, BEFORE any learned transform is fit —
this is the single most important guard against data leakage.
"""
from __future__ import annotations

import glob
import json
import os
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.config import data_dir


def load_raw(file_unsafe: str, file_bots: str) -> Tuple[List[str], List[str], np.ndarray]:
    """Read two JSON dumps into parallel lists of raw strings + labels."""
    headers: List[str] = []
    requests: List[str] = []
    labels: List[int] = []

    def _load(path: str, label: int) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for item in data:
            headers.append(item.get("headers", "{}"))
            requests.append(item.get("request", "{}"))
            labels.append(label)

    _load(file_unsafe, 0)
    _load(file_bots, 1)
    return headers, requests, np.array(labels, dtype=np.int64)


def load_source(source: str = "outbrain", directory: str | None = None):
    """Load ``{source}-unsafe-*.json`` + ``{source}-bot-*.json`` by name."""
    directory = directory or data_dir()
    unsafe = _first(os.path.join(directory, f"{source}-unsafe-*.json"))
    bots = _first(os.path.join(directory, f"{source}-bot-*.json"))
    return load_raw(unsafe, bots)


def list_sources(directory: str | None = None) -> List[str]:
    """Traffic sources that have both a bot and an unsafe dump on disk."""
    directory = directory or data_dir()
    bots = {os.path.basename(p).split("-bot-")[0]
            for p in glob.glob(os.path.join(directory, "*-bot-*.json"))}
    unsafe = {os.path.basename(p).split("-unsafe-")[0]
              for p in glob.glob(os.path.join(directory, "*-unsafe-*.json"))}
    return sorted(bots & unsafe)


def split_raw(headers, requests, labels, test_size=0.3, seed=42):
    """Stratified split on raw text. Returns six parallel containers:
    (h_tr, h_te, r_tr, r_te, y_tr, y_te) plus the train/test index arrays."""
    idx = np.arange(len(labels))
    idx_tr, idx_te = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=labels
    )

    def sub(lst, ids):
        return [lst[i] for i in ids]

    return (
        sub(headers, idx_tr), sub(headers, idx_te),
        sub(requests, idx_tr), sub(requests, idx_te),
        labels[idx_tr], labels[idx_te],
        idx_tr, idx_te,
    )


def _first(pattern: str) -> str:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matches {pattern}")
    return matches[0]
