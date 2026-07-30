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
    """Load one source's unsafe + bot dumps.

    Supports both the older flat layout::

        data/outbrain-unsafe-2026-07.json
        data/outbrain-bot-2026-07.json

    and the traffic-source export layout::

        data/raw/outbrain/outbrain-unsafe.json
        data/raw/outbrain/outbrain-bot.json
    """
    directories = [directory] if directory else _default_data_dirs()
    unsafe = _first(_source_patterns(directories, source, "unsafe"))
    bots = _first(_source_patterns(directories, source, "bot"))
    return load_raw(unsafe, bots)


def list_sources(directory: str | None = None) -> List[str]:
    """Traffic sources that have both a bot and an unsafe dump on disk."""
    directories = [directory] if directory else _default_data_dirs()
    bots = _sources_with_group(directories, "bot")
    unsafe = _sources_with_group(directories, "unsafe")
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


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _default_data_dirs() -> List[str]:
    """Search project-local exports first, then the legacy shared data dir."""
    dirs = [
        os.path.join(_project_root(), "data", "raw"),
        data_dir(),
    ]
    out: List[str] = []
    for path in dirs:
        if path not in out:
            out.append(path)
    return out


def _source_patterns(directories: List[str], source: str, group: str) -> List[str]:
    return [
        pattern
        for directory in directories
        for pattern in (
            os.path.join(directory, f"{source}-{group}.json"),
            os.path.join(directory, f"{source}-{group}-*.json"),
            os.path.join(directory, source, f"{source}-{group}.json"),
            os.path.join(directory, source, f"{source}-{group}-*.json"),
        )
    ]


def _sources_with_group(directories: List[str], group: str) -> set[str]:
    sources = set()
    for directory in directories:
        flat_patterns = [
            os.path.join(directory, f"*-{group}.json"),
            os.path.join(directory, f"*-{group}-*.json"),
        ]
        nested_patterns = [
            os.path.join(directory, "*", f"*-{group}.json"),
            os.path.join(directory, "*", f"*-{group}-*.json"),
        ]
        for pattern in flat_patterns + nested_patterns:
            for path in glob.glob(pattern):
                name = os.path.basename(path)
                marker = f"-{group}"
                if marker in name:
                    sources.add(name.split(marker)[0])
    return sources


def _first(patterns: List[str]) -> str:
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No file matches any of: {patterns}")
