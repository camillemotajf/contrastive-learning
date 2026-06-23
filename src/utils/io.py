"""Small reproducible IO helpers (JSON / CSV / NPY)."""
from __future__ import annotations

import json
import os
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def ensure_dir(path: str) -> str:
    """Create ``path`` (a directory) if needed and return it."""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Any, path: str, indent: int = 2) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False, default=_default)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_npy(arr: np.ndarray, path: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    np.save(path, np.asarray(arr))


def save_csv(rows: Sequence[Mapping[str, Any]], path: str,
             columns: Iterable[str] | None = None) -> None:
    """Write a list of dict rows to CSV without pulling in pandas."""
    import csv

    rows = list(rows)
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    if not rows:
        # still write a header if columns were given
        cols = list(columns) if columns else []
        with open(path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(cols)
        return
    cols = list(columns) if columns else list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _default(o: Any):
    """JSON fallback for numpy scalars/arrays."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serializable: {type(o)}")
