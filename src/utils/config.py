"""YAML config loading with attribute access and sensible defaults."""
from __future__ import annotations

import os
from typing import Any, Dict


class Config(dict):
    """A dict that also supports attribute access (``cfg.epochs``) and nested
    defaults via :meth:`get`. Unknown keys raise on attribute access so typos
    in a config surface early."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: str) -> Config:
    """Load a YAML file into a :class:`Config`."""
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        data: Dict[str, Any] = yaml.safe_load(fh) or {}
    return Config(data)


def merge(base: Dict[str, Any], override: Dict[str, Any]) -> Config:
    """Shallow-merge ``override`` onto ``base`` (override wins)."""
    out = dict(base)
    out.update({k: v for k, v in override.items() if v is not None})
    return Config(out)


def data_dir() -> str:
    """Absolute path to the dataset directory (../../data from project root)."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.abspath(os.path.join(root, "..", "..", "data"))
