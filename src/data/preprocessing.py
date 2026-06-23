"""Leakage-free feature pipeline (TF-IDF char n-grams -> SVD -> z-score).

The :class:`Preprocessor` encapsulates the fit/transform discipline:

  * ``fit`` is called ONLY on the training split.
  * ``transform`` is the only thing the test split (and any augmented view) ever
    sees — it reuses the vectoriser/SVD/scaler fitted on train.

This is what keeps the two-view Contrastive Clustering honest: augmented views
are *transformed* through the already-fitted pipeline, never refitted.

``build_features`` is kept as a thin function wrapper for backward compatibility
with the existing experiments (same signature/behaviour as the old
``src/pipeline.build_features``).
"""
from __future__ import annotations

import json
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .loading import load_raw, split_raw


def canonicalize(headers_str: str) -> str:
    """Stable re-serialisation (sorted keys) to neutralise ordering artefacts."""
    try:
        d = json.loads(headers_str)
        return json.dumps(d, sort_keys=True, separators=(",", ":"))
    except Exception:
        return headers_str


def manual_features(headers: Sequence[str], requests: Sequence[str]) -> np.ndarray:
    return np.array(
        [[len(h), len(r), h.count(":"), r.count(":")] for h, r in zip(headers, requests)],
        dtype=np.float32,
    )


class Preprocessor:
    """Fit-on-train, transform-everything-else feature extractor."""

    def __init__(self, svd_dim: int = 64, use_manual: bool = True,
                 canonicalize_headers: bool = False, max_features: int = 5000,
                 ngram_range: Tuple[int, int] = (2, 4), seed: int = 42):
        self.svd_dim = svd_dim
        self.use_manual = use_manual
        self.canonicalize_headers = canonicalize_headers
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.seed = seed
        self._vec: Optional[TfidfVectorizer] = None
        self._svd: Optional[TruncatedSVD] = None
        self._mean = None
        self._std = None
        self._fitted = False

    # -- internals -------------------------------------------------------- #
    def _text(self, headers: Sequence[str], requests: Sequence[str]) -> List[str]:
        if self.canonicalize_headers:
            headers = [canonicalize(h) for h in headers]
        # the vectoriser sees headers; manual/request signal added separately
        return list(headers)

    # -- fit / transform -------------------------------------------------- #
    def fit(self, headers: Sequence[str], requests: Sequence[str]) -> "Preprocessor":
        text = self._text(headers, requests)
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=self.ngram_range,
                                    max_features=self.max_features)
        tfidf = self._vec.fit_transform(text)
        self._svd = TruncatedSVD(n_components=self.svd_dim, random_state=self.seed)
        X = self._svd.fit_transform(tfidf)
        if self.use_manual:
            X = np.hstack([X, manual_features(self._maybe_canon(headers), requests)])
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-8
        self._fitted = True
        return self

    def transform(self, headers: Sequence[str], requests: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Preprocessor.transform called before fit")
        text = self._text(headers, requests)
        X = self._svd.transform(self._vec.transform(text))
        if self.use_manual:
            X = np.hstack([X, manual_features(self._maybe_canon(headers), requests)])
        return ((X - self._mean) / self._std).astype(np.float32)

    def fit_transform(self, headers, requests) -> np.ndarray:
        return self.fit(headers, requests).transform(headers, requests)

    def _maybe_canon(self, headers):
        return [canonicalize(h) for h in headers] if self.canonicalize_headers else headers


def build_features(file_unsafe, file_bots, test_size=0.3, seed=42, svd_dim=64,
                   use_manual=True, canonicalize=False):
    """Backward-compatible wrapper: returns Xtr, Xte, ytr, yte (no leakage).

    Mirrors the original ``src.pipeline.build_features`` exactly so existing
    experiments keep working.
    """
    headers, requests, labels = load_raw(file_unsafe, file_bots)
    h_tr, h_te, r_tr, r_te, y_tr, y_te, _, _ = split_raw(
        headers, requests, labels, test_size=test_size, seed=seed
    )
    pre = Preprocessor(svd_dim=svd_dim, use_manual=use_manual,
                       canonicalize_headers=canonicalize, seed=seed).fit(h_tr, r_tr)
    return pre.transform(h_tr, r_tr), pre.transform(h_te, r_te), y_tr, y_te
