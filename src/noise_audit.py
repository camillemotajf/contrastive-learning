"""Label-noise auditing via Confident Learning + high-precision heuristics.

The labels come from the `decision` field of an existing system, so they are
"mostly correct" but not guaranteed. This module:

  1. Estimates the label-noise rate with Confident Learning (Northcutt et al.,
     2021): out-of-fold predicted probabilities + per-class confidence
     thresholds (the "confident joint") flag likely mislabeled samples.
  2. Ranks the most suspicious samples so they can be hand-verified to build a
     small "gold" test set.
  3. Derives high-precision heuristic anchors (unsubstituted `{{...}}` request
     templates, known crawler User-Agents) to cross-check the flags.

Feature extraction here is label-free (TF-IDF + SVD), so fitting it on the full
corpus introduces no label leakage; the classifier is isolated per fold via
cross_val_predict.
"""
import json
import ast
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from src.pipeline import load_raw, _manual_features


# --------------------------------------------------------------------------- #
# Features (label-free) over the full corpus
# --------------------------------------------------------------------------- #
def build_full_features(headers, requests, seed=42, svd_dim=64):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=5000)
    X = svd = TruncatedSVD(n_components=svd_dim, random_state=seed).fit_transform(
        vec.fit_transform(headers)
    )
    X = np.hstack([X, _manual_features(headers, requests)])
    mean, std = X.mean(0), X.std(0) + 1e-8
    return ((X - mean) / std).astype(np.float32)


# --------------------------------------------------------------------------- #
# Confident Learning
# --------------------------------------------------------------------------- #
def oof_probabilities(X, y, seed=42, n_splits=5):
    """Out-of-fold predicted probabilities from a strong classifier."""
    clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(clf, X, y, cv=cv, method="predict_proba", n_jobs=-1)


def confident_learning(y, proba):
    """Flag likely label errors using the confident-joint logic.

    Returns
    -------
    flags     : bool array, True = likely mislabeled
    suspicion : float array, P(other class) - P(given label); higher = more suspicious
    thresholds: per-class mean self-confidence
    """
    y = np.asarray(y)
    classes = np.unique(y)
    thresholds = np.array([proba[y == c, c].mean() for c in classes])

    # Confident class: among classes whose prob clears their threshold, the argmax.
    eligible = proba >= thresholds[None, :]
    masked = np.where(eligible, proba, -1.0)
    confident_class = masked.argmax(axis=1)
    has_conf = eligible.any(axis=1)

    flags = has_conf & (confident_class != y)
    given_conf = proba[np.arange(len(y)), y]
    other_conf = proba.max(axis=1)  # binary: max == other-class prob when flagged
    suspicion = other_conf - given_conf
    return flags, suspicion, thresholds


# --------------------------------------------------------------------------- #
# High-precision heuristic anchors
# --------------------------------------------------------------------------- #
_CRAWLER_UA = re.compile(
    r"facebookexternalhit|bot|crawler|spider|slurp|bingpreview|headless|python-requests|curl|wget",
    re.IGNORECASE,
)


def _parse_headers(h):
    try:
        return json.loads(h)
    except Exception:
        try:
            return ast.literal_eval(h)
        except Exception:
            return {}


def heuristic_labels(headers, requests):
    """High-precision weak labels: 1 = almost-certainly bot, 0 = almost-certainly
    human, -1 = no confident heuristic. Used to cross-check Confident Learning,
    NOT as model features."""
    out = np.full(len(headers), -1, dtype=np.int64)
    for i, (h, r) in enumerate(zip(headers, requests)):
        hd = _parse_headers(h)
        ua = ""
        if isinstance(hd, dict):
            ua = str(hd.get("User-Agent", hd.get("user-agent", "")))
        if "{{" in str(r) or _CRAWLER_UA.search(ua):
            out[i] = 1  # unsubstituted template or crawler UA -> bot
    return out
