"""Non-contrastive baselines: KMeans, PCA+KMeans, LogReg, RandomForest."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def kmeans_predict(Xtr, Xte, n_clusters=2, seed=42):
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(Xtr)
    return km, km.predict(Xte)


def pca_kmeans_predict(Xtr, Xte, n_components=32, n_clusters=2, seed=42):
    pca = PCA(n_components=n_components, random_state=seed).fit(Xtr)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(pca.transform(Xtr))
    return km, km.predict(pca.transform(Xte))


def fit_logreg(Xtr, ytr, seed=42, max_iter=2000):
    return LogisticRegression(max_iter=max_iter, random_state=seed).fit(Xtr, ytr)


def fit_rf(Xtr, ytr, seed=42, n_estimators=200):
    return RandomForestClassifier(n_estimators=n_estimators, random_state=seed,
                                  n_jobs=-1).fit(Xtr, ytr)


def supervised_metrics(clf, Xte, yte) -> Dict[str, float]:
    from src.evaluation.classification import classification_metrics

    proba = clf.predict_proba(Xte)[:, 1]
    return classification_metrics(yte, proba)
