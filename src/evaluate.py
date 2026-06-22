"""Métricas de avaliação honestas para os experimentos.

- Clustering (não-supervisionado): ARI, NMI e acurácia de clustering com
  matching Húngaro contra os rótulos verdadeiros. NÃO usamos Silhouette como
  métrica principal, pois ele mede apenas coesão geométrica e é circular quando
  o modelo foi treinado com os próprios rótulos.
- Supervisionado: F1, ROC-AUC e acurácia de uma sonda treinada SOBRE os
  embeddings (linear probe), avaliada no conjunto de teste held-out.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
)
from sklearn.linear_model import LogisticRegression


def clustering_accuracy(y_true, y_pred):
    """Acurácia de clustering com matching ótimo (Húngaro) entre clusters e classes."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = max(y_true.max(), y_pred.max()) + 1
    cost = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cost[p, t] += 1
    row, col = linear_sum_assignment(-cost)
    mapping = {r: c for r, c in zip(row, col)}
    acc = sum(cost[r, mapping[r]] for r in row) / len(y_true)
    return acc


def clustering_metrics(y_true, y_pred):
    return {
        "ari": adjusted_rand_score(y_true, y_pred),
        "nmi": normalized_mutual_info_score(y_true, y_pred),
        "cluster_acc": clustering_accuracy(y_true, y_pred),
    }


def linear_probe_metrics(emb_tr, y_tr, emb_te, y_te, seed=42):
    """Treina uma regressão logística nos embeddings de treino e avalia no teste."""
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(emb_tr, y_tr)
    pred = clf.predict(emb_te)
    proba = clf.predict_proba(emb_te)[:, 1]
    return {
        "f1": f1_score(y_te, pred),
        "auc": roc_auc_score(y_te, proba),
        "acc": accuracy_score(y_te, pred),
    }
