"""Generate the main figures from saved results into results/figures/.

Gracefully skips any figure whose input JSON/CSV is missing, so it can be run
after any subset of experiments. Uses matplotlib only; embeddings are projected
with PCA 2D (no UMAP dependency).

Usage:
    python -m src.experiments.make_figures
"""
from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.utils.io import load_json  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)


def _try(fn):
    try:
        fn()
    except FileNotFoundError as e:
        print(f"  skip ({e})")
    except Exception as e:  # pragma: no cover
        print(f"  skip ({type(e).__name__}: {e})")


def fig_classification():
    data = load_json(os.path.join(RES, "baseline", "metrics.json"))["aggregate"]
    methods = ["logreg", "rf", "triplet_probe", "ssl_probe"]
    f1 = [data[m]["f1"]["mean"] for m in methods]
    auc = [data[m]["auc"]["mean"] for m in methods]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, f1, 0.4, label="F1")
    ax.bar(x + 0.2, auc, 0.4, label="ROC-AUC")
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=20)
    ax.set_ylim(0, 1); ax.set_title("Classification — F1 / ROC-AUC"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_classification.pdf")); plt.close(fig)
    print("  fig1_classification.pdf")


def fig_clustering():
    data = load_json(os.path.join(RES, "baseline", "metrics.json"))["aggregate"]
    methods = [m for m in ("raw_kmeans", "pca_kmeans") if m in data]
    metrics = ["ari", "nmi", "cluster_acc"]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, met in enumerate(metrics):
        ax.bar(x + (i - 1) * 0.25, [data[m][met]["mean"] for m in methods], 0.25, label=met)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_title("Clustering — ARI / NMI / accuracy"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_clustering.pdf")); plt.close(fig)
    print("  fig2_clustering.pdf")


def fig_noise_roc():
    agg_path = os.path.join(RES, "synthetic_noise", "summary_aggregate.csv")
    rows = _read_csv(agg_path)
    rates = sorted({float(r["noise_rate"]) for r in rows})
    scores = sorted({r["score"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 5))
    for sc in scores:
        ys = [next((float(r["roc_auc_mean"]) for r in rows
                    if r["score"] == sc and abs(float(r["noise_rate"]) - rt) < 1e-9), np.nan)
              for rt in rates]
        ax.plot(rates, ys, marker="o", label=sc)
    ax.set_xlabel("synthetic noise rate"); ax.set_ylabel("ROC-AUC of detection")
    ax.set_title("Noise detection — ROC-AUC by method"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_noise_roc.pdf")); plt.close(fig)
    print("  fig3_noise_roc.pdf")


def fig_precision_at_k():
    rows = _read_csv(os.path.join(RES, "synthetic_noise", "summary_aggregate.csv"))
    rate = 0.10
    sel = [r for r in rows if abs(float(r["noise_rate"]) - rate) < 1e-9]
    if not sel:
        sel = [r for r in rows if abs(float(r["noise_rate"]) - float(rows[0]["noise_rate"])) < 1e-9]
    sel = sorted(sel, key=lambda r: float(r["precision@100_mean"]), reverse=True)
    names = [r["score"] for r in sel]
    vals = [float(r["precision@100_mean"]) for r in sel]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names[::-1], vals[::-1])
    ax.set_xlabel("precision@100"); ax.set_title("Noise detection — precision@100")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_precision_at_k.pdf")); plt.close(fig)
    print("  fig4_precision_at_k.pdf")


def fig_score_distribution():
    rows = _read_csv(os.path.join(RES, "noise_audit", "all_noise_scores.csv"))
    ens = np.array([float(r["score_ensemble"]) for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ens, bins=50)
    ax.set_xlabel("ensemble suspicion score"); ax.set_ylabel("count")
    ax.set_title("Distribution of ensemble suspicion scores (real data)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_score_distribution.pdf")); plt.close(fig)
    print("  fig5_score_distribution.pdf")


def fig_embedding_pca():
    """PCA-2D of CC embeddings coloured by observed label (the run_noise_audit
    full-corpus embedding if present, else the first CC run)."""
    emb_path, lab = _find_embedding()
    if emb_path is None:
        raise FileNotFoundError("no CC embeddings_train.npy found")
    from sklearn.decomposition import PCA
    emb = np.load(emb_path)
    proj = PCA(n_components=2, random_state=0).fit_transform(emb)
    fig, ax = plt.subplots(figsize=(6, 5))
    if lab is not None and len(lab) == len(proj):
        for c in np.unique(lab):
            m = lab == c
            ax.scatter(proj[m, 0], proj[m, 1], s=4, alpha=0.4,
                       label=f"label {int(c)}")
        ax.legend()
    else:
        ax.scatter(proj[:, 0], proj[:, 1], s=4, alpha=0.4)
    ax.set_title("CC embeddings — PCA 2D (coloured by observed label)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_embedding_pca.pdf")); plt.close(fig)
    print("  fig6_embedding_pca.pdf")


def _find_embedding():
    # prefer a CC run directory
    ccroot = os.path.join(RES, "contrastive_clustering")
    if os.path.isdir(ccroot):
        for d in sorted(os.listdir(ccroot)):
            p = os.path.join(ccroot, d, "embeddings_train.npy")
            c = os.path.join(ccroot, d, "cluster_assignments_train.csv")
            if os.path.isfile(p):
                lab = None
                if os.path.isfile(c):
                    rows = _read_csv(c)
                    lab = np.array([int(r["observed_label"]) for r in rows])
                return p, lab
    return None, None


def _read_csv(path):
    import csv
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    print(f"Writing figures to {FIG}")
    for fn in (fig_classification, fig_clustering, fig_noise_roc,
               fig_precision_at_k, fig_score_distribution, fig_embedding_pca):
        _try(fn)


if __name__ == "__main__":
    main()
