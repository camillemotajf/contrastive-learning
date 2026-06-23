"""Baselines — Evaluation A (clustering) + Evaluation B (classification).

Clustering:     KMeans raw, PCA+KMeans            (ARI/NMI/Hungarian acc)
Classification: LogReg, RandomForest, Triplet+probe, SSL+probe
                (F1/AUC/acc/precision/recall on the held-out test set)

Aggregated over multiple seeds. Saves results/baseline/metrics.json.

Usage:
    python -m src.experiments.run_baselines --config configs/baseline.yaml
    python -m src.experiments.run_baselines --source outbrain --subsample 3000
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from src.evaluation.classification import classification_metrics, linear_probe_metrics
from src.evaluation.clustering import clustering_metrics
from src.models import baselines as bl
from src.models.ssl import ssl_pretrain, backbone_embed
from src.models.triplet import train_triplet
from src.utils.io import save_json
from src.utils.seeds import set_seed
from ._common import add_common_args, make_split, out_dir, resolve_config, log

DEFAULTS = {"source": "outbrain", "seeds": [42, 7, 123], "test_size": 0.3, "svd_dim": 64}


def run_once(cfg, seed):
    cfg = dict(cfg); cfg["seed"] = seed
    from src.utils.config import Config
    sp = make_split(Config(cfg))
    res = {}

    # -- Evaluation A: clustering --
    _, km = bl.kmeans_predict(sp.Xtr, sp.Xte, seed=seed)
    res["raw_kmeans"] = clustering_metrics(sp.y_te, km)
    _, pkm = bl.pca_kmeans_predict(sp.Xtr, sp.Xte, seed=seed)
    res["pca_kmeans"] = clustering_metrics(sp.y_te, pkm)

    # -- Evaluation B: classification --
    res["logreg"] = bl.supervised_metrics(bl.fit_logreg(sp.Xtr, sp.y_tr, seed), sp.Xte, sp.y_te)
    res["rf"] = bl.supervised_metrics(bl.fit_rf(sp.Xtr, sp.y_tr, seed), sp.Xte, sp.y_te)

    e_tr, e_te = train_triplet(sp.Xtr, sp.y_tr, sp.Xte)
    res["triplet_probe"] = linear_probe_metrics(e_tr, sp.y_tr, e_te, sp.y_te, seed)

    backbone, ssl_tr = ssl_pretrain(sp.Xtr)
    ssl_te = backbone_embed(backbone, sp.Xte)
    res["ssl_probe"] = linear_probe_metrics(ssl_tr, sp.y_tr, ssl_te, sp.y_te, seed)
    return res


def aggregate(runs):
    methods = runs[0].keys()
    agg = {}
    for m in methods:
        agg[m] = {}
        for metric in runs[0][m]:
            if isinstance(runs[0][m][metric], (int, float)):
                vals = [r[m][metric] for r in runs]
                agg[m][metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = resolve_config(args, DEFAULTS)
    seeds = cfg.get("seeds", [cfg.get("seed", 42)])
    if args.seed is not None:
        seeds = [args.seed]

    log.info(f"Baselines | source={cfg.get('source')} seeds={seeds}")
    runs = []
    for s in seeds:
        set_seed(s)
        runs.append(run_once(cfg, s))
        log.info(f"  seed {s}: RF f1={runs[-1]['rf']['f1']:.3f} "
                 f"Triplet f1={runs[-1]['triplet_probe']['f1']:.3f} "
                 f"SSL f1={runs[-1]['ssl_probe']['f1']:.3f}")

    agg = aggregate(runs)
    path = out_dir(cfg, "baseline")
    save_json({"config": dict(cfg), "seeds": seeds, "aggregate": agg, "per_seed": runs},
              f"{path}/metrics.json")

    print("\n=== Classification (mean F1 / AUC over seeds) ===")
    for m in ["logreg", "rf", "triplet_probe", "ssl_probe"]:
        print(f"  {m:<14} f1={agg[m]['f1']['mean']:.3f}  auc={agg[m]['auc']['mean']:.3f}")
    print("=== Clustering (mean over seeds) ===")
    for m in ["raw_kmeans", "pca_kmeans"]:
        print(f"  {m:<14} ari={agg[m]['ari']['mean']:.3f}  nmi={agg[m]['nmi']['mean']:.3f}  "
              f"acc={agg[m]['cluster_acc']['mean']:.3f}")
    print(f"\nSaved {path}/metrics.json")


if __name__ == "__main__":
    main()
