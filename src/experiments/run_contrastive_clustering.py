"""Train the improved Contrastive Clustering and evaluate it on all three axes.

  A. clustering quality   — ARI/NMI/Hungarian acc + cluster size/entropy
  B. representation        — logistic probe on frozen CC embeddings (F1/AUC/...)
  C. noise-audit utility   — per-sample suspicion scores (saved for later use)

Saves, under results/contrastive_clustering/<run>/:
  metrics.json, config.json,
  embeddings_train.npy, embeddings_test.npy,
  cluster_probs_train.npy, cluster_probs_test.npy,
  cluster_assignments_train.csv, cluster_assignments_test.csv

Usage:
    python -m src.experiments.run_contrastive_clustering --config configs/contrastive_clustering.yaml
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from src.evaluation.classification import linear_probe_metrics
from src.evaluation.clustering import clustering_metrics
from src.models.contrastive_clustering import CCConfig, train_cc
from src.utils.config import Config
from src.utils.io import save_csv, save_json, save_npy
from src.utils.seeds import set_seed
from ._common import add_common_args, make_split, out_dir, resolve_config, log

DEFAULTS = {
    "source": "outbrain", "seed": 42, "test_size": 0.3, "svd_dim": 64,
    "embedding_dim": 64, "projection_dim": 16, "num_clusters": 2,
    "lambda_instance": 1.0, "lambda_cluster": 1.0, "lambda_entropy": 1.0,
    "temperature_instance": 0.5, "temperature_cluster": 1.0,
    "batch_size": 256, "epochs": 30, "learning_rate": 5e-4, "weight_decay": 0.0,
    "augmentation_type": "medium_http_aug", "run_name": None,
}


def cc_config_from(cfg: Config) -> CCConfig:
    return CCConfig(
        embedding_dim=int(cfg.get("embedding_dim", 64)),
        projection_dim=int(cfg.get("projection_dim", 16)),
        num_clusters=int(cfg.get("num_clusters", 2)),
        lambda_instance=float(cfg.get("lambda_instance", 1.0)),
        lambda_cluster=float(cfg.get("lambda_cluster", 1.0)),
        lambda_entropy=float(cfg.get("lambda_entropy", 1.0)),
        temperature_instance=float(cfg.get("temperature_instance", 0.5)),
        temperature_cluster=float(cfg.get("temperature_cluster", 1.0)),
        batch_size=int(cfg.get("batch_size", 256)),
        epochs=int(cfg.get("epochs", 30)),
        learning_rate=float(cfg.get("learning_rate", 5e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        augmentation_type=cfg.get("augmentation_type", "medium_http_aug"),
        seed=int(cfg.get("seed", 42)),
    )


def evaluate_and_save(sp, out, cfg, base):
    cc = cc_config_from(cfg)
    metrics = {
        "config": cc.to_dict(),
        "clustering_train": clustering_metrics(sp.y_tr, out["cluster_assignments_train"]),
        "representation_probe": linear_probe_metrics(
            out["embeddings_train"], sp.y_tr, out["embeddings_test"], sp.y_te, cc.seed),
    }
    if "cluster_assignments_test" in out:
        metrics["clustering_test"] = clustering_metrics(sp.y_te, out["cluster_assignments_test"])

    save_json(metrics, os.path.join(base, "metrics.json"))
    save_json(cc.to_dict(), os.path.join(base, "config.json"))
    save_npy(out["embeddings_train"], os.path.join(base, "embeddings_train.npy"))
    save_npy(out["cluster_probs_train"], os.path.join(base, "cluster_probs_train.npy"))
    save_csv([{"index": int(sp.idx_tr[i]), "observed_label": int(sp.y_tr[i]),
               "cluster": int(out["cluster_assignments_train"][i])}
              for i in range(len(sp.y_tr))],
             os.path.join(base, "cluster_assignments_train.csv"))
    if "embeddings_test" in out:
        save_npy(out["embeddings_test"], os.path.join(base, "embeddings_test.npy"))
        save_npy(out["cluster_probs_test"], os.path.join(base, "cluster_probs_test.npy"))
        save_csv([{"index": int(sp.idx_te[i]), "observed_label": int(sp.y_te[i]),
                   "cluster": int(out["cluster_assignments_test"][i])}
                  for i in range(len(sp.y_te))],
                 os.path.join(base, "cluster_assignments_test.csv"))
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--num-clusters", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--augmentation-type", default=None)
    args = ap.parse_args()
    cfg = resolve_config(args, DEFAULTS)
    for k in ("num_clusters", "epochs", "augmentation_type"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    set_seed(int(cfg.get("seed", 42)))
    sp = make_split(cfg)
    cc = cc_config_from(cfg)
    run_name = cfg.get("run_name") or f"k{cc.num_clusters}_{cc.augmentation_type}_s{cc.seed}"
    base = os.path.join(out_dir(cfg, "contrastive_clustering"), run_name)
    os.makedirs(base, exist_ok=True)

    log.info(f"Training CC: clusters={cc.num_clusters} aug={cc.augmentation_type} "
             f"epochs={cc.epochs} -> {base}")
    out = train_cc(sp.h_tr, sp.r_tr, sp.pre, cc, h_te=sp.h_te, r_te=sp.r_te, verbose=True)
    metrics = evaluate_and_save(sp, out, cfg, base)

    print("\n=== Contrastive Clustering ===")
    ctr = metrics["clustering_train"]
    print(f"  clustering(train): ari={ctr['ari']:.3f} nmi={ctr['nmi']:.3f} "
          f"acc={ctr['cluster_acc']:.3f} | cluster sizes={ctr['distribution']['sizes']}")
    pr = metrics["representation_probe"]
    print(f"  probe(test):       f1={pr['f1']:.3f} auc={pr['auc']:.3f} acc={pr['acc']:.3f}")
    print(f"\nSaved artifacts under {base}")


if __name__ == "__main__":
    main()
