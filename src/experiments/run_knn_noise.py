"""KNN noise-suspicion — raw features vs Contrastive-Clustering embedding.

Directly answers: does KNN on the CC embedding detect synthetic noise better
than KNN on the raw features? We inject known noise, score with both, and
compare ROC-AUC / precision@k against the noise mask, over multiple seeds.

Saves results/noise_audit/knn_comparison.{json,csv}.

Usage:
    python -m src.experiments.run_knn_noise --config configs/noise_audit.yaml
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from src.evaluation.noise_detection import noise_detection_metrics
from src.models.contrastive_clustering import CCConfig, train_cc
from src.noise.knn_noise import knn_label_disagreement_score, knn_weighted_disagreement_score
from src.noise.synthetic_noise import inject_synthetic_label_noise
from src.utils.config import Config
from src.utils.io import save_csv, save_json
from src.utils.seeds import set_seed
from ._common import add_common_args, make_split, out_dir, resolve_config, log

DEFAULTS = {
    "source": "outbrain", "test_size": 0.3, "svd_dim": 64,
    "seeds": [42, 7, 123], "noise_rate": 0.10, "noise_mode": "symmetric",
    "cc_epochs": 20, "cc_num_clusters": 4, "cc_aug": "medium_http_aug",
    "knn_k": 20, "ks": [50, 100, 200],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--cc-epochs", type=int, default=None)
    ap.add_argument("--noise-rate", type=float, default=None)
    args = ap.parse_args()
    cfg = resolve_config(args, DEFAULTS)
    if args.cc_epochs is not None:
        cfg["cc_epochs"] = args.cc_epochs
    if args.noise_rate is not None:
        cfg["noise_rate"] = args.noise_rate

    seeds = cfg.get("seeds", [42])
    rate = float(cfg.get("noise_rate", 0.10))
    ks = cfg.get("ks", [50, 100, 200])
    k = int(cfg.get("knn_k", 20))
    rows = []

    for seed in seeds:
        set_seed(seed)
        sp = make_split(Config({**cfg, "seed": seed}))
        cc_cfg = CCConfig(num_clusters=int(cfg.get("cc_num_clusters", 4)),
                          augmentation_type=cfg.get("cc_aug", "medium_http_aug"),
                          epochs=int(cfg.get("cc_epochs", 20)), seed=seed)
        cc = train_cc(sp.h_tr, sp.r_tr, sp.pre, cc_cfg, verbose=False)
        emb = cc["embeddings_train"]

        y_noisy, mask = inject_synthetic_label_noise(sp.y_tr, rate,
                                                     mode=cfg.get("noise_mode", "symmetric"),
                                                     random_state=seed)
        variants = {
            "knn_raw": knn_label_disagreement_score(sp.Xtr, y_noisy, k=k),
            "knn_cc": knn_label_disagreement_score(emb, y_noisy, k=k),
            "knn_raw_weighted": knn_weighted_disagreement_score(sp.Xtr, y_noisy, k=k),
            "knn_cc_weighted": knn_weighted_disagreement_score(emb, y_noisy, k=k),
        }
        for name, s in variants.items():
            m = noise_detection_metrics(s, mask, ks=ks)
            row = {"variant": name, "seed": seed, "noise_rate": rate,
                   "roc_auc": round(m["roc_auc"], 4), "auprc": round(m["auprc"], 4)}
            for kk in ks:
                row[f"precision@{kk}"] = round(m[f"precision@{kk}"], 4)
            rows.append(row)
        log.info(f"[seed {seed}] raw auc={noise_detection_metrics(variants['knn_raw'], mask, ks)['roc_auc']:.3f} "
                 f"cc auc={noise_detection_metrics(variants['knn_cc'], mask, ks)['roc_auc']:.3f}")

    # aggregate
    agg = {}
    for name in sorted({r["variant"] for r in rows}):
        sel = [r for r in rows if r["variant"] == name]
        agg[name] = {
            "roc_auc_mean": round(float(np.nanmean([r["roc_auc"] for r in sel])), 4),
            "roc_auc_std": round(float(np.nanstd([r["roc_auc"] for r in sel])), 4),
            "auprc_mean": round(float(np.nanmean([r["auprc"] for r in sel])), 4),
        }

    base = out_dir(cfg, "noise_audit")
    save_json({"config": dict(cfg), "per_seed": rows, "aggregate": agg},
              os.path.join(base, "knn_comparison.json"))
    save_csv(rows, os.path.join(base, "knn_comparison.csv"))

    print(f"\n=== KNN noise detection @ rate={rate} (mean ROC-AUC over {len(seeds)} seeds) ===")
    for name, a in agg.items():
        print(f"  {name:<18} auc={a['roc_auc_mean']:.3f}±{a['roc_auc_std']:.3f}  "
              f"auprc={a['auprc_mean']:.3f}")
    print(f"\nSaved {base}/knn_comparison.json (+ .csv)")


if __name__ == "__main__":
    main()
