"""Synthetic-noise evaluation — the only setting with a noise GROUND TRUTH.

We flip a known fraction of TRAIN labels, then score every sample with each
detector and grade the ranking against the known noise_mask (ROC-AUC, AUPRC,
precision@k, recall@k). This is what lets us claim a detector "works".

Scores evaluated:
  confident_learning, knn_raw, knn_cc, cc_cluster_entropy, cc_view_instability,
  cc_cluster_label_mismatch, centroid_own_distance, centroid_relative_distance,
  ensemble

Grid: noise_rates x seeds. CC (unsupervised) is trained once per seed and reused
across rates. Saves results/synthetic_noise/{summary_metrics.json,csv,
per_sample_scores.csv, top_suspects.csv}.

Usage:
    python -m src.experiments.run_synthetic_noise_eval --config configs/synthetic_noise.yaml
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from src.evaluation.noise_detection import noise_detection_metrics
from src.models.contrastive_clustering import CCConfig, train_cc, view_instability
from src.noise.confident_learning import confident_learning_scores, oof_probabilities
from src.noise.knn_noise import knn_label_disagreement_score
from src.noise.scoring import (
    centroid_distance_scores, cluster_entropy_score, cluster_label_mismatch_score,
    ensemble_score,
)
from src.noise.synthetic_noise import inject_synthetic_label_noise
from src.utils.config import Config
from src.utils.io import save_csv, save_json
from src.utils.seeds import set_seed
from ._common import add_common_args, make_split, out_dir, resolve_config, log

DEFAULTS = {
    "source": "outbrain", "test_size": 0.3, "svd_dim": 64,
    "noise_rates": [0.01, 0.03, 0.05, 0.10, 0.20],
    "seeds": [42, 7, 123, 2024, 999],
    "noise_mode": "symmetric",
    "cc_epochs": 20, "cc_num_clusters": 4, "cc_aug": "medium_http_aug",
    "knn_k": 20, "instability_views": 5, "ks": [50, 100, 200],
}

SCORE_NAMES = [
    "confident_learning", "knn_raw", "knn_cc", "cc_cluster_entropy",
    "cc_view_instability", "cc_cluster_label_mismatch",
    "centroid_own_distance", "centroid_relative_distance", "ensemble",
]


def compute_scores(sp, y_noisy, cc_out, cc_cfg, cfg):
    """All per-sample suspicion scores, computed from the NOISY labels."""
    X = sp.Xtr
    emb = cc_out["embeddings_train"]
    probs = cc_out["cluster_probs_train"]
    assign = cc_out["cluster_assignments_train"]
    k = int(cfg.get("knn_k", 20))

    proba = oof_probabilities(X, y_noisy, seed=cc_cfg.seed)
    cl_score, _, _ = confident_learning_scores(y_noisy, proba)
    cd = centroid_distance_scores(emb, y_noisy)

    scores = {
        "confident_learning": cl_score,
        "knn_raw": knn_label_disagreement_score(X, y_noisy, k=k),
        "knn_cc": knn_label_disagreement_score(emb, y_noisy, k=k),
        "cc_cluster_entropy": cluster_entropy_score(probs),
        "cc_view_instability": view_instability(
            sp.h_tr, sp.r_tr, sp.pre, cc_out["model"], cc_cfg.augmentation_type,
            n_views=int(cfg.get("instability_views", 5)), seed=cc_cfg.seed),
        "cc_cluster_label_mismatch": cluster_label_mismatch_score(assign, y_noisy),
        "centroid_own_distance": cd["own_distance"],
        "centroid_relative_distance": cd["relative_distance"],
    }
    scores["ensemble"] = ensemble_score([
        scores["confident_learning"], scores["knn_raw"], scores["knn_cc"],
        scores["cc_cluster_entropy"], scores["cc_view_instability"],
        scores["centroid_relative_distance"],
    ])
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--cc-epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = resolve_config(args, DEFAULTS)
    if args.cc_epochs is not None:
        cfg["cc_epochs"] = args.cc_epochs

    seeds = cfg.get("seeds", [42])
    rates = cfg.get("noise_rates", [0.05])
    ks = cfg.get("ks", [50, 100, 200])
    mode = cfg.get("noise_mode", "symmetric")

    summary = []                  # one row per (score, rate, seed)
    per_sample_dump = None        # saved for a representative (rate, seed)

    for seed in seeds:
        set_seed(seed)
        sp = make_split(Config({**cfg, "seed": seed}))
        cc_cfg = CCConfig(num_clusters=int(cfg.get("cc_num_clusters", 4)),
                          augmentation_type=cfg.get("cc_aug", "medium_http_aug"),
                          epochs=int(cfg.get("cc_epochs", 20)), seed=seed)
        log.info(f"[seed {seed}] training CC once (reused across rates)")
        cc_out = train_cc(sp.h_tr, sp.r_tr, sp.pre, cc_cfg, verbose=False)

        for rate in rates:
            y_noisy, mask = inject_synthetic_label_noise(
                sp.y_tr, rate, mode=mode, random_state=seed)
            scores = compute_scores(sp, y_noisy, cc_out, cc_cfg, cfg)
            for name, s in scores.items():
                m = noise_detection_metrics(s, mask, ks=ks)
                row = {"score": name, "noise_rate": rate, "seed": seed,
                       "roc_auc": m["roc_auc"], "auprc": m["auprc"]}
                for k in ks:
                    row[f"precision@{k}"] = m[f"precision@{k}"]
                    row[f"recall@{k}"] = m[f"recall@{k}"]
                summary.append(row)
            log.info(f"  rate={rate:<5} flips={int(mask.sum())} "
                     f"| CL auc={noise_detection_metrics(scores['confident_learning'], mask, ks)['roc_auc']:.3f} "
                     f"knn_cc auc={noise_detection_metrics(scores['knn_cc'], mask, ks)['roc_auc']:.3f} "
                     f"ens auc={noise_detection_metrics(scores['ensemble'], mask, ks)['roc_auc']:.3f}")

            if per_sample_dump is None and abs(rate - 0.10) < 1e-9:
                per_sample_dump = _per_sample(sp, y_noisy, mask, scores)

    base = out_dir(cfg, "synthetic_noise")
    save_json({"config": dict(cfg), "rows": summary}, os.path.join(base, "summary_metrics.json"))
    save_csv(summary, os.path.join(base, "summary_metrics.csv"))
    _save_aggregate(summary, base, rates, ks)
    if per_sample_dump is not None:
        save_csv(per_sample_dump, os.path.join(base, "per_sample_scores.csv"))
        top = sorted(per_sample_dump, key=lambda r: r["ensemble"], reverse=True)[:200]
        save_csv(top, os.path.join(base, "top_suspects.csv"))
    print(f"\nSaved synthetic-noise results under {base}")


def _per_sample(sp, y_noisy, mask, scores):
    rows = []
    for i in range(len(y_noisy)):
        row = {"index": int(sp.idx_tr[i]), "observed_label": int(y_noisy[i]),
               "is_synthetic_noise": bool(mask[i])}
        for name, s in scores.items():
            row[name] = float(s[i])
        rows.append(row)
    return rows


def _save_aggregate(summary, base, rates, ks):
    """Mean ROC-AUC per (score, rate) across seeds — the headline table."""
    import numpy as np
    agg = []
    scores = sorted({r["score"] for r in summary})
    for name in scores:
        for rate in rates:
            sel = [r for r in summary if r["score"] == name and r["noise_rate"] == rate]
            if not sel:
                continue
            agg.append({
                "score": name, "noise_rate": rate,
                "roc_auc_mean": round(float(np.nanmean([r["roc_auc"] for r in sel])), 4),
                "roc_auc_std": round(float(np.nanstd([r["roc_auc"] for r in sel])), 4),
                "auprc_mean": round(float(np.nanmean([r["auprc"] for r in sel])), 4),
                "precision@100_mean": round(float(np.nanmean([r["precision@100"] for r in sel])), 4),
            })
    save_csv(agg, os.path.join(base, "summary_aggregate.csv"))


if __name__ == "__main__":
    main()
