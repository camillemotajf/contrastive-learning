"""Real-data noise audit — apply the suspicion scores WITHOUT corrupting labels.

There is no ground truth here, so every output is a RANKING of *candidates for
review*, not a list of confirmed errors. We compute all scores, cross-check them
against high-precision heuristics, and report where independent methods agree
(agreement raises confidence; it does not prove noise).

Outputs (results/noise_audit/):
  all_noise_scores.csv, top_50_suspects.csv, top_100_suspects.csv,
  top_200_suspects.csv, suspect_overlap_report.json

Usage:
    python -m src.experiments.run_noise_audit --config configs/noise_audit.yaml
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from src.data.loading import load_source
from src.data.preprocessing import Preprocessor
from src.models.contrastive_clustering import CCConfig, train_cc, view_instability
from src.noise.confident_learning import confident_learning_scores, oof_probabilities
from src.noise.heuristic_rules import heuristic_flags
from src.noise.knn_noise import knn_label_disagreement_score
from src.noise.scoring import (
    centroid_distance_scores, cluster_entropy_score, cluster_label_mismatch_score,
    ensemble_score, normalize01,
)
from src.utils.io import save_csv, save_json
from src.utils.seeds import set_seed
from ._common import out_dir, resolve_config, log

DEFAULTS = {
    "source": "outbrain", "seed": 42, "svd_dim": 64, "subsample": None,
    "cc_epochs": 20, "cc_num_clusters": 4, "cc_aug": "medium_http_aug",
    "knn_k": 20, "instability_views": 5,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--cc-epochs", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = resolve_config(args, DEFAULTS)
    if args.cc_epochs is not None:
        cfg["cc_epochs"] = args.cc_epochs

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # Audit covers the FULL corpus. Features are label-free (TF-IDF/SVD), so
    # fitting on all rows leaks no labels — appropriate for scoring given labels.
    h, r, y = load_source(cfg.get("source", "outbrain"))
    if cfg.get("subsample"):
        sel = np.random.RandomState(seed).choice(len(y), int(cfg["subsample"]), replace=False)
        h = [h[i] for i in sel]; r = [r[i] for i in sel]; y = y[sel]
    log.info(f"Auditing {len(y)} samples from source={cfg.get('source')}")

    pre = Preprocessor(svd_dim=int(cfg.get("svd_dim", 64)), seed=seed).fit(h, r)
    X = pre.transform(h, r)

    cc_cfg = CCConfig(num_clusters=int(cfg.get("cc_num_clusters", 4)),
                      augmentation_type=cfg.get("cc_aug", "medium_http_aug"),
                      epochs=int(cfg.get("cc_epochs", 20)), seed=seed)
    log.info("Training CC on full corpus for embeddings / cluster scores")
    cc = train_cc(h, r, pre, cc_cfg, verbose=False)
    emb, probs, assign = cc["embeddings_train"], cc["cluster_probs_train"], cc["cluster_assignments_train"]

    log.info("Computing scores (Confident Learning OOF, KNN, CC, centroids)")
    proba = oof_probabilities(X, y, seed=seed)
    cl_score, cl_flags, _ = confident_learning_scores(y, proba)
    cd = centroid_distance_scores(emb, y)
    k = int(cfg.get("knn_k", 20))

    scores = {
        "score_knn_raw": knn_label_disagreement_score(X, y, k=k),
        "score_knn_cc": knn_label_disagreement_score(emb, y, k=k),
        "score_confident_learning": cl_score,
        "score_cluster_entropy": cluster_entropy_score(probs),
        "score_view_instability": view_instability(
            h, r, pre, cc["model"], cc_cfg.augmentation_type,
            n_views=int(cfg.get("instability_views", 5)), seed=seed),
        "score_cluster_label_mismatch": cluster_label_mismatch_score(assign, y),
        "score_centroid_distance": cd["relative_distance"],
    }
    scores["score_ensemble"] = ensemble_score([
        scores["score_knn_raw"], scores["score_knn_cc"],
        scores["score_confident_learning"], scores["score_cluster_entropy"],
        scores["score_view_instability"], scores["score_centroid_distance"],
    ])

    heur = heuristic_flags(h, r)

    rows = []
    for i in range(len(y)):
        row = {
            "sample_id": i,
            "observed_label": int(y[i]),
            "headers_raw": h[i][:500],
            "request_raw": str(r[i])[:300],
            "user_agent": str(heur["user_agent"][i])[:300],
            "has_template": bool(heur["has_template"][i]),
            "has_crawler_ua": bool(heur["has_crawler_ua"][i]),
            "has_script_ua": bool(heur["has_script_ua"][i]),
            "heuristic_bot_flag": bool(heur["heuristic_bot_flag"][i]),
            "cl_flag": bool(cl_flags[i]),
        }
        for name, s in scores.items():
            row[name] = round(float(s[i]), 6)
        rows.append(row)

    base = out_dir(cfg, "noise_audit")
    cols = ["sample_id", "observed_label", "score_knn_raw", "score_knn_cc",
            "score_confident_learning", "score_cluster_entropy",
            "score_view_instability", "score_cluster_label_mismatch",
            "score_centroid_distance", "score_ensemble", "headers_raw",
            "request_raw", "user_agent", "has_template", "has_crawler_ua",
            "has_script_ua", "heuristic_bot_flag", "cl_flag"]
    save_csv(rows, os.path.join(base, "all_noise_scores.csv"), columns=cols)
    ranked = sorted(rows, key=lambda r_: r_["score_ensemble"], reverse=True)
    for n in (50, 100, 200):
        save_csv(ranked[:n], os.path.join(base, f"top_{n}_suspects.csv"), columns=cols)

    overlap = _overlap_report(scores, heur["heuristic_bot_flag"], cl_flags, y, k=100)
    save_json(overlap, os.path.join(base, "suspect_overlap_report.json"))

    print("\n=== Real-data noise audit ===")
    print(f"  samples audited           : {len(y)}")
    print(f"  Confident-Learning flags  : {int(cl_flags.sum())} "
          f"({100 * cl_flags.mean():.2f}%)")
    print(f"  heuristic bot flags       : {int(heur['heuristic_bot_flag'].sum())}")
    print(f"  top-100 KNN_raw ∩ KNN_cc  : {overlap['top100_knn_raw_AND_knn_cc']['intersection']}")
    print(f"  top-100 KNN_cc ∩ CL       : {overlap['top100_knn_cc_AND_confident_learning']['intersection']}")
    print(f"  top-100 ensemble ∩ heur.  : {overlap['top100_ensemble_AND_heuristic']['intersection']}")
    print(f"\nSaved audit outputs under {base}")


def _topk_idx(score, k):
    return set(np.argsort(score)[::-1][:k].tolist())


def _overlap_report(scores, heuristic_flag, cl_flags, y, k=100):
    def ov(a, b):
        inter = len(a & b)
        union = len(a | b)
        return {"intersection": inter, "jaccard": round(inter / union, 4) if union else 0.0}

    knn_raw = _topk_idx(scores["score_knn_raw"], k)
    knn_cc = _topk_idx(scores["score_knn_cc"], k)
    cl = _topk_idx(scores["score_confident_learning"], k)
    ens = _topk_idx(scores["score_ensemble"], k)
    heur_idx = set(np.where(heuristic_flag)[0].tolist())
    return {
        "k": k,
        "top100_knn_raw_AND_knn_cc": ov(knn_raw, knn_cc),
        "top100_knn_cc_AND_confident_learning": ov(knn_cc, cl),
        "top100_ensemble_AND_heuristic": ov(ens, heur_idx),
        "confident_learning_flag_count": int(cl_flags.sum()),
        "heuristic_flag_count": int(len(heur_idx)),
        "note": "Overlap raises confidence in a candidate; it is not proof of a "
                "mislabel. Only synthetic-noise evaluation has a ground truth.",
    }


if __name__ == "__main__":
    main()
