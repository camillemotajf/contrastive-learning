"""Per-source study runner — one traffic source in, saved artifacts out.

This is the single place that runs the full study for ONE traffic source and
*persists* every result, so the numbers behind the article's figures are
reproducible files on disk rather than transient notebook outputs.

For each source it computes and saves, under ``results/cross_source/<source>/``:

* ``classification.csv``  — F1/AUC/Acc/Precision/Recall (mean ± std over seeds)
  for LogReg, Random Forest, Triplet+probe, SSL+probe.
* ``clustering.csv``      — ARI/NMI/Hungarian-acc for KMeans and PCA+KMeans, and
  for the Contrastive Clustering (CC) partition.
* ``cc_probe.json``       — CC frozen-embedding linear-probe metrics + collapse
  diagnostics (non-empty clusters, cluster-size entropy).
* ``synthetic_noise.csv`` — ROC-AUC / AUPRC / precision@100 of every suspicion
  score against the *known* synthetic-noise mask, per noise rate and seed.
* ``knn_comparison.csv``  — KNN noise detection on raw features vs the CC
  embedding (the honest, data-dependent question).
* ``real_audit.json`` + ``top_suspects.csv`` — applying the validated scores to
  the real labels (no ground truth — *candidates for review*).
* ``summary.json``        — the headline numbers used for the cross-source plots.

Everything goes through :func:`src.experiments._common.make_split`, so the
"split on raw text BEFORE fitting any transform" rule is enforced in one place
and the comparison across sources stays leakage-free.

Honesty: real-data outputs are *suspicion rankings* / *candidates for review*,
never "real probabilities of noise". Detector quality is only ever established on
the synthetic setting, where a ground-truth flip mask exists.
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.experiments._common import RESULTS_DIR, make_split
from src.utils.config import Config
from src.utils.io import load_json, save_csv, save_json, save_npy
from src.utils.seeds import set_seed

from src.models import baselines as bl
from src.models.triplet import train_triplet
from src.models.ssl import ssl_pretrain, backbone_embed
from src.models.contrastive_clustering import CCConfig, train_cc, view_instability

from src.evaluation.classification import classification_metrics, linear_probe_metrics
from src.evaluation.clustering import clustering_metrics
from src.evaluation.noise_detection import noise_detection_metrics

from src.noise.synthetic_noise import inject_synthetic_label_noise
from src.noise.confident_learning import oof_probabilities, confident_learning_scores
from src.noise.knn_noise import (knn_label_disagreement_score,
                                 knn_weighted_disagreement_score)
from src.noise.scoring import (centroid_distance_scores, cluster_entropy_score,
                               ensemble_score)
from src.noise.heuristic_rules import heuristic_flags


# Suspicion scores graded against the synthetic mask (order is the table order).
SCORE_NAMES = [
    "confident_learning", "knn_raw", "knn_cc", "cc_cluster_entropy",
    "cc_view_instability", "centroid_relative_distance", "ensemble",
]
CLS_METHODS = ["LogReg", "RandomForest", "Triplet + probe", "SSL + probe"]


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def source_out_dir(source: str, out_root: Optional[str] = None) -> str:
    return os.path.join(out_root or os.path.join(RESULTS_DIR, "cross_source"), source)


def _read_table(path: str) -> List[dict]:
    """Read a saved CSV back into a list of dicts, coercing numeric cells."""
    import csv
    rows: List[dict] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out = {}
            for k, v in row.items():
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = v
            rows.append(out)
    return rows


def load_source_study(source: str, out_root: Optional[str] = None) -> Optional[Dict[str, object]]:
    """Reconstruct a results dict from previously-saved artifacts, or ``None`` if
    this source has not been run yet. Lets the report notebook *load and plot*
    without recomputing — the heavy run is done once by ``run_cross_source``."""
    out_dir = source_out_dir(source, out_root)
    summary_path = os.path.join(out_dir, "summary.json")
    if not os.path.exists(summary_path):
        return None
    cc = load_json(os.path.join(out_dir, "cc_probe.json"))
    return {
        "source": source,
        "out_dir": out_dir,
        "summary": load_json(summary_path),
        "classification": _read_table(os.path.join(out_dir, "classification.csv")),
        "clustering": _read_table(os.path.join(out_dir, "clustering.csv")),
        "cc_probe": cc.get("probe", {}),
        "cc_collapse": cc.get("collapse", {}),
        "real_audit": load_json(os.path.join(out_dir, "real_audit.json")),
    }


def run_source_study(
    source: str,
    seeds: Sequence[int] = (42, 7),
    subsample: Optional[int] = 4000,
    test_size: float = 0.3,
    svd_dim: int = 64,
    cc_clusters: int = 4,
    cc_epochs: int = 15,
    cc_aug: str = "medium_http_aug",
    noise_rates: Sequence[float] = (0.05, 0.10, 0.20),
    knn_k: int = 20,
    out_root: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run the full study for one source, save artifacts, return a results dict.

    The returned dict mirrors the saved files so a notebook can plot without
    re-reading disk. All randomness is seeded; CC is trained once per seed
    (it is unsupervised) and reused across noise rates.
    """
    out_dir = os.path.join(out_root or os.path.join(RESULTS_DIR, "cross_source"), source)
    os.makedirs(out_dir, exist_ok=True)

    base_cfg = dict(source=source, subsample=subsample, test_size=test_size, svd_dim=svd_dim)

    def _log(*a):
        if verbose:
            print(f"[{source}]", *a)

    t_start = time.time()

    # -- balance sheet (from the first seed's split) ---------------------- #
    sp0 = make_split(Config({**base_cfg, "seed": int(seeds[0])}))
    balance = {
        "n_train": int(len(sp0.y_tr)), "n_test": int(len(sp0.y_te)),
        "train_human": int((sp0.y_tr == 0).sum()), "train_bot": int((sp0.y_tr == 1).sum()),
        "test_human": int((sp0.y_te == 0).sum()), "test_bot": int((sp0.y_te == 1).sum()),
        "feature_dim": int(sp0.Xtr.shape[1]),
    }

    # ------------------------------------------------------------------ #
    # 1. Classification + clustering baselines (multi-seed)
    # ------------------------------------------------------------------ #
    cls_runs: Dict[str, List[dict]] = {m: [] for m in CLS_METHODS}
    clu_runs: Dict[str, List[dict]] = {m: [] for m in ["KMeans (raw)", "PCA + KMeans"]}
    cc_clu_runs: List[dict] = []
    cc_probe_runs: List[dict] = []
    cc_collapse_runs: List[dict] = []

    # ------------------------------------------------------------------ #
    # 2. Synthetic-noise detection + KNN raw-vs-CC (CC trained once / seed)
    # ------------------------------------------------------------------ #
    syn_rows: List[dict] = []
    knn_rows: List[dict] = []

    for seed in seeds:
        set_seed(seed)
        s = make_split(Config({**base_cfg, "seed": int(seed)}))

        # --- baselines ---
        cls_runs["LogReg"].append(bl.supervised_metrics(bl.fit_logreg(s.Xtr, s.y_tr, seed), s.Xte, s.y_te))
        cls_runs["RandomForest"].append(bl.supervised_metrics(bl.fit_rf(s.Xtr, s.y_tr, seed), s.Xte, s.y_te))
        e_tr, e_te = train_triplet(s.Xtr, s.y_tr, s.Xte)
        cls_runs["Triplet + probe"].append(linear_probe_metrics(e_tr, s.y_tr, e_te, s.y_te, seed))
        backbone, ssl_tr = ssl_pretrain(s.Xtr)
        cls_runs["SSL + probe"].append(
            linear_probe_metrics(ssl_tr, s.y_tr, backbone_embed(backbone, s.Xte), s.y_te, seed))
        _, km = bl.kmeans_predict(s.Xtr, s.Xte, seed=seed)
        clu_runs["KMeans (raw)"].append(clustering_metrics(s.y_te, km))
        _, pkm = bl.pca_kmeans_predict(s.Xtr, s.Xte, seed=seed)
        clu_runs["PCA + KMeans"].append(clustering_metrics(s.y_te, pkm))

        # --- Contrastive Clustering (unsupervised; trained once per seed) ---
        cfg = CCConfig(num_clusters=cc_clusters, augmentation_type=cc_aug,
                       epochs=cc_epochs, seed=seed)
        cc = train_cc(s.h_tr, s.r_tr, s.pre, cfg, h_te=s.h_te, r_te=s.r_te, verbose=False)
        emb = cc["embeddings_train"]
        probs = cc["cluster_probs_train"]
        assign = cc["cluster_assignments_train"]

        cc_clu = clustering_metrics(s.y_tr, assign)
        cc_clu_runs.append({k: cc_clu[k] for k in ("ari", "nmi", "cluster_acc")})
        cc_collapse_runs.append({
            "n_nonempty": int(cc_clu["distribution"]["n_nonempty"]),
            "entropy": float(cc_clu["distribution"]["entropy"]),
        })
        cc_probe_runs.append(
            linear_probe_metrics(emb, s.y_tr, cc["embeddings_test"], s.y_te, seed))

        # --- synthetic-noise detection ---
        for rate in noise_rates:
            y_noisy, mask = inject_synthetic_label_noise(
                s.y_tr, rate, mode="symmetric", random_state=seed)
            proba = oof_probabilities(s.Xtr, y_noisy, seed=seed)
            cl_score, _, _ = confident_learning_scores(y_noisy, proba)
            cd = centroid_distance_scores(emb, y_noisy)
            sc = {
                "confident_learning": cl_score,
                "knn_raw": knn_label_disagreement_score(s.Xtr, y_noisy, k=knn_k),
                "knn_cc": knn_label_disagreement_score(emb, y_noisy, k=knn_k),
                "cc_cluster_entropy": cluster_entropy_score(probs),
                "cc_view_instability": view_instability(
                    s.h_tr, s.r_tr, s.pre, cc["model"], cc_aug, n_views=3, seed=seed),
                "centroid_relative_distance": cd["relative_distance"],
            }
            sc["ensemble"] = ensemble_score([
                sc["confident_learning"], sc["knn_raw"], sc["knn_cc"],
                sc["cc_cluster_entropy"], sc["cc_view_instability"],
                sc["centroid_relative_distance"]])
            for name in SCORE_NAMES:
                m = noise_detection_metrics(sc[name], mask, ks=[50, 100, 200])
                syn_rows.append({
                    "source": source, "score": name, "rate": float(rate), "seed": int(seed),
                    "roc_auc": m["roc_auc"], "auprc": m["auprc"],
                    "precision@100": m["precision@100"], "recall@100": m["recall@100"],
                })

        # --- KNN raw vs CC at a fixed 10% rate ---
        y_noisy, mask = inject_synthetic_label_noise(
            s.y_tr, 0.10, mode="symmetric", random_state=seed)
        knn_variants = {
            "KNN raw": knn_label_disagreement_score(s.Xtr, y_noisy, k=knn_k),
            "KNN CC": knn_label_disagreement_score(emb, y_noisy, k=knn_k),
            "KNN raw (weighted)": knn_weighted_disagreement_score(s.Xtr, y_noisy, k=knn_k),
            "KNN CC (weighted)": knn_weighted_disagreement_score(emb, y_noisy, k=knn_k),
        }
        for name, score in knn_variants.items():
            knn_rows.append({
                "source": source, "variant": name, "seed": int(seed),
                "roc_auc": noise_detection_metrics(score, mask, ks=[100])["roc_auc"],
            })

        _log(f"seed {seed} done ({time.time()-t_start:.0f}s elapsed)")

    # ------------------------------------------------------------------ #
    # Aggregate the baseline / CC tables
    # ------------------------------------------------------------------ #
    metric_keys = ["f1", "auc", "acc", "precision", "recall"]
    cls_table = []
    for m in CLS_METHODS:
        row = {"method": m}
        for k in metric_keys:
            ms = _mean_std([d[k] for d in cls_runs[m]])
            row[f"{k}_mean"], row[f"{k}_std"] = ms["mean"], ms["std"]
        cls_table.append(row)

    clu_keys = ["ari", "nmi", "cluster_acc"]
    clu_table = []
    for m, runs in {**clu_runs, "Contrastive Clustering": cc_clu_runs}.items():
        row = {"method": m}
        for k in clu_keys:
            ms = _mean_std([d[k] for d in runs])
            row[f"{k}_mean"], row[f"{k}_std"] = ms["mean"], ms["std"]
        clu_table.append(row)

    cc_probe = {k: _mean_std([d[k] for d in cc_probe_runs]) for k in metric_keys}
    cc_collapse = {
        "n_nonempty": _mean_std([d["n_nonempty"] for d in cc_collapse_runs]),
        "entropy": _mean_std([d["entropy"] for d in cc_collapse_runs]),
        "num_clusters": int(cc_clusters),
    }

    # ------------------------------------------------------------------ #
    # 3. Real-data audit (no ground truth — candidates for review)
    # ------------------------------------------------------------------ #
    seed0 = int(seeds[0])
    set_seed(seed0)
    s = make_split(Config({**base_cfg, "seed": seed0}))
    cfg = CCConfig(num_clusters=cc_clusters, augmentation_type=cc_aug, epochs=cc_epochs, seed=seed0)
    cc_r = train_cc(s.h_tr, s.r_tr, s.pre, cfg, verbose=False)
    emb = cc_r["embeddings_train"]
    probs = cc_r["cluster_probs_train"]

    proba = oof_probabilities(s.Xtr, s.y_tr, seed=seed0)
    cl_score, cl_flags, _ = confident_learning_scores(s.y_tr, proba)
    cd = centroid_distance_scores(emb, s.y_tr)
    real_scores = {
        "score_confident_learning": cl_score,
        "score_knn_raw": knn_label_disagreement_score(s.Xtr, s.y_tr, k=knn_k),
        "score_knn_cc": knn_label_disagreement_score(emb, s.y_tr, k=knn_k),
        "score_cluster_entropy": cluster_entropy_score(probs),
        "score_view_instability": view_instability(
            s.h_tr, s.r_tr, s.pre, cc_r["model"], cc_aug, n_views=3, seed=seed0),
        "score_centroid_distance": cd["relative_distance"],
    }
    real_scores["score_ensemble"] = ensemble_score(list(real_scores.values()))
    heur = heuristic_flags(s.h_tr, s.r_tr)

    real_audit = {
        "source": source,
        "n_train": int(len(s.y_tr)),
        "confident_learning_flags": int(cl_flags.sum()),
        "confident_learning_flag_rate": float(cl_flags.mean()),
        "heuristic_bot_flags": int(heur["heuristic_bot_flag"].sum()),
    }
    # top-100 ranking overlaps (agreement raises confidence, not proof)
    def _topk(score, k=100):
        return set(np.argsort(score)[::-1][:k].tolist())
    real_audit["overlap_top100"] = {
        "knn_raw_vs_knn_cc": len(_topk(real_scores["score_knn_raw"]) & _topk(real_scores["score_knn_cc"])),
        "knn_cc_vs_confident_learning": len(_topk(real_scores["score_knn_cc"]) & _topk(real_scores["score_confident_learning"])),
        "ensemble_vs_heuristic_bots": len(_topk(real_scores["score_ensemble"]) & set(np.where(heur["heuristic_bot_flag"])[0].tolist())),
    }

    top_n = 100
    order = np.argsort(real_scores["score_ensemble"])[::-1][:top_n]
    top_suspects = [{
        "rank": int(i + 1),
        "train_index": int(j),
        "observed_label": int(s.y_tr[j]),
        "ensemble": float(real_scores["score_ensemble"][j]),
        "confident_learning": float(real_scores["score_confident_learning"][j]),
        "knn_cc": float(real_scores["score_knn_cc"][j]),
        "has_template": bool(heur["has_template"][j]),
        "crawler_ua": bool(heur["has_crawler_ua"][j]),
        "script_ua": bool(heur["has_script_ua"][j]),
        "user_agent": str(heur["user_agent"][j])[:80],
    } for i, j in enumerate(order)]

    # ------------------------------------------------------------------ #
    # Persist everything
    # ------------------------------------------------------------------ #
    save_csv(cls_table, os.path.join(out_dir, "classification.csv"))
    save_csv(clu_table, os.path.join(out_dir, "clustering.csv"))
    save_json({"probe": cc_probe, "collapse": cc_collapse}, os.path.join(out_dir, "cc_probe.json"))
    save_csv(syn_rows, os.path.join(out_dir, "synthetic_noise.csv"))
    save_csv(knn_rows, os.path.join(out_dir, "knn_comparison.csv"))
    save_json(real_audit, os.path.join(out_dir, "real_audit.json"))
    save_csv(top_suspects, os.path.join(out_dir, "top_suspects.csv"))
    save_npy(real_scores["score_ensemble"], os.path.join(out_dir, "ensemble_score_train.npy"))

    # headline numbers for cross-source comparison
    best_cls = max(cls_table, key=lambda r: r["f1_mean"])
    syn_ens = [r["roc_auc"] for r in syn_rows if r["score"] == "ensemble" and not np.isnan(r["roc_auc"])]
    knn_raw = [r["roc_auc"] for r in knn_rows if r["variant"] == "KNN raw"]
    knn_cc = [r["roc_auc"] for r in knn_rows if r["variant"] == "KNN CC"]
    summary = {
        "source": source,
        "seeds": list(map(int, seeds)),
        "balance": balance,
        "best_classifier": {"method": best_cls["method"],
                            "f1": best_cls["f1_mean"], "f1_std": best_cls["f1_std"],
                            "auc": best_cls["auc_mean"]},
        "rf_f1": next(r["f1_mean"] for r in cls_table if r["method"] == "RandomForest"),
        "cc_probe_f1": cc_probe["f1"]["mean"],
        "cc_clustering_ari": next(r["ari_mean"] for r in clu_table if r["method"] == "Contrastive Clustering"),
        "cc_nonempty_clusters": cc_collapse["n_nonempty"]["mean"],
        "ensemble_noise_roc_auc": _mean_std(syn_ens) if syn_ens else None,
        "knn_raw_roc_auc": _mean_std(knn_raw) if knn_raw else None,
        "knn_cc_roc_auc": _mean_std(knn_cc) if knn_cc else None,
        "confident_learning_flag_rate": real_audit["confident_learning_flag_rate"],
        "heuristic_bot_flags": real_audit["heuristic_bot_flags"],
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    save_json(summary, os.path.join(out_dir, "summary.json"))
    _log(f"saved artifacts to {out_dir} ({summary['runtime_seconds']:.0f}s)")

    return {
        "source": source, "out_dir": out_dir, "summary": summary,
        "classification": cls_table, "clustering": clu_table,
        "cc_probe": cc_probe, "cc_collapse": cc_collapse,
        "synthetic_noise": syn_rows, "knn_comparison": knn_rows,
        "real_audit": real_audit, "top_suspects": top_suspects,
    }
