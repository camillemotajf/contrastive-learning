"""Label-efficiency curve — F1 vs fraction of labels available.

Compares supervised baselines (LogReg, RF, MLP-from-scratch) against
representation-based methods (SSL probe, Triplet probe) as labels become scarce.
Hypothesis: representation methods degrade more gracefully in the low-label
regime. Aggregated over seeds. Saves results/baseline/label_efficiency.json.

Usage:
    python -m src.experiments.run_label_efficiency --config configs/baseline.yaml
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.classification import classification_metrics, linear_probe_metrics
from src.evaluation.label_efficiency import label_subset
from src.models import baselines as bl
from src.models.encoder import TrafficEncoder
from src.models.ssl import ssl_pretrain, backbone_embed
from src.models.triplet import train_triplet
from src.utils.config import Config
from src.utils.io import save_json
from src.utils.seeds import set_seed
from ._common import add_common_args, make_split, out_dir, resolve_config, log

DEFAULTS = {
    "source": "outbrain", "test_size": 0.3, "svd_dim": 64,
    "seeds": [42, 7, 123], "fractions": [0.01, 0.05, 0.10, 0.25, 0.50, 1.00],
}
METHODS = ["logreg", "rf", "mlp_scratch", "ssl_probe", "triplet_probe"]


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mlp_scratch(Xsub, ysub, Xte, epochs=60, batch_size=128, lr=1e-3):
    device = _device()
    Xs = torch.tensor(Xsub, device=device); ys = torch.tensor(ysub, device=device)
    backbone = TrafficEncoder(Xsub.shape[1], embed_dim=64).to(device)
    head = nn.Linear(64, 2).to(device)
    opt = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=lr)
    crit = nn.CrossEntropyLoss()
    n = len(ys); bs = min(batch_size, n)
    backbone.train(); head.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, max(1, n - bs + 1), bs):
            idx = perm[i:i + bs]
            loss = crit(head(backbone(Xs[idx])), ys[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    backbone.eval(); head.eval()
    with torch.no_grad():
        return torch.softmax(head(backbone(torch.tensor(Xte, device=device))), 1)[:, 1].cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = resolve_config(args, DEFAULTS)
    seeds = cfg.get("seeds", [42])
    fractions = cfg.get("fractions", [0.01, 0.1, 1.0])

    store = {m: {f: [] for f in fractions} for m in METHODS}
    for seed in seeds:
        set_seed(seed)
        sp = make_split(Config({**cfg, "seed": seed}))
        backbone, ssl_tr = ssl_pretrain(sp.Xtr)
        ssl_te = backbone_embed(backbone, sp.Xte)
        log.info(f"[seed {seed}] train={len(sp.y_tr)} test={len(sp.y_te)}")

        for f in fractions:
            idx = label_subset(sp.y_tr, f, seed=seed)
            Xs, ys = sp.Xtr[idx], sp.y_tr[idx]

            p = bl.fit_logreg(Xs, ys, seed).predict_proba(sp.Xte)[:, 1]
            store["logreg"][f].append(classification_metrics(sp.y_te, p))
            p = bl.fit_rf(Xs, ys, seed).predict_proba(sp.Xte)[:, 1]
            store["rf"][f].append(classification_metrics(sp.y_te, p))
            p = mlp_scratch(Xs, ys, sp.Xte)
            store["mlp_scratch"][f].append(classification_metrics(sp.y_te, p))
            store["ssl_probe"][f].append(
                linear_probe_metrics(ssl_tr[idx], ys, ssl_te, sp.y_te, seed))
            e_tr, e_te = train_triplet(Xs, ys, sp.Xte)
            store["triplet_probe"][f].append(
                linear_probe_metrics(e_tr, ys, e_te, sp.y_te, seed))
            log.info(f"  f={f:<5} RF={store['rf'][f][-1]['f1']:.3f} "
                     f"SSL={store['ssl_probe'][f][-1]['f1']:.3f} "
                     f"TRI={store['triplet_probe'][f][-1]['f1']:.3f}")

    out = {m: {"fractions": fractions, "f1_mean": [], "f1_std": [], "auc_mean": []}
           for m in METHODS}
    for m in METHODS:
        for f in fractions:
            f1 = [r["f1"] for r in store[m][f]]
            auc = [r["auc"] for r in store[m][f]]
            out[m]["f1_mean"].append(round(float(np.mean(f1)), 4))
            out[m]["f1_std"].append(round(float(np.std(f1)), 4))
            out[m]["auc_mean"].append(round(float(np.mean(auc)), 4))

    base = out_dir(cfg, "baseline")
    save_json({"config": dict(cfg), "seeds": seeds, "results": out},
              os.path.join(base, "label_efficiency.json"))

    print("\n=== F1 by label fraction (mean over seeds) ===")
    print("frac     " + "".join(f"{m:>14}" for m in METHODS))
    for i, f in enumerate(fractions):
        print(f"{f:<8} " + "".join(f"{out[m]['f1_mean'][i]:>14.3f}" for m in METHODS))
    print(f"\nSaved {base}/label_efficiency.json")


if __name__ == "__main__":
    main()
