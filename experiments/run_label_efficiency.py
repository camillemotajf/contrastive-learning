"""Label-efficiency curve — the experiment where contrastive learning can win.

For each training-label fraction f in {1%, 5%, 10%, 25%, 50%, 100%} we compare:

  - logreg_raw    : LogisticRegression on raw features, trained on the f-subset
  - rf_raw        : RandomForest on raw features, trained on the f-subset
  - mlp_scratch   : supervised neural net (same backbone) trained from scratch
                    on the f-subset
  - ssl_probe     : SELF-SUPERVISED contrastive pretraining (InfoNCE / NT-Xent)
                    on ALL unlabeled training data, frozen backbone, then a
                    logistic probe trained on the f-subset
  - triplet_probe : supervised contrastive (TripletMarginLoss) trained on the
                    f-subset, then a logistic probe

All methods are evaluated on the same held-out TEST set (F1, ROC-AUC).
The hypothesis: ssl_probe degrades more gracefully than supervised-from-scratch
as labels become scarce, because the encoder learns structure without labels.

Results are aggregated over 3 seeds and saved to experiments/label_efficiency.json.
"""
import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import build_features
from src.encoder import TrafficEncoder
from src.contrastive_clustering import InstanceLoss

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
FILE_UNSAFE = os.path.join(DATA_DIR, "outbrain-unsafe-10k.json")
FILE_BOTS = os.path.join(DATA_DIR, "outbrain-bot-10k.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 7, 123]
FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def augment(x, mask_prob=0.15, noise_std=0.05):
    mask = (torch.rand_like(x) > mask_prob).float()
    return x * mask + torch.randn_like(x) * noise_std


# --------------------------------------------------------------------------- #
# Self-supervised contrastive pretraining (NT-Xent / InfoNCE, NO labels)
# --------------------------------------------------------------------------- #
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=64, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim))

    def forward(self, h):
        return F.normalize(self.net(h), dim=1)


def ssl_pretrain(Xtr, epochs=40, batch_size=256, lr=1e-3):
    """Returns a frozen backbone that maps inputs -> 64-d representation."""
    Xtr_t = torch.tensor(Xtr, device=DEVICE)
    n = Xtr.shape[0]
    backbone = TrafficEncoder(input_dim=Xtr.shape[1], embed_dim=64).to(DEVICE)
    proj = ProjectionHead(64, 32).to(DEVICE)
    opt = torch.optim.Adam(list(backbone.parameters()) + list(proj.parameters()), lr=lr)
    crit = InstanceLoss(temperature=0.5).to(DEVICE)

    backbone.train(); proj.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n - batch_size + 1, batch_size):
            xb = Xtr_t[perm[i : i + batch_size]]
            zi = proj(backbone(augment(xb)))
            zj = proj(backbone(augment(xb)))
            loss = crit(zi, zj)
            opt.zero_grad(); loss.backward(); opt.step()

    backbone.eval()
    with torch.no_grad():
        emb_tr = backbone(Xtr_t).cpu().numpy()
    return backbone, emb_tr


def backbone_embed(backbone, X):
    with torch.no_grad():
        return backbone(torch.tensor(X, device=DEVICE)).cpu().numpy()


# --------------------------------------------------------------------------- #
# Supervised neural net from scratch (same backbone + linear head)
# --------------------------------------------------------------------------- #
def mlp_scratch(Xsub, ysub, Xte, epochs=60, batch_size=128, lr=1e-3):
    Xs = torch.tensor(Xsub, device=DEVICE); ys = torch.tensor(ysub, device=DEVICE)
    backbone = TrafficEncoder(input_dim=Xsub.shape[1], embed_dim=64).to(DEVICE)
    head = nn.Linear(64, 2).to(DEVICE)
    opt = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=lr)
    crit = nn.CrossEntropyLoss()
    n = len(ys); bs = min(batch_size, n)

    backbone.train(); head.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, max(1, n - bs + 1), bs):
            idx = perm[i : i + bs]
            logits = head(backbone(Xs[idx]))
            loss = crit(logits, ys[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    backbone.eval(); head.eval()
    with torch.no_grad():
        proba = torch.softmax(head(backbone(torch.tensor(Xte, device=DEVICE))), dim=1)[:, 1].cpu().numpy()
    return proba


# --------------------------------------------------------------------------- #
# Supervised contrastive (triplet) on the labeled subset
# --------------------------------------------------------------------------- #
def triplet_embed(Xsub, ysub, Xte, embed_dim=32, epochs=40, batch_size=128, lr=1e-3):
    Xs = torch.tensor(Xsub, device=DEVICE)
    ysub = np.asarray(ysub)
    cls_idx = {c: np.where(ysub == c)[0] for c in np.unique(ysub)}
    classes = list(cls_idx)
    if len(classes) < 2:
        return None, None
    n = len(ysub); bs = min(batch_size, n)
    model = TrafficEncoder(input_dim=Xsub.shape[1], embed_dim=embed_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.TripletMarginLoss(margin=1.0, p=2)

    model.train()
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, max(1, n - bs + 1), bs):
            a = perm[i : i + bs]; al = ysub[a]
            p = np.array([np.random.choice(cls_idx[l]) for l in al])
            nl = np.array([random.choice([c for c in classes if c != l]) for l in al])
            ng = np.array([np.random.choice(cls_idx[l]) for l in nl])
            loss = crit(model(Xs[a]), model(Xs[p]), model(Xs[ng]))
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        e_tr = model(Xs).cpu().numpy()
        e_te = model(torch.tensor(Xte, device=DEVICE)).cpu().numpy()
    return e_tr, e_te


def probe(emb_tr, ytr, emb_te, yte, seed):
    clf = LogisticRegression(max_iter=2000, random_state=seed).fit(emb_tr, ytr)
    pr = clf.predict_proba(emb_te)[:, 1]
    return f1_score(yte, (pr >= 0.5).astype(int)), roc_auc_score(yte, pr)


# --------------------------------------------------------------------------- #
def run():
    print(f"Device: {DEVICE} | seeds={SEEDS} | fractions={FRACTIONS}")
    methods = ["logreg_raw", "rf_raw", "mlp_scratch", "ssl_probe", "triplet_probe"]
    # store[method][frac] = list of (f1, auc) over seeds
    store = {m: {f: [] for f in FRACTIONS} for m in methods}

    for seed in SEEDS:
        set_seed(seed)
        Xtr, Xte, ytr, yte = build_features(FILE_UNSAFE, FILE_BOTS, test_size=0.3, seed=seed)
        print(f"\n[seed {seed}] train={len(ytr)} test={len(yte)} | SSL pretraining...")
        backbone, ssl_tr = ssl_pretrain(Xtr)
        ssl_te = backbone_embed(backbone, Xte)

        for f in FRACTIONS:
            if f < 1.0:
                idx, _ = train_test_split(np.arange(len(ytr)), train_size=f,
                                          random_state=seed, stratify=ytr)
            else:
                idx = np.arange(len(ytr))
            Xs, ys = Xtr[idx], ytr[idx]

            # baselines
            lr = LogisticRegression(max_iter=2000, random_state=seed).fit(Xs, ys)
            p = lr.predict_proba(Xte)[:, 1]
            store["logreg_raw"][f].append((f1_score(yte, (p >= .5).astype(int)), roc_auc_score(yte, p)))

            rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1).fit(Xs, ys)
            p = rf.predict_proba(Xte)[:, 1]
            store["rf_raw"][f].append((f1_score(yte, (p >= .5).astype(int)), roc_auc_score(yte, p)))

            # supervised MLP from scratch
            p = mlp_scratch(Xs, ys, Xte)
            store["mlp_scratch"][f].append((f1_score(yte, (p >= .5).astype(int)), roc_auc_score(yte, p)))

            # SSL pretrain (frozen) + probe on subset
            store["ssl_probe"][f].append(probe(ssl_tr[idx], ys, ssl_te, yte, seed))

            # supervised contrastive (triplet) + probe
            e_tr, e_te = triplet_embed(Xs, ys, Xte)
            store["triplet_probe"][f].append(probe(e_tr, ys, e_te, yte, seed))

            print(f"  f={f:>4}: "
                  f"RF={store['rf_raw'][f][-1][0]:.3f} "
                  f"MLP={store['mlp_scratch'][f][-1][0]:.3f} "
                  f"SSL={store['ssl_probe'][f][-1][0]:.3f} "
                  f"TRI={store['triplet_probe'][f][-1][0]:.3f}")

    # aggregate
    out = {m: {"frac": FRACTIONS, "f1_mean": [], "f1_std": [], "auc_mean": [], "auc_std": []}
           for m in methods}
    for m in methods:
        for f in FRACTIONS:
            arr = np.array(store[m][f])  # (seeds, 2)
            out[m]["f1_mean"].append(float(arr[:, 0].mean()))
            out[m]["f1_std"].append(float(arr[:, 0].std()))
            out[m]["auc_mean"].append(float(arr[:, 1].mean()))
            out[m]["auc_std"].append(float(arr[:, 1].std()))

    path = os.path.join(os.path.dirname(__file__), "label_efficiency.json")
    with open(path, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"\nSaved {path}")

    print("\n=== F1 by label fraction (mean over seeds) ===")
    hdr = "frac     " + "".join(f"{m:>14}" for m in methods)
    print(hdr)
    for i, f in enumerate(FRACTIONS):
        row = f"{f:<8} " + "".join(f"{out[m]['f1_mean'][i]:>14.3f}" for m in methods)
        print(row)


if __name__ == "__main__":
    run()
