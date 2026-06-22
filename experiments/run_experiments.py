"""Runner de experimentos reprodutível para o estudo de contrastive learning.

Para cada seed e configuração de features (sem data leakage):
  1. Baselines não-supervisionados : raw+KMeans, PCA+KMeans
  2. Baselines supervisionados      : LogisticRegression, RandomForest (upper bound)
  3. Contrastive supervisionado     : TrafficEncoder + TripletMarginLoss -> linear probe
  4. Contrastive não-supervisionado : ContrastiveClusteringNet (InstanceLoss + ClusterLoss
                                      com entropia) -> ARI/NMI/cluster-acc + linear probe

Tudo avaliado em conjunto de TESTE held-out. Resultados agregados (média ± desvio)
em múltiplas seeds e salvos em experiments/results.json.

Uso:
    python experiments/run_experiments.py
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
from src.evaluate import clustering_metrics, linear_probe_metrics
from src.encoder import TrafficEncoder
from src.contrastive_clustering import ContrastiveClusteringNet, InstanceLoss, ClusterLoss

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
FILE_UNSAFE = os.path.join(DATA_DIR, "outbrain-unsafe-10k.json")
FILE_BOTS = os.path.join(DATA_DIR, "outbrain-bot-10k.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 7, 123]


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# --------------------------------------------------------------------------- #
# Treino dos modelos contrastivos (operam sobre tensores já normalizados)
# --------------------------------------------------------------------------- #
def train_triplet(Xtr, ytr, Xte, embed_dim=32, epochs=20, batch_size=256, lr=1e-3):
    """Contrastive supervisionado: triplets sorteados pelos rótulos do TREINO."""
    Xtr_t = torch.tensor(Xtr, device=DEVICE)
    Xte_t = torch.tensor(Xte, device=DEVICE)
    ytr = np.asarray(ytr)
    cls_idx = {c: np.where(ytr == c)[0] for c in np.unique(ytr)}
    classes = list(cls_idx)
    n = len(ytr)

    model = TrafficEncoder(input_dim=Xtr.shape[1], embed_dim=embed_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.TripletMarginLoss(margin=1.0, p=2)

    model.train()
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n - batch_size + 1, batch_size):
            a_idx = perm[i : i + batch_size]
            a_lab = ytr[a_idx]
            p_idx = np.array([np.random.choice(cls_idx[l]) for l in a_lab])
            neg_lab = np.array([random.choice([c for c in classes if c != l]) for l in a_lab])
            n_idx = np.array([np.random.choice(cls_idx[l]) for l in neg_lab])

            ea = model(Xtr_t[a_idx])
            ep = model(Xtr_t[p_idx])
            en = model(Xtr_t[n_idx])
            loss = crit(ea, ep, en)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        emb_tr = model(Xtr_t).cpu().numpy()
        emb_te = model(Xte_t).cpu().numpy()
    return emb_tr, emb_te


def train_contrastive_clustering(
    Xtr, Xte, num_clusters=2, epochs=30, batch_size=256, lr=5e-4,
    mask_prob=0.15, noise_std=0.05,
):
    """Contrastive não-supervisionado (CC): sem rótulos, duas views por augmentation."""
    Xtr_t = torch.tensor(Xtr, device=DEVICE)
    Xte_t = torch.tensor(Xte, device=DEVICE)
    n = Xtr.shape[0]

    model = ContrastiveClusteringNet(input_dim=Xtr.shape[1], num_clusters=num_clusters).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_ins = InstanceLoss(temperature=0.5).to(DEVICE)
    loss_clu = ClusterLoss(temperature=1.0, entropy_weight=1.0).to(DEVICE)

    def augment(x):
        mask = (torch.rand_like(x) > mask_prob).float()
        return x * mask + torch.randn_like(x) * noise_std

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n - batch_size + 1, batch_size):
            xb = Xtr_t[perm[i : i + batch_size]]
            za, pa = model(augment(xb))
            zb, pb = model(augment(xb))
            loss = loss_ins(za, zb) + loss_clu(pa, pb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        z_tr, _ = model(Xtr_t)
        z_te, p_te = model(Xte_t)
        clusters_te = torch.argmax(p_te, dim=1).cpu().numpy()
    return z_tr.cpu().numpy(), z_te.cpu().numpy(), clusters_te


# --------------------------------------------------------------------------- #
# Experimento completo para uma config + seed
# --------------------------------------------------------------------------- #
def run_once(seed, use_manual=True, canonicalize=False):
    set_seed(seed)
    Xtr, Xte, ytr, yte = build_features(
        FILE_UNSAFE, FILE_BOTS, test_size=0.3, seed=seed,
        use_manual=use_manual, canonicalize=canonicalize,
    )
    out = {}

    # 1. Baseline não-supervisionado: raw + KMeans
    km = KMeans(n_clusters=2, random_state=seed, n_init=10).fit(Xtr)
    out["raw_kmeans"] = clustering_metrics(yte, km.predict(Xte))

    # 2. Baseline não-supervisionado: PCA(32) + KMeans
    pca = PCA(n_components=32, random_state=seed).fit(Xtr)
    km2 = KMeans(n_clusters=2, random_state=seed, n_init=10).fit(pca.transform(Xtr))
    out["pca_kmeans"] = clustering_metrics(yte, km2.predict(pca.transform(Xte)))

    # 3. Baselines supervisionados (referência / upper bound)
    def sup(clf):
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        proba = clf.predict_proba(Xte)[:, 1]
        return {"f1": f1_score(yte, pred), "auc": roc_auc_score(yte, proba),
                "acc": accuracy_score(yte, pred)}
    out["logreg_raw"] = sup(LogisticRegression(max_iter=2000, random_state=seed))
    out["rf_raw"] = sup(RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1))

    # 4. Contrastive supervisionado (triplet) -> linear probe
    emb_tr, emb_te = train_triplet(Xtr, ytr, Xte)
    out["triplet_probe"] = linear_probe_metrics(emb_tr, ytr, emb_te, yte, seed=seed)

    # 5. Contrastive clustering (não-supervisionado)
    z_tr, z_te, clusters_te = train_contrastive_clustering(Xtr, Xte)
    out["cc_cluster"] = clustering_metrics(yte, clusters_te)            # qualidade do clustering
    out["cc_probe"] = linear_probe_metrics(z_tr, ytr, z_te, yte, seed=seed)  # qualidade da representação

    return out


def aggregate(runs):
    """runs: lista de dicts iguais -> média e desvio por método/métrica."""
    agg = {}
    for method in runs[0]:
        agg[method] = {}
        for metric in runs[0][method]:
            vals = [r[method][metric] for r in runs]
            agg[method][metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def fmt(d, metrics):
    return "  ".join(f"{m}={d[m]['mean']:.3f}±{d[m]['std']:.3f}" for m in metrics)


def main():
    print(f"Device: {DEVICE} | Seeds: {SEEDS}\n")
    results = {}

    # ---- Experimento principal: features completas ----
    print("=== EXPERIMENTO PRINCIPAL (features completas) ===")
    runs = [run_once(s, use_manual=True, canonicalize=False) for s in SEEDS]
    main_agg = aggregate(runs)
    results["main"] = main_agg

    clu = ["ari", "nmi", "cluster_acc"]
    sup = ["f1", "auc", "acc"]
    print("\n-- Não-supervisionado (clustering vs rótulos) --")
    print(f"  raw + KMeans      : {fmt(main_agg['raw_kmeans'], clu)}")
    print(f"  PCA(32) + KMeans  : {fmt(main_agg['pca_kmeans'], clu)}")
    print(f"  Contrastive Clust.: {fmt(main_agg['cc_cluster'], clu)}")
    print("\n-- Supervisionado (F1/AUC em teste) --")
    print(f"  LogReg (raw)      : {fmt(main_agg['logreg_raw'], sup)}")
    print(f"  RandomForest (raw): {fmt(main_agg['rf_raw'], sup)}")
    print(f"  Triplet + probe   : {fmt(main_agg['triplet_probe'], sup)}")
    print(f"  CC repr + probe   : {fmt(main_agg['cc_probe'], sup)}")

    # ---- Ablações (1 seed) para auditar atalhos ----
    print("\n=== ABLAÇÃO (seed 42): dependência de atalhos ===")
    configs = {
        "full": dict(use_manual=True, canonicalize=False),
        "no_manual": dict(use_manual=False, canonicalize=False),
        "canonical": dict(use_manual=True, canonicalize=True),
        "canon_no_manual": dict(use_manual=False, canonicalize=True),
    }
    abl = {}
    for name, cfg in configs.items():
        r = run_once(42, **cfg)
        abl[name] = r
        print(f"\n  [{name}]")
        print(f"    RandomForest (raw): {r['rf_raw']['f1']:.3f} F1 | {r['rf_raw']['auc']:.3f} AUC")
        print(f"    Triplet + probe   : {r['triplet_probe']['f1']:.3f} F1 | {r['triplet_probe']['auc']:.3f} AUC")
        print(f"    CC cluster-acc    : {r['cc_cluster']['cluster_acc']:.3f} | ARI {r['cc_cluster']['ari']:.3f}")
    results["ablation"] = abl

    with open(os.path.join(os.path.dirname(__file__), "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nResultados salvos em experiments/results.json")


if __name__ == "__main__":
    main()
