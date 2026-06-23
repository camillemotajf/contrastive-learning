# Results interpretation

> Template prepared in advance. Fill each section from the saved `results/`
> artifacts after running the experiments. Keep the honest framing: suspicion
> scores, candidates for review, synthetic vs real noise.

## 0. Setup

- Source(s): _____ · seeds: _____ · split: _____ · feature dim (SVD): _____
- Date / commit: _____

## 1. Classification (Evaluation B) — `results/baseline/metrics.json`

| method | F1 | ROC-AUC | acc | precision | recall |
|---|---|---|---|---|---|
| LogReg | | | | | |
| RandomForest | | | | | |
| Triplet + probe | | | | | |
| SSL + probe | | | | | |
| CC + probe | | | | | |

- Best classifier: _____
- Does any contrastive method beat RF? _____

## 2. Clustering (Evaluation A)

| method | ARI | NMI | cluster-acc | #non-empty clusters | entropy |
|---|---|---|---|---|---|
| KMeans raw | | | | | |
| PCA + KMeans | | | | | |
| CC | | | | | |

## 4. Synthetic-noise detection — `results/synthetic_noise/summary_aggregate.csv`

ROC-AUC by method and noise rate (mean over seeds):

| score \ rate | 1% | 3% | 5% | 10% | 20% |
|---|---|---|---|---|---|
| confident_learning | | | | | |
| knn_raw | | | | | |
| knn_cc | | | | | |
| cc_cluster_entropy | | | | | |
| cc_view_instability | | | | | |
| centroid_relative_distance | | | | | |
| ensemble | | | | | |

- Best single detector: _____
- KNN-CC vs KNN-raw: _____  (answers question 4)
- Does the ensemble beat the best single score? _____

## 5. Real-data audit — `results/noise_audit/`

- Confident-Learning flag rate: _____ (lower bound on real noise)
- top-100 overlaps (KNN_raw∩KNN_cc, KNN_cc∩CL, ensemble∩heuristic): _____
- Qualitative read of `top_100_suspects.csv` (templates / crawler UAs labeled
  "human"?): _____

## 6. Verdict on the seven questions (Part 13)

1. CC + realistic HTTP augmentations — improves? _____
2. `num_clusters > 2` — useful subgroups? _____
3. CC embeddings beat raw features for the probe? _____
4. KNN-CC > KNN-raw for synthetic-noise detection? _____
5. CC scores help prioritise real suspects? _____
6. Does the combined ranking look interpretable? _____
7. Treat CC as classifier / clustering / audit tool? _____

## 7. Limitations

- No gold labels for real noise — real-data audit is unvalidated ranking.
- Confident Learning detects only model-visible noise (lower bound).
- TF-IDF is over headers only; request signal enters via manual features.
- Augmentations encode our assumptions about HTTP invariance.
