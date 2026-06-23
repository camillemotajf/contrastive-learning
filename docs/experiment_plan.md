# Experiment plan

All experiments split train/test on raw text **before** any learned transform,
fit the Preprocessor on **train only**, use fixed seeds, and save reproducible
JSON/CSV under `results/`.

| # | Experiment | Script | Input | Output | Metrics | Hypothesis tested |
|---|---|---|---|---|---|---|
| 1 | Baselines (clustering + classification) | `src.experiments.run_baselines` | `configs/baseline.yaml` | `results/baseline/metrics.json` | ARI, NMI, cluster-acc; F1, AUC, acc, P, R | RF is the strongest classifier; KMeans barely tracks labels |
| 2 | Label-efficiency curve | `src.experiments.run_label_efficiency` | `configs/baseline.yaml` | `results/baseline/label_efficiency.json` | F1, AUC vs label fraction | Do representation methods degrade more gracefully at low labels? |
| 3 | Contrastive Clustering (single run) | `src.experiments.run_contrastive_clustering` | `configs/contrastive_clustering.yaml` | `results/contrastive_clustering/<run>/*` | clustering + probe metrics; embeddings/probs/assignments | Does HTTP-augmented CC learn a useful representation? |
| 5 | Synthetic-noise detection | `src.experiments.run_synthetic_noise_eval` | `configs/synthetic_noise.yaml` | `results/synthetic_noise/*` | ROC-AUC, AUPRC, P@k, R@k | Which scores detect *known* noise? Does CC embedding help? |
| 6 | KNN raw vs KNN CC | `src.experiments.run_knn_noise` | `configs/noise_audit.yaml` | `results/noise_audit/knn_comparison.{json,csv}` | ROC-AUC, AUPRC, P@k | Is KNN-in-CC-embedding a better noise detector than KNN-raw? |
| 7 | Real-data noise audit | `src.experiments.run_noise_audit` | `configs/noise_audit.yaml` | `results/noise_audit/*` | rankings + overlaps (no ground truth) | Apply detectors to real labels; where do methods agree? |
| 8 | Figures | `src.experiments.make_figures` | `results/**` | `results/figures/*.pdf` | — | — |

## Recommended order

1. `run_baselines` — establishes the reference numbers.
2. `run_contrastive_clustering` — CC representation study (sweep `--num-clusters` manually if desired).
3. `run_synthetic_noise_eval` + `run_knn_noise` — validate the detectors.
4. `run_noise_audit` — apply detectors to the real labels.
5. `run_label_efficiency` (optional, heavier).
6. `make_figures`.

## Quick / smoke settings

Every script accepts `--subsample N` and (where relevant) `--epochs` /
`--cc-epochs` to run a fast version, e.g.:

```bash
python -m src.experiments.run_baselines --source outbrain --subsample 3000
python -m src.experiments.run_contrastive_clustering --subsample 3000 --epochs 5 --num-clusters 4
python -m src.experiments.run_synthetic_noise_eval --config configs/_smoke_noise.yaml
```

## Data

Sources discovered automatically from `../../data` via
`src.data.list_sources()` — currently `outbrain`, `taboola`, `tiktok` (10k bot +
10k unsafe each). Switch with `--source`. Labels: 0 = unsafe/human, 1 = bot,
read from the dumps' `decision`/file split.
