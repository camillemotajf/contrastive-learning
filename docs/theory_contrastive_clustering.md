# Contrastive Clustering — theory and design choices

## 1. What is contrastive learning?

Contrastive learning trains a representation by **pulling together** two *views*
of the same sample and **pushing apart** views of different samples. No class
labels are required — the supervision comes from the pairing "these two views
are the same underlying object". Formally, with L2-normalised projections, the
InfoNCE / NT-Xent loss maximises the agreement of a positive pair relative to all
negatives in the batch.

The whole method therefore rests on **what an augmentation is**: an augmentation
defines the *invariance* we want the representation to have. Good augmentations
produce views that are genuinely the same object seen differently; bad ones
produce views that are either trivially identical (nothing to learn) or
semantically different (the model learns a wrong invariance).

## 2. What is Contrastive Clustering (CC)?

CC (Li et al., 2021) adds a second head so that representation learning and
clustering happen jointly:

- **Instance head** — projects each sample to a vector; the instance loss is
  NT-Xent over the *rows* (samples) of a batch. This shapes the embedding.
- **Cluster head** — a softmax over `num_clusters`; the cluster loss is NT-Xent
  over the *columns* (clusters), treating each cluster's assignment vector across
  the batch as something that must be consistent between the two views.

A crucial detail is the **entropy regulariser** on the marginal cluster
distribution. Without it the cluster head can collapse — assign everything to one
cluster — and still minimise the contrastive term. We add `-H(P)` (scaled by
`lambda_entropy`) to force balanced cluster usage. This is implemented in
`src/losses/contrastive_losses.ClusterLoss`.

## 3. Instance loss vs cluster loss

| | operates on | shapes | answers |
|---|---|---|---|
| Instance loss | rows (samples) | the embedding geometry | "are two views of a sample close?" |
| Cluster loss | columns (clusters) | the partition | "is the cluster structure view-consistent?" |

They are complementary: the instance loss gives a useful embedding even if the
clustering is poor; the cluster loss gives a partition that may or may not align
with the labels.

## 4. Why SVD-space augmentations are weak

The original implementation augmented the **already-compressed 64-d SVD vector**
with column masking + Gaussian noise. Those perturbations have no HTTP meaning:
randomly zeroing an SVD coordinate is not "a plausible variation of this
request". The two views were therefore not realistic variants, the positive pair
was nearly trivial, and the instance head learned little. This is the leading
hypothesis for why CC underperformed simple baselines.

## 5. Why HTTP-level augmentations are more appropriate

`src/data/http_augmentations.py` augments at the **text / JSON level, before**
TF-IDF/SVD: reorder header keys, drop optional/unstable headers, mask values
while keeping keys, vary header-key case (HTTP keys are case-insensitive), jitter
JSON spacing, partially blur User-Agent versions, mask request values while
preserving structure. Both views are then pushed through the **same fitted**
vectoriser/SVD/scaler (transform-only — no refitting, no leakage). Now the two
views are genuinely the same request serialised differently, which is the
invariance we actually want for bot/human traffic.

Named configs (`light/medium/strong`, plus the optional pairs `ua_preserving` vs
`ua_masking`, `template_preserving` vs `template_masking`) let us optionally test
whether CC depends on a specific shortcut signal by changing `augmentation_type`.

## 6. Why test num_clusters > 2

The final task is binary (bot vs human), but the data plausibly contains several
natural **subtypes**: explicit crawler, templated bot (`{{...}}`), silent bot,
desktop human, mobile human, etc. Forcing `num_clusters = 2` can blur these. By
sweeping `num_clusters ∈ {2, 4, 8, 16}` we test whether finer clustering
discovers coherent subgroups (a better embedding for the probe, more interpretable
clusters) even though the labels remain binary.

## 7. Why evaluate CC on three separate axes

A method can fail as a classifier yet be useful elsewhere. We evaluate CC as:

- **A. Clustering** — ARI / NMI / Hungarian accuracy + cluster size/entropy.
- **B. Representation** — a logistic *linear probe* on frozen embeddings, against
  raw-feature LogReg/RF and Triplet/SSL probes.
- **C. Noise-audit tool** — not a classifier at all, but a generator of
  per-sample suspicion scores (cluster entropy, view instability, cluster–label
  mismatch, centroid distances, KNN-in-embedding).

Separating the axes keeps the comparison honest: if Random Forest wins axis B, we
report that, and ask whether CC still earns its place on axis C.
