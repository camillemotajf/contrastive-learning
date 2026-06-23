# Label-noise detection — theory and honest framing

## 1. Noise vs hard sample vs outlier vs boundary case

These are easy to conflate and matter for interpretation:

- **Label noise** — the recorded label is wrong (the true class is the other one).
- **Hard sample** — the label is right, but the sample is intrinsically difficult
  (ambiguous features).
- **Outlier** — the sample is far from everything, regardless of label
  correctness.
- **Boundary case** — the sample sits near the true decision boundary; small
  feature changes flip the prediction, yet the label may be perfectly correct.

Every score in this project responds to **all four**. A high suspicion score is
therefore a *candidate for review*, never a proof of mislabeling. We use the
language "suspicion score", "local inconsistency", "candidate for review".

## 2. Why KNN is not an oracle

`knn_label_disagreement_score` measures the fraction of a sample's nearest
neighbours that carry a different label. A bot whose 18 of 20 neighbours are
human scores 0.90. But the same score arises for a genuine hard/boundary human
sample sitting inside a bot cluster. KNN measures **local label inconsistency in
a chosen feature space**, which correlates with noise but also with difficulty
and with the quality of that feature space. Hence we report it as a ranking
signal and compare KNN in raw space vs KNN in the CC embedding — the embedding
can sharpen or blur the neighbourhood.

## 3. Confident Learning

Confident Learning (Northcutt et al., 2021) gives a second, model-based opinion:

1. **Out-of-fold probabilities** — 5-fold cross-validation, so each sample is
   scored by a model that never trained on it (no self-confirmation of errors).
2. **Per-class confidence thresholds** — `threshold[c] = mean P(c)` over samples
   labeled `c`. A class-calibrated bar, not a flat 0.5.
3. **Confident joint** — a sample is flagged when the model *confidently*
   (above the class threshold) assigns it to a class **different** from its given
   label.

We expose a continuous score `P(other class) − P(given label)` for ranking and a
boolean flag for counts. Caveat: CL only finds noise the model *disagrees with*;
shared blind spots between the labelling system and the auditor model stay
invisible, so the flagged rate is a **lower bound**.

## 4. Why synthetic noise is necessary

On the real dataset we do **not** know which labels are wrong, so we cannot
measure whether any detector works — every "suspect" is unverifiable. Synthetic
noise (`src/noise/synthetic_noise.py`) flips a **known** fraction of labels and
returns a `noise_mask`. Now detection is a supervised problem with a ground
truth: we grade each score's ranking with ROC-AUC, AUPRC, precision@k, recall@k.
Conclusions about *which detector is better* come ONLY from this synthetic
setting; the real-data audit then *applies* the detectors but cannot validate
them.

## 5. Scores are rankings, not truth

All scores follow "higher = more suspicious" and are min-max normalised before
ensembling. The ensemble is a mean of normalised scores — a consensus ranking.
None is a calibrated probability of noise. We never claim "this label is
definitely wrong", only "these samples should be reviewed first".

## 6. Interpreting agreement between methods

When independent methods (KNN-raw, KNN-CC, Confident Learning, heuristics) put
the same sample near the top, our **confidence that it deserves review** goes up,
because they rely on different assumptions (local geometry vs a global model vs
hand rules). Agreement is corroboration, not proof. High-precision heuristics
(crawler/script User-Agents, unsubstituted `{{...}}` templates) act as external
anchors: if the ensemble's top suspects are rich in heuristic-flagged bots
labeled "human", that is strong circumstantial evidence — still to be confirmed
by hand, since we have no gold labels for real noise.
