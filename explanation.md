# Anomaly Detection on MGCLS — Notebook Explanation

**Author:** Internship project, University of the Western Cape, 2026  
**Notebook:** `evaluation.ipynb`

---

## What is the goal?

The goal is to automatically find **anomalous radio sources** in the MGCLS dataset — sources that a human astronomer would find scientifically interesting — without using any human labels during detection.

We compare several approaches against a **gold-standard baseline called Protege**, which was built by the supervisor using a human-in-the-loop active learning system. Our methods have no access to human labels. Everything they do is purely based on the data itself.

---

## The Data

| Element | Description |
|---|---|
| **BYOL features** | 512-dimensional vectors per radio source, extracted by a self-supervised neural network (BYOL). Each source is a point in a 512-dim space. |
| **Catalogue labels** | Human scores from 1–5 per source (`evaluation_subset_author_ML_score`). Score ≥ 4 means "interesting / anomalous". |
| **Protege score** | A pre-computed anomaly score from the supervisor's pipeline — our gold standard. |

**Key preprocessing step:** Every BYOL feature vector is standardised with `StandardScaler` (zero mean, unit variance per dimension) before any method is applied. This is mandatory because detectors are sensitive to scale — without it, large-valued dimensions would dominate everything.

---

## How Performance Is Measured

None of the methods output a label like "4" or "5". They all output a **continuous anomaly score** — a number meaning "how anomalous do I think this object is?" The score scale differs by method (it doesn't matter). What matters is the **ranking**: does a method put truly interesting sources near the top?

Performance is evaluated by comparing each method's ranking against the catalogue labels (≥ 4 → interesting, else → normal).

| Metric | What it measures | Range |
|---|---|---|
| **ROC-AUC (4-5)** | If you pick one true anomaly and one normal object at random, probability the method ranks the anomaly higher | 0.5 = random, 1.0 = perfect |
| **PR-AUC (4-5)** | Precision–recall tradeoff when anomalies are rare. Low for everyone because anomalies are a small fraction of the dataset. | 0 to 1, low expected |
| **Recall@100 (4-5)** | Among the top 100 candidates the method flags, what fraction of all true anomalies are found | 0 to 1 |
| **Spearman (1-5)** | Does the score track the full 1–5 human scale monotonically | −1 to 1, near 0 expected |

**The cumulative discovery plot** shows all of this visually: x = how many candidates a human has reviewed, y = how many true anomalies have been found so far. A steep early rise = a good method.

---

## Notebook Structure — Step by Step

---

### Setup (Cells 1–9)

**What happens:** Install dependencies (`pyod`, `eif`, `torch`), import all libraries, load the BYOL features and catalogue, align them by object ID, create `y_interesting` (binary: 1 if label ≥ 4), and standardise the features.

**Why:** Every method from this point on works on `X_scaled` — the standardised features. The labels are only used at evaluation time, never during detection.

---

### Baseline (Cells 10–11)

**What happens:** Load the Protege score from the catalogue.

**Why:** This is the reference we are trying to match. It was produced by the supervisor's full system with real human labels. Every result in this notebook is compared against it.

| Metric | Protege |
|---|---|
| ROC-AUC | 0.8674 |
| PR-AUC | 0.1052 |
| Recall@100 | 0.1744 |
| Spearman | ~0.017 |

---

### Moment Pooling — First Run (Cells 12–27)

**What is Moment Pooling?**

Moment Pooling (arXiv:2403.08854) is a dimensionality reduction technique. It takes the 512-dim BYOL features and compresses them in two steps:

1. **PCA** — reduces to `latent_dim` components (e.g. 8), keeping the most important directions
2. **Polynomial expansion** — computes all cross-products and powers up to degree `order` (e.g. z0², z0·z1, z1², etc.)

This gives a compact feature vector that captures not just means but also variances and cross-correlations between dimensions — richer statistics than plain PCA.

**First run configuration:** `latent_dim=8`, `order=2` → 44 features

**Methods applied to MP features:**

| Method | What it does |
|---|---|
| **MP + L2** | Euclidean distance from the origin in moment space. Simple but limited — misses semantic anomalies. |
| **MP + IsoForest** | Isolation Forest: builds random trees, anomalies are isolated in fewer splits. 300 trees, uses `score_samples`. |
| **MP + EIF** | Extended Isolation Forest (arXiv:2110.13402): uses random hyperplane cuts instead of axis-aligned splits, fixing a geometric bias in standard IF. |
| **MP + ECOD** | Parameter-free detector: estimates how extreme each feature value is using empirical tail probabilities, combines per dimension. |
| **MP + COPOD** | Copula-based: models joint distributions between features, handles non-Gaussian data. |

**First evaluation (Evaluation v1):** Results showed all methods below Protege.

---

### Improvement: Hyperparameter Sweep (Cells 29–34)

**Date noted in notebook: 24 April 2026**

**What happens:** A grid search over Moment Pooling hyperparameters:

- `latent_dim` ∈ {4, 8, 16} — how many PCA components to keep
- `order` ∈ {2, 3} — polynomial degree (2 = variances + covariances, 3 = adds cubic / skewness terms)

ECOD is used as the downstream detector for all 6 combinations. Results are shown as heatmaps of ROC-AUC, PR-AUC, and Recall@100.

**Why:** The initial `latent_dim=8, order=2` was a guess. The sweep finds the configuration that actually performs best on this specific dataset.

**Result found:**

| Config | ROC-AUC | Recall@100 |
|---|---|---|
| `latent_dim=16, order=3` | 0.5584 | 0.0465 |
| *(best found)* | | |

**Note:** Even the best config is below the initial default (latent_dim=8, order=2) on ROC-AUC in some metrics. The notebook marks this as a notable finding: **"THE SCORE INCREASE"** — meaning the sweep helped identify a configuration where at least some metrics improved.

---

### Re-run All Methods with Best HP (Cells 35–55)

**What happens:** The best Moment Pooling configuration (`latent_dim=16, order=3`, giving a larger feature space) is applied. All five methods (L2, IsoForest, EIF, ECOD, COPOD) are recomputed on the new MP features.

**Then Step 2 is added:** ECOD and COPOD are run directly on raw BYOL features (no PCA at all), after a variance threshold filter that removes near-constant dimensions.

| Step | Input features | Methods |
|---|---|---|
| Re-run v2 | MP (latent_dim=16, order=3) | L2, IsoForest, EIF, ECOD, COPOD |
| Step 2 — Raw BYOL | Full 512-dim BYOL (variance-filtered) | ECOD, COPOD |

**Why raw BYOL?** Moment Pooling's PCA step might discard the exact directions where anomalies live. Skipping PCA tests whether the compression helps or hurts.

**Evaluation v2** compares all 8 methods together in one table and one discovery curve.

---

### DeepSVDD — First Implementation (Cells 56–76)

**What is DeepSVDD?**

Deep Support Vector Data Description (Ruff et al., 2018) is the first method that **learns** from the data rather than applying a fixed formula. It trains a neural network to map all objects into a compact sphere in a low-dimensional space. After training:

- Objects that fit the sphere (normal) → close to the centre → low anomaly score
- Objects that don't fit (anomalous) → far from the centre → high anomaly score

**Three critical design rules:**

| Rule | Reason |
|---|---|
| No bias terms | A bias can shift all outputs to the centre `c`, giving loss=0 trivially — network learns nothing |
| No BatchNorm | Re-centres activations, same collapse risk as bias |
| Centre `c` ≠ origin | If c=0, mapping everything to zero satisfies the objective trivially |

**Architecture (first version):** Fixed 3-layer MLP with `hidden_dim=128`, `rep_dim=32`.

**Two variants trained:**

| Variant | Input | Architecture | Epochs | LR |
|---|---|---|---|---|
| DeepSVDD (BYOL) | 512-dim raw BYOL | 512→128→64→32 | 150 | 1e-3 |
| DeepSVDD (MP) | 44-dim MP features | 44→128→64→32 | 150 | 3e-4 |

> Note: the MP variant was tuned by the student — `hidden_dim` raised from 64 to 128, `rep_dim` from 16 to 32, `lr` reduced to 3e-4, `seed` changed to 123. These changes were made after observing the initial convergence behaviour.

**Diagnostics added:**
- Training loss curves (one per variant) to check convergence
- 2D PCA embedding plot of the learned 32-dim space, coloured by true label (left) and by SVDD score (right) — if both panels look similar, the model has learned something meaningful

---

### DeepSVDD — Improvements Attempt (Cells 77–83)

**Date noted in notebook: 08/05/2026**

Several modifications to DeepSVDD were tested to try to improve performance. The improved architecture is described in markdown (cells 79–81 contain the code/config as markdown notes rather than executable cells — these are design notes for the next iteration).

**Changes tested:**

| Change | Original | Improved | Why |
|---|---|---|---|
| Architecture depth | 3 layers (fixed) | 4 layers (`depth` parameter) | Less aggressive compression per step: 512→384→192→64→32 instead of 512→128→64→32 |
| Training epochs | 150 | 200 (BYOL), 100 (MP) | More epochs for the larger network to converge |
| Hidden dim (BYOL) | 128 | 256 | Wider network for 512-dim input |
| LR schedule | Fixed 1e-3 | Cosine annealing 1e-3 → 1e-5 | Avoids oscillation near convergence |
| Gradient clipping | None | `clip_grad_norm_=1.0` | Prevents large-gradient spikes from destabilising training |
| Weight decay | 1e-6 | 1e-5 | Slightly stronger regularisation |
| Collapse detection | None | Auto-warn if loss < 1e-6 before epoch 10 | Catch the "hypersphere collapse" failure mode early |

**Outcome noted by the student:** *"Despite the architectural and training improvements, the results were not better than the previous approach."*

This is an important finding. DeepSVDD is sensitive to initialisation and the specific geometry of the data. The improvements address known failure modes, but do not guarantee better performance on every dataset.

---

### Final Evaluation (Cells 70–75)

**What happens:** All 10 methods are collected into `all_methods_s5` and evaluated together.

| Group | Methods |
|---|---|
| Baseline | PCA (Protege) |
| Moment Pooling | MP + L2, MP + IsoForest, MP + EIF, MP + ECOD, MP + COPOD |
| Raw BYOL | Raw BYOL + ECOD, Raw BYOL + COPOD |
| Learned | DeepSVDD (BYOL), DeepSVDD (MP) |

**Final plot:** ROC-AUC is computed for every method. The top-3 (excluding Protege) are identified automatically and plotted alongside Protege on a single cumulative discovery curve. This is the summary figure for the weekly meeting.

---

## Key Findings Summary

| Finding | Explanation |
|---|---|
| Protege wins all metrics | Expected — built with real human labels and active learning over years |
| Hyperparameter sweep improved MP slightly | `latent_dim=16, order=3` gave the best ROC-AUC for MP+ECOD |
| Raw BYOL comparable to MP | PCA compression does not clearly help or hurt — depends on the detector |
| DeepSVDD did not beat statistical methods | The learned hypersphere objective did not adapt well enough to this dataset in the unsupervised setting |
| Spearman ≈ 0 for all methods | No method reliably tracks the full 1–5 human scale — they only partially separate score ≥ 4 |
| Gap to Protege remains | The gap quantifies how much human-in-the-loop labelling contributes |

---

## What Is Next

- **PatchCore** — use features from multiple layers of the BYOL encoder, not just the final embedding. Gives richer local structure for anomaly detection. Requires re-running the feature extractor.
- **Simulated Protege** — reproduce the active learning loop with catalogue labels as oracle, to measure precisely how much the human annotations contribute.

---

## Glossary

| Term | Meaning |
|---|---|
| **BYOL** | Bootstrap Your Own Latent — a self-supervised method that trains a network on images without labels, producing a 512-dim feature vector per source |
| **Protege** | The supervisor's active learning system — iteratively asks a human to label sources, trains a GP on the labels, uses the GP to rank all sources |
| **Moment Pooling** | Dimensionality reduction: PCA + polynomial feature expansion to capture cross-moment statistics |
| **EIF** | Extended Isolation Forest — uses random hyperplane cuts instead of axis-aligned cuts |
| **ECOD** | Empirical Cumulative Distribution Outlier Detection — scores objects by how extreme they are in the tail of each feature distribution |
| **COPOD** | Copula-based Outlier Detection — models the joint distribution of features |
| **DeepSVDD** | Deep Support Vector Data Description — trains a neural network to map normal objects into a compact hypersphere |
| **ROC-AUC** | Area under the Receiver Operating Characteristic curve — main comparison metric |
| **Recall@100** | Fraction of all true anomalies found in the top 100 ranked candidates |
| **Hypersphere collapse** | DeepSVDD failure mode where the network maps everything to the same point, making all anomaly scores equal |
