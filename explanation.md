# PatchCore Feature Extraction — Explanation
**MGCLS Anomaly Detection · University of the Western Cape · 2026**

---

## Context: Why PatchCore After BYOL-Based Methods

All previous methods (Moment Pooling, ECOD, COPOD, DeepSVDD) used the final
**512-dim BYOL embedding** — a single vector that summarises the whole radio
image. By the time information reaches this final layer, all spatial detail
about local morphology (jets, halos, unusual structures) has been compressed
away.

PatchCore goes back to the **raw `.png` images** and extracts features from
intermediate layers of a CNN, where spatial information is still preserved.
Two notebooks implement this, each going further than the previous.

---

## Notebook 1: `patchcore_extraction.ipynb` — Image-Level PatchCore

### What this notebook does

Implements PatchCore (Roth et al., CVPR 2022 — arXiv:2106.08265) using a
pretrained WideResNet50 as a frozen feature extractor on the raw radio images.
No training. No human labels. One forward pass per image.

---

### Step 1 — Image Discovery

**Problem encountered:** image filenames (`0.png`, `1.png`, ...) are integers,
not the `objid` strings (`img42_isl2234`). Two bugs were found and fixed:

| Bug | Cause | Fix |
|---|---|---|
| 0 images matched | Code compared filename stem directly to objid string | Use `protege_rank` column as the bridge — filenames = Protege rank of each object |
| `ValueError: invalid literal for int()` | Code tried `int('img42_isl2234')` | Remove all `int(oid)` casts — objids are strings throughout |

**Final mapping logic:**
```
image_map_int[int(p.stem)] = p          # filename integer → file path
image_map[row['objid']] = image_map_int[row['protege_rank']]  # objid → path
```

This relies on the fact that `0.png` = the object Protege ranked #0 (most anomalous),
`1.png` = rank #1, and so on — documented in the project README.

---

### Step 2 — Feature Extractor (`PatchCoreExtractor`)

A frozen WideResNet50 pretrained on ImageNet. Two intermediate layers are tapped:

| Layer | Output channels | What it captures |
|---|---|---|
| `layer2` | 256 | Local texture, edges, spatial patterns, unusual morphology |
| `layer3` | 512 | Higher-level semantic content, global structure |

Each layer's spatial feature map is collapsed to one vector per image using
**global average pooling** (AdaptiveAvgPool2d → (C, 1, 1)), then **L2-normalised**
independently. The two vectors are concatenated:

```
256 (layer2) + 512 (layer3) = 768-dim per image
```

Radio images are single-channel grayscale. The transform replicates the channel
three times (`T.Lambda(lambda x: x.repeat(3, 1, 1))`) before applying
ImageNet normalisation — required for pretrained weights.

**The network is never updated.** `requires_grad = False` on all parameters.

---

### Step 3 — Feature Extraction with Caching

Images are processed in batches of 32. Results are saved to
`data/patchcore_features.parquet`. If the file already exists on subsequent
runs, it loads instantly without re-running the CNN.

```
6161 images × 768 dims → patchcore_features.parquet
```

---

### Step 4 — Alignment with Evaluation Set

Some catalogue objects may not have corresponding images (either missing from
the image folders or not matched by the protege_rank mapping). The evaluation
is restricted to the intersection — objects with both a catalogue label and
an extracted feature vector. All metrics are computed on this common subset,
including the Protege baseline score, so the comparison is always fair.

---

### Step 5 — Whitening

After concatenation the 512-dim layer3 block contributes twice as many dimensions
as the 256-dim layer2 block. Without whitening, layer3 dominates every Euclidean
distance calculation simply by having more numbers.

`StandardScaler` sets mean=0 and std=1 per dimension across all objects.
After whitening, both layers contribute equally to nearest-neighbour distances.

---

### Step 6 — Memory Bank

All whitened 768-dim vectors are stored as the memory bank. At scoring time,
each object's anomaly score is its distance to its nearest neighbour in the bank:

```
Small distance → object looks like something we have seen before → normal
Large distance → object is unlike anything in the bank → anomalous
```

For larger datasets, a **greedy coreset** reduces the memory bank:
iteratively select the vector furthest from all already-selected vectors,
keeping the bank maximally spread out in feature space.

---

### Step 7 — k-NN Anomaly Scoring

Score = **mean distance to the k nearest neighbours** in the memory bank.
Using k > 1 is more robust than k = 1 — a single noisy neighbour cannot
dominate the score.

A **k sweep** tests k ∈ {1, 3, 5, 10}:

| k | Behaviour |
|---|---|
| 1 | Fastest, sensitive to noisy neighbours |
| 3 | Good robustness, default choice |
| 5–10 | More smoothing, slower |

Best k is selected automatically by ROC-AUC.

---

### Step 8 — Layer Ablation

Each layer is tested independently to verify that combining both helps:

| Variant | Dims | Source |
|---|---|---|
| Layer 2 only | 256 | Local structure |
| Layer 3 only | 512 | Semantic content |
| Combined (full PatchCore) | 768 | Both |

If combined > both individual layers: multi-scale representation adds genuine
information. If layer3 ≈ combined: the semantic layer already captures everything.

---

### Step 9 — Evaluation and Final Plot

Standard 4-metric table (same format as `evaluation.ipynb`):

| Metric | What it measures |
|---|---|
| ROC-AUC (4-5) | Probability anomaly ranks above normal object |
| PR-AUC (4-5) | Precision-recall at all thresholds |
| Recall@100 (4-5) | Fraction of true anomalies in top-100 candidates |
| Spearman (1-5) | Monotonic correlation with full 1–5 human scale |

All PatchCore variants are ranked by ROC-AUC. The **top-3 (excluding Protege)**
are plotted alongside the Protege baseline on a single cumulative discovery curve.

**Key comparison:** how far is the best PatchCore curve from the Protege curve?
That gap is the cost of not having human labels.

---

## Notebook 2: `patchcore_anomalib.ipynb` — Patch-Level PatchCore

### What this notebook does

Uses the **anomalib library** to implement PatchCore with true patch-level
feature aggregation — the key improvement over the custom implementation.

---

### The core difference: patch-level vs image-level

| Aspect | Custom (Notebook 1) | anomalib (Notebook 2) |
|---|---|---|
| Aggregation | Global average pool → **1 vector** per image | No pooling → **784 patch vectors** per image |
| Memory bank | ~6K image vectors | ~4.8M patch vectors → subsampled |
| Anomaly score | Mean k-NN of single image vector | **Max** k-NN across all image patches |
| Backbone | WideResNet50 only | WideResNet50 + ResNet18 (compared) |

**Why 784 patches?** The feature map at layer2 for a 224×224 image is
28×28 spatial positions = 784 locations. Each position is one patch vector.
Instead of averaging all 784 into one, anomalib keeps all 784 separately.

**Why max instead of mean for scoring?**
Image-level mean dilutes a single unusual region across 783 normal patches.
One anomalous local structure contributes only 1/784 of the final score.
Taking the **maximum** patch score means one unusual region anywhere in the
image is enough to flag the whole source — critical for detecting unusual
local radio morphology like compact jets or extended halos.

```
Custom:   score = mean_k_NN(single 768-dim image vector)
anomalib: score = max over 784 patches of mean_k_NN(patch vector)
```

---

### Feature extraction with anomalib

`TimmFeatureExtractor` from anomalib wraps any timm backbone and returns
the full spatial feature maps at requested layers — no pooling:

| Layer | Output shape | Patches |
|---|---|---|
| `layer2` | (N, 512, 28, 28) | 784 patches × 512-dim |
| `layer3` | (N, 1024, 14, 14) | pool to 28×28 → 784 patches × 1024-dim |

Both layers resampled to 28×28 and concatenated:
**784 patches × 1536-dim per image**

---

### Coreset subsampling

With 784 patches × 6161 images = ~4.8 million patch vectors, a full
nearest-neighbour search would be impractical. Greedy coreset subsampling
reduces this to 10% (~480K vectors) while covering the feature space evenly.

---

### Backbone comparison

Both **WideResNet50** (original paper backbone, wider = more capacity) and
**ResNet18** (faster, fewer parameters) are tested. This answers whether the
extra compute of WideResNet50 is justified for radio source anomaly detection.

---

### Final comparison

All methods from both PatchCore notebooks are combined with the Protege baseline
into one ranked table and discovery curve. Custom PatchCore scores from
`patchcore_features.parquet` are loaded automatically if available.

The top-3 methods (excluding Protege) are identified dynamically and plotted
alongside Protege — the summary figure for the weekly meeting.

---

## How to Read the Discovery Curve

```
Y axis: cumulative count of true anomalies (label ≥ 4) found so far
X axis: number of candidates a human has reviewed (rank position)

Steep early rise = good method (anomalies concentrated at top of ranked list)
Diagonal line    = random baseline (no skill whatsoever)
```

The gap between the Protege curve and the best PatchCore curve at any x position
is the number of additional true anomalies a human finds using Protege over
using PatchCore for the same review effort. Closing this gap is the goal.

---

## Summary of Improvements Made

| Notebook | Improvement over previous |
|---|---|
| `patchcore_extraction.ipynb` | Goes from BYOL final embedding → real CNN intermediate features from images |
| `patchcore_anomalib.ipynb` | Goes from image-level (1 vector) → patch-level (784 vectors), max scoring |

Each step preserves more of the original spatial information in the radio images
that successive compression stages (BYOL training → final embedding) had discarded.
