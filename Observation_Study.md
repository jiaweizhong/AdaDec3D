# Observation Study

**Project**: Adaptive Decoder Computation for Efficient 3D Medical Image Segmentation

> **Context**: This document is the execution guide for Stage 2 of the research pipeline defined in `Research_Proposal.md §7.3`.
> For scientific motivation and Go/No-Go criteria see `Research_Proposal.md §4, §5, §7.4`.
> For baseline training commands (E0/E1 needed for O5) see `Experiment-Design-Observation.md Part 3`.

---

# Two-Paper Strategy

This observation study is designed to support **two sequential publications**:

| Paper | Scope | Target Venue | Prerequisite |
| ----- | ----- | ------------ | ------------ |
| **Paper A** | Empirical analysis (O1–O11): where does decoder computation matter in 3D medical segmentation? | MIDL / MLMI / ISBI | O1–O5 pass Go criteria |
| **Paper B** | AdaDec3D method: adaptive decoder via difficulty-aware routing and ROI refinement | MICCAI / TMI | Paper A accepted |

**Paper A is self-contained.** Its hypothesis is:

> *The marginal benefit of additional decoder capacity is heterogeneous across
> anatomical structures and spatial regions, and may be predictable from a
> lightweight difficulty signal.*

O1–O5 and O9 test this hypothesis. The paper must report the observed selection
budget and recovered gain rather than assuming a 20/80 result in advance.

**Paper B builds on Paper A.** AdaDec3D is motivated by the finding that uniform decoder computation is wasteful, and is designed to allocate capacity only where O9 shows it is needed.

---

# Goal

Before proposing AdaDec3D,

we first investigate whether decoder computation is truly required everywhere.

Instead of assuming adaptive computation is beneficial,

we seek empirical evidence supporting or rejecting this hypothesis.

This study therefore answers the following scientific question.

> **Where does decoder computation actually matter?**

---

# Overall Workflow

```text
Reproduce Full Decoder (E0) + EffiDec3D (E1), using matched training budgets
                │
                ▼
      Save Predictions + Checkpoints
                │
                ▼
  O1: Error Distribution  O2: Difficulty Maps
                │
                ▼
  O3: Difficulty vs Error Correlation
                │
                ▼
  O4: Organ-wise Difficulty
                │
                ▼
  O5: Decoder Gain Analysis  ◄── Critical Go/No-Go gate
                │
    ┌───────────┼──────────────┐
    ▼           ▼              ▼
  O6: Difficulty  O7: Cross-   O8: Backbone
  Evolution     Dataset       Consistency
                │
                ▼
  O9: Pareto Curve  ◄── Headline finding for Paper A
                │
                ▼
  O10: Organ Size vs Difficulty
                │
                ▼
  O11: Routing Signal Comparison
                │
                ▼
        Paper A Submission
```

---

# O1 Prediction Error Distribution

## Research Question

Are segmentation errors uniformly distributed?

---

## Motivation

If prediction errors are uniformly distributed,

uniform decoder computation is reasonable.

Otherwise,

adaptive computation may be beneficial.

---

## Input

Validation Dataset

EffiDec3D Prediction

Ground Truth

---

## Output

Error Map

```python
error = prediction != gt
```

---

## Statistics

Voxel Error Rate

Organ-wise Error

Boundary Error

Interior Error

---

## Figure

Figure O1

```text
Image

Prediction

Ground Truth

Error Map
```

---

## Expected Result

Errors mainly appear around

* organ boundary

* thin structures

* small organs

---

## Reviewer Question

Are errors spatially clustered?

---

# O2 Difficulty Distribution

## Research Question

Where are difficult voxels?

---

## Motivation

Adaptive computation requires

difficulty estimation.

---

## Candidate Difficulty Signals

Entropy

Confidence

Feature Variance

MC Dropout

Boundary Probability

---

## Default Implementation

Entropy

```python
entropy = -(p * torch.log(p + 1e-8)).sum(1)
```

---

## Output

Difficulty Map

---

## Statistics

Difficulty Histogram

Difficulty Percentile

Difficulty Heatmap

---

## Figure

Difficulty Map Overlay

---

## Reviewer Question

Does difficulty occupy only a small percentage of voxels?

---

# O3 Difficulty vs Error

## Research Question

Does predicted difficulty really indicate segmentation difficulty?

---

## Input

Difficulty Map

Prediction Error Map

---

## Statistics

Pearson Correlation

Spearman Correlation

Calibration Curve

---

## Code

```python
from scipy.stats import pearsonr, spearmanr

flat_diff = difficulty_map.flatten().numpy()
flat_err  = error_map.float().flatten().numpy()

r_p, _ = pearsonr(flat_diff, flat_err)
r_s, _ = spearmanr(flat_diff, flat_err)
print(f"Pearson r={r_p:.3f}  Spearman r={r_s:.3f}")
```

---

## Figure

```text
Prediction Error

↑

│

└──────────────► Difficulty
```

---

## Expected

Positive Correlation

---

## Reviewer Question

Why is Difficulty a reasonable routing signal?

---

# O4 Difficulty vs Anatomy

## Research Question

Which organs are difficult?

---

## Statistics

For each organ

Mean Difficulty

Dice

HD95

Boundary Difficulty

---

## Code

```python
organ_names = ["Aorta","Gallbladder","Spleen","Left Kidney","Right Kidney",
               "Liver","Stomach","Aorta","IVC","Portal Vein","Pancreas",
               "Right Adrenal","Left Adrenal"]
for i, name in enumerate(organ_names):
    mask = (lbl == i + 1)
    organ_diff = difficulty_map[mask].mean().item()
    print(f"{name:20s}  mean_difficulty={organ_diff:.4f}")
```

---

## Figure

Organ-wise Bar Plot

---

## Expected

Pancreas

Adrenal

Esophagus

Highest Difficulty

---

## Reviewer Question

Do different organs require different decoder capacity?

---

# O5 Decoder Gain Analysis

## Research Question

Where does a stronger decoder actually help?

---

## Motivation

This experiment is the **critical Go/No-Go gate**.

It tests the opportunity for adaptive decoder computation by comparing

EffiDec3D (E1) against a full-capacity decoder (E0).

---

## Input

E0 predictions (Full decoder)

E1 predictions (EffiDec3D)

Ground Truth

Difficulty maps from O2

---

## Code

```python
import torch
from scipy.stats import pearsonr

# Report both directions. Net benefit is positive transitions minus regressions.
positive = ((pred_full == lbl) & (pred_effi != lbl)).float()
negative = ((pred_full != lbl) & (pred_effi == lbl)).float()
gain = positive - negative

# bin by entropy quantile
n_bins = 10
percentiles = torch.quantile(entropy_map.flatten(), torch.linspace(0, 1, n_bins + 1))
x, y = [], []
for k in range(n_bins):
    lo, hi = percentiles[k], percentiles[k + 1]
    mask = (entropy_map >= lo) & (entropy_map < hi)
    if mask.sum() > 0:
        x.append(entropy_map[mask].mean().item())
        y.append(gain[mask].mean().item())

r, p = pearsonr(x, y)
print(f"{'GO  ✓' if r > 0.50 else 'NO-GO ✗'}: Pearson r={r:.3f}  p={p:.4f}")
```

---

## Go Criterion

Report positive transitions, negative transitions, and net benefit separately by
subject, organ, and physical-distance boundary band. Proceed only when a
deployable signal predicts held-out net benefit better than matched random and
boundary controls with a subject-bootstrap 95% confidence interval. Correlation
over pooled voxels or bins is descriptive, not a significance test.

---

## Figure

```text
Difficulty Map  +  Gain Map  →  Overlay
```

---

## Expected

Higher decoder capacity mainly benefits

* difficult voxels (high entropy)

* boundaries

* small organs

---

## Reviewer Question

Is adaptive decoder computation actually necessary?

---

# O6 Difficulty Evolution During Training

## Research Question

Does prediction difficulty decrease over training, and does it concentrate near boundaries/hard organs as training progresses?

---

## Motivation

If difficulty is transient and disappears as the model trains, adaptive routing offers little permanent benefit.

If difficulty persists and localizes, it is a stable routing signal.

---

## Setup

Save model checkpoints at epochs {5, 10, 20, 30, 50} during E0/E1 training.

---

## Code

```python
# Add to trainer: save checkpoints at milestones
MILESTONES = [5, 10, 20, 30, 50]

for epoch in MILESTONES:
    ckpt_path = f"checkpoints/effidec3d_epoch{epoch:03d}.pt"
    model = load_model("3DUXNET_EffiDec3D", ckpt_path)
    entropy_maps = []
    for img, lbl in val_loader:
        with torch.no_grad():
            logits = model(img.cuda())
        prob = torch.softmax(logits, dim=1)
        ent  = -(prob * torch.log(prob + 1e-8)).sum(1)
        entropy_maps.append(ent.cpu())
    mean_ent = torch.stack(entropy_maps).mean()
    print(f"Epoch {epoch:3d}  mean_entropy={mean_ent:.4f}")
```

---

## Figure

Line plot: mean entropy vs training epoch

Spatial map: entropy at epochs 5, 20, 50 (side-by-side)

---

## Expected

Mean entropy decreases, but residual high-entropy voxels stabilize at boundaries/hard organs by epoch 30.

---

## Paper A Role

Supports the claim that difficulty is a stable, persistent signal — not transient noise.

---

# O7 Cross-Dataset Consistency

## Research Question

Do the O1–O5 findings replicate on FeTA (fetal brain MRI)?

---

## Motivation

If decoder gain concentrates on difficult voxels across both CT (BTCV) and MRI (FeTA),

the finding is dataset-agnostic and generalizes beyond one modality.

---

## Setup

Repeat O1–O5 pipeline on FeTA validation set using EffiDec3D trained on FeTA.

---

## Code

```python
# run identical O1-O5 analysis, but with FeTA data
feta_loader = get_loader_feta(data_dir, batch_size=1, num_workers=4)
model_feta = load_model("3DUXNET_EffiDec3D", ckpt_path="checkpoints/effidec3d_feta.pt")
# then call same error/difficulty/gain analysis functions
```

---

## Statistics

Pearson r (O5 equivalent) on FeTA

Organ-wise difficulty bar plot for fetal brain structures

---

## Expected

r > 0.50 on FeTA, confirming the relationship is modality-agnostic.

---

## Paper A Role

Cross-dataset replication is a key criterion for MIDL/MLMI reviewers.

---

# O8 Backbone Consistency

## Research Question

Does the O5 result hold when the backbone changes from UXNET to SwinUNETR?

---

## Motivation

If the decoder gain–difficulty correlation depends on the specific backbone,

AdaDec3D cannot claim general applicability.

---

## Setup

Train SwinUNETR_EffiDec3D on BTCV.

Run O5 analysis using SwinUNETR predictions.

---

## Code

```python
model_swin = load_model("SwinUNETR_EffiDec3D",
                         ckpt_path="checkpoints/swin_effidec3d.pt")
# identical O5 analysis
gain_swin = ((pred_swin_full == lbl) & (pred_swin_effi != lbl)).float()
r_swin, _ = pearsonr(x_swin, y_swin)
print(f"SwinUNETR backbone: r={r_swin:.3f}")
```

---

## Expected

r > 0.45 for SwinUNETR, confirming backbone-agnostic signal.

---

## Paper A Role

Backbone consistency strengthens the claim that the finding is architectural rather than model-specific.

---

# O9 Selective-Allocation Opportunity

## Research Question

At fixed selection budgets, how much positive decoder transition can each
test-time signal recover, and does it outperform matched random selection?

---

## Motivation

This is an opportunity analysis, not a demonstration of computational savings.
Full-decoder predictions may depend on dense surrounding computation. Paper B
must separately demonstrate that contextual region refinement realizes this
opportunity with lower executed cost.

---

## Code

```python
import numpy as np

# Canonical executable implementation: Experiment-Design-Observation.md O9.
# Compute recovery per subject at 5/10/20/30/50% union-foreground budgets.
# Compare entropy, confidence, boundary, foreground, organ-size, 100 random
# selections, and an analysis-only oracle positive-transition ranking.
# Bootstrap subjects for 95% confidence intervals; never pool all scan voxels.
```

---

## Go Criterion for Paper A

Entropy or another deployable signal must outperform matched random selection at
10–30% budgets with a subject-bootstrap 95% confidence interval. The observed
budget/recovery pair is reported without imposing a 20/80 threshold.

---

## Figure

Recovered positive transitions versus selection budget, with random confidence
band and boundary/oracle reference curves.

---

## Expected

No numerical result is assumed before the experiment.

---

## Paper A Headline

> *The marginal utility of decoder capacity is spatially heterogeneous, and a
> lightweight held-out signal identifies beneficial regions better than matched
> random and anatomical heuristics.*

---

# O10 Organ Size vs Difficulty

## Research Question

Is difficulty driven by organ size, or is it independent?

---

## Motivation

Reviewers will ask: "Is your difficulty signal just a proxy for small organs?"

If small organs are uniformly difficult but large organs are not, difficulty is confounded by size.

If difficulty is heterogeneous even within organ types, it is a richer signal than size alone.

---

## Code

```python
# organ volume (proxy for size) vs mean difficulty
for i, name in enumerate(organ_names):
    mask = (lbl == i + 1)
    size  = mask.float().sum().item()
    diff  = difficulty_map[mask].mean().item()
    print(f"{name:20s}  size={size:8.0f}  difficulty={diff:.4f}")

# scatter plot: size vs difficulty across organs
from scipy.stats import spearmanr
r_size, _ = spearmanr(sizes, difficulties)
print(f"Size vs Difficulty  Spearman r={r_size:.3f}")
```

---

## Figure

Scatter plot: organ volume vs mean difficulty (with organ labels)

---

## Expected

Weak-to-moderate negative correlation (smaller organs are harder on average),

but high residual variance: some large organs (stomach, liver boundary) also have high difficulty.

---

## Paper A Role

Shows difficulty is not merely a size proxy, justifying entropy over a simple size-based routing rule.

---

# O11 Routing Signal Comparison

## Motivation

Difficulty estimation should not rely on a single signal.

The best routing signal for AdaDec3D should balance correlation with segmentation error,

compute overhead, and stability across datasets.

---

## Signals

| Signal | Description |
| ------ | ----------- |
| Entropy | `-(p log p).sum(1)` over softmax output |
| Confidence | `1 - max(p)` |
| Feature Variance | variance of decoder feature maps |
| MC Dropout | variance over T stochastic forward passes |
| Boundary Probability | distance-to-boundary probability map |

---

## Evaluation

Correlation with segmentation error (Pearson r)

Inference latency overhead (ms/volume)

GPU memory overhead (MB)

Stability across datasets (BTCV vs FeTA r difference)

---

## Table

| Signal | Corr (BTCV) | Corr (FeTA) | Latency (ms) | Memory (MB) |
| ------ | ----------- | ----------- | ------------ | ----------- |
| Entropy | | | | |
| Confidence | | | | |
| Feature Var | | | | |
| MC Dropout | | | | |
| Boundary | | | | |

---

## Goal

Select the routing signal for AdaDec3D.

Expected winner: Entropy (highest correlation, near-zero overhead).

---

# Go / No-Go Decision

## Minimum criteria to proceed to Paper A submission

| Observation | Criterion | Status |
| ----------- | --------- | ------ |
| O3 | Pearson r(difficulty, error) > 0.40 | ☐ |
| O5 | Held-out signal predicts net benefit beyond random/boundary controls | ☐ |
| O9 | Deployable ranking beats random at 10–30% budgets with subject-level CI | ☐ |
| O2 | Report observed difficulty concentration without a preset cutoff | ☐ |

**All four must pass** to proceed to Paper A write-up.

## Additional criteria to proceed to Paper B (AdaDec3D)

| Observation | Criterion | Status |
| ----------- | --------- | ------ |
| O7 | FeTA replication: r > 0.40 | ☐ |
| O8 | SwinUNETR backbone: r > 0.45 | ☐ |
| O11 | Entropy is best or tied-best routing signal | ☐ |

**All three must pass** to justify AdaDec3D design choices.

---

# Deliverables

## Notebooks

| Notebook | Observations Covered |
| -------- | -------------------- |
| `observe_error.ipynb` | O1 |
| `observe_difficulty.ipynb` | O2, O10 |
| `observe_correlation.ipynb` | O3, O4 |
| `observe_decoder_gain.ipynb` | O5, O9 |
| `observe_evolution.ipynb` | O6 |
| `observe_crossdataset.ipynb` | O7, O8 |
| `observe_routing_signal.ipynb` | O11 |

---

# Final Outputs

## Figures

| Figure | Content | Paper |
| ------ | ------- | ----- |
| Fig 1 | Error distribution map (O1) | A |
| Fig 2 | Difficulty heatmap overlay (O2) | A |
| Fig 3 | Difficulty–error scatter (O3) | A |
| Fig 4 | Organ-wise difficulty bar plot (O4) | A |
| Fig 5 | Decoder gain vs difficulty (O5) | A |
| Fig 6 | Difficulty evolution over training (O6) | A |
| Fig 7 | Pareto curve: cumulative gain vs uncertainty % (O9) | A (headline) |
| Fig 8 | Organ size vs difficulty scatter (O10) | A |
| Fig 9 | Routing signal comparison table (O11) | B |

## Tables

| Table | Content |
| ----- | ------- |
| T1 | Organ-wise statistics (difficulty, Dice, HD95) |
| T2 | Cross-dataset replication (O7) |
| T3 | Backbone consistency (O8) |
| T4 | Routing signal comparison (O11) |

---

# Expected Scientific Discovery

If all observations are validated,

this study supports the following conclusion.

> Decoder redundancy in efficient 3D medical image segmentation is **spatially heterogeneous**.
> A held-out lightweight signal identifies regions with positive marginal decoder
> utility better than matched random and anatomical controls, while explicitly
> accounting for decoder regressions and confidently missed structures.
> This heterogeneity is consistent across datasets (BTCV, FeTA) and backbone architectures (UXNET, SwinUNETR).

This conclusion is the core contribution of **Paper A** and directly motivates the AdaDec3D framework in **Paper B**.
