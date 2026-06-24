# Observation Study

**Project**: Adaptive Decoder Computation for Efficient 3D Medical Image Segmentation

> **Context**: This document is the execution guide for Stage 2 of the research pipeline defined in `1_Research_Proposal.md §7.3`.
> For scientific motivation and Go/No-Go criteria see `1_Research_Proposal.md §4, §5, §7.4`.
> For baseline training commands (E0/E1 needed for O5) see `Experiment-Design.md Part 3`.

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
Reproduce EffiDec3D
        │
        ▼
Save Predictions
        │
        ▼
Generate Difficulty Maps
        │
        ▼
Generate Error Maps
        │
        ▼
Statistical Analysis
        │
        ▼
Scientific Insight
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
entropy = -(p*log(p)).sum(1)
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

Difficulty

Prediction Error

---

## Statistics

Pearson Correlation

Spearman Correlation

Calibration Curve

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

This experiment validates the necessity of adaptive decoder computation.

---

## Input

EffiDec3D

Large Decoder

---

## Generate

Improvement Map

```python
gain = full_decoder - effidec
```

---

## Statistics

Gain Histogram

Gain vs Difficulty

Gain vs Organ

Gain vs Boundary

---

## Figure

```text
Difficulty Map

+

Gain Map

Overlay
```

---

## Expected

Higher decoder capacity mainly benefits

* difficult voxels

* boundaries

* small organs

---

## Reviewer Question

Is adaptive decoder computation actually necessary?

---

# O6 Routing Signal Comparison

## Motivation

Difficulty estimation should not rely on a single signal.

---

## Signals

Entropy

Confidence

Feature Variance

MC Dropout

Boundary Probability

---

## Evaluation

Correlation

Latency

Memory

Stability

---

## Table

| Signal | Corr | Time | Memory |
| ------ | ---- | ---- | ------ |

---

## Goal

Select the routing signal

for AdaDec3D.

---

# Go / No-Go Decision

Proceed only if

✔ Difficulty correlates with prediction error

✔ Decoder gain correlates with difficulty

✔ Difficult regions occupy only a small portion of voxels

Otherwise,

adaptive decoder computation should be reconsidered.

---

# Deliverables

Notebook

observe_error.ipynb

observe_difficulty.ipynb

observe_decoder_gain.ipynb

observe_organs.ipynb

observe_routing_signal.ipynb

---

# Final Outputs

Figures

Figure 2

Figure 3

Figure 4

Tables

Organ Statistics

Routing Comparison

Correlation Analysis

---

# Expected Scientific Discovery

If all observations are validated,

this study supports the following conclusion.

> Decoder redundancy in efficient 3D medical image segmentation is spatially heterogeneous.

This conclusion directly motivates the proposed AdaDec3D framework.
