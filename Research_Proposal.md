# AdaDec3D: Adaptive Decoder Computation for Efficient 3D Medical Image Segmentation

**Full Name**

Adaptive Decoder Computation via Difficulty-Aware Routing and Selective Refinement for Efficient 3D Medical Image Segmentation

---

# Research Proposal

**Version:** v2.0

**Target Venues**

Paper A (observation study): MIDL / MLMI / ISBI

Paper B (AdaDec3D method): MICCAI 2026 / IEEE Transactions on Medical Imaging (TMI) / JBHI

---

# 0. Executive Summary

Recent advances in 3D medical image segmentation have significantly improved segmentation accuracy through increasingly powerful encoder-decoder architectures. However, these improvements often come at the cost of excessive computational complexity, particularly in decoder design.

EffiDec3D (CVPR 2025) demonstrates that modern 3D segmentation decoders are heavily over-parameterized. By aggressively reducing decoder channels and removing computationally expensive high-resolution decoder stages, EffiDec3D reduces decoder FLOPs by more than 90% while maintaining nearly identical segmentation performance across multiple datasets.

Despite its impressive efficiency, EffiDec3D follows a fundamentally static computation paradigm:

* every voxel receives identical decoder computation,
* every anatomical structure uses identical decoder capacity,
* every patient is processed using the same decoding strategy.

This proposal argues that such uniform computation is inherently suboptimal.

Instead of immediately proposing a new decoder architecture, we first investigate a more fundamental scientific question:

> **Where does decoder computation actually matter in 3D medical image segmentation?**

We hypothesize that decoder redundancy is **spatially heterogeneous**. Most voxels are easy to segment and require only lightweight decoding, while a relatively small subset of difficult voxels—typically around anatomical boundaries and small structures—benefit substantially from stronger decoder capacity.

Based on this hypothesis, we propose **Adaptive Decoder Computation (AdaDec3D)**, a difficulty-aware decoding framework that dynamically allocates decoder computation according to predicted segmentation difficulty rather than uniformly across the entire volume.

Unlike existing efficient segmentation methods that optimize decoder architectures statically, AdaDec3D focuses on **adaptive computation allocation**, allowing computational resources to be concentrated only where they are truly needed.

---

# 0.5 Two-Paper Publication Strategy

This project is structured as two sequential publications that build on each other.

| | Paper A — Observation | Paper B — Method |
|---|---|---|
| **Claim** | Decoder redundancy is spatially heterogeneous | AdaDec3D exploits this heterogeneity |
| **Contribution** | Empirical analysis (O1–O11) | Adaptive decoder architecture |
| **Target venue** | MIDL / MLMI / ISBI | MICCAI 2026 / TMI |
| **Prerequisite** | E0 + E1 baselines | Paper A accepted |
| **Key finding** | Top 20% uncertain voxels → 80% of gain | DICE ≥ EffiDec3D + 0.3% at same FLOPs |

**Paper A is self-contained** and can be submitted before any AdaDec3D implementation is complete. Its headline claim:

> *The top 20% of uncertain voxels account for >80% of the performance gain from additional decoder capacity, suggesting that selective allocation can recover most of the full-decoder benefit at a fraction of the compute.*

**Paper B builds on Paper A** by proposing AdaDec3D as the practical realization of this observation. The acceptance of Paper A provides both motivation and reviewer confidence for the AdaDec3D design choices.

---

# 1. Motivation

## 1.1 Background

3D medical image segmentation has become a fundamental component of numerous clinical applications, including surgical planning, disease diagnosis, radiotherapy planning, and image-guided interventions.

Recent architectures such as

* UNETR
* SwinUNETR
* SwinUNETRv2
* 3D UX-Net
* MedNeXt

have achieved remarkable segmentation performance by employing increasingly powerful encoder-decoder structures.

However, these models also introduce substantial computational cost, making deployment difficult in many real-world clinical environments.

---

## 1.2 Decoder Redundancy

Recent work, particularly **EffiDec3D**, revisited the computational distribution inside modern segmentation networks and demonstrated an important observation:

> **The decoder contributes a surprisingly large proportion of overall computation while providing relatively limited performance gain.**

EffiDec3D shows that:

* decoder channels can be drastically reduced,
* the highest-resolution decoder stage can be removed,

while maintaining almost identical segmentation accuracy.

These findings suggest that modern decoders contain significant computational redundancy.

---

## 1.3 Limitation of Existing Efficient Decoders

Although EffiDec3D successfully reduces decoder complexity, it adopts a completely static computation strategy.

Specifically,

all voxels,

all organs,

and all patients

share exactly the same decoder architecture.

Formally,

for every voxel

[
C(v)=C_{fixed}
]

where

* (C(v)) denotes decoder computation allocated to voxel (v),
* (C_{fixed}) is identical for every voxel.

This assumption greatly simplifies decoder design, but it implicitly assumes that

> every voxel contributes equally to segmentation difficulty.

Such an assumption is unlikely to hold in practice.

---

## 1.4 Motivation from Clinical Data

Medical images exhibit highly heterogeneous anatomical complexity.

For example,

large organs such as

* liver,
* spleen,
* kidneys,

usually contain smooth boundaries and occupy relatively large volumes.

In contrast,

small anatomical structures including

* pancreas,
* adrenal glands,
* esophagus,
* vessels,

often present

* irregular geometry,
* blurry boundaries,
* severe class imbalance,
* limited contextual information.

These structures are consistently among the most challenging organs across nearly all public segmentation benchmarks.

Furthermore,

prediction errors in modern segmentation networks are rarely distributed uniformly.

Instead,

they tend to cluster around

* organ boundaries,
* small anatomical structures,
* ambiguous tissue interfaces,
* low-contrast regions.

This naturally raises an important question:

> **Do these difficult regions require more decoder computation than easy regions?**

---

## 1.5 Research Gap

Existing efficient segmentation methods primarily optimize decoder computation using **static architectural simplifications**, including

* channel reduction,
* layer removal,
* lightweight convolutions,
* efficient attention modules.

In contrast,

dynamic computation has been extensively studied in

* image classification,
* natural language processing,
* vision transformers,

through techniques such as

* dynamic inference,
* adaptive computation,
* sparse mixture-of-experts.

However,

little attention has been paid to **adaptive decoder computation** in 3D medical image segmentation.

More importantly,

existing methods rarely investigate **whether decoder computation itself should be spatially adaptive**.

As a result,

a fundamental scientific question remains unanswered:

> **Where should decoder computation be allocated?**

---

# 2. Research Questions

Rather than designing another lightweight decoder directly,

this project first seeks to answer several fundamental scientific questions.

---

## RQ1

### Where are segmentation errors concentrated?

Are segmentation errors uniformly distributed across the image volume,

or concentrated around specific anatomical regions?

---

## RQ2

### Where is prediction uncertainty concentrated?

Can prediction uncertainty reliably identify difficult anatomical regions?

---

## RQ3

### Who actually benefits from larger decoder capacity?

Does every voxel benefit equally from stronger decoder computation,

or only a small subset of difficult voxels?

---

## RQ4

### Can decoder computation be allocated adaptively?

If decoder redundancy is spatially heterogeneous,

can adaptive decoder computation improve the efficiency–accuracy trade-off?

---

# 3. Overall Research Strategy

Unlike conventional method-driven research,

this project follows an **observation-driven research paradigm**.

The workflow is illustrated below.

```text
Reproduce EffiDec3D
        │
        ▼
Empirical Observation
        │
        ▼
Scientific Insight
        │
        ▼
Research Hypothesis
        │
        ▼
AdaDec3D Framework
        │
        ▼
Experimental Validation
```

The central philosophy of this project is

> **Observation first, method second.**

Instead of assuming adaptive computation is beneficial,

we first validate whether decoder redundancy is indeed spatially heterogeneous.

Only after this hypothesis is supported by empirical evidence will the adaptive decoder framework be developed.

# 4. Empirical Observation

> **This section does not introduce a new method.**
>
> Instead, it aims to understand where decoder computation is actually needed before designing a new decoder architecture.

Unlike conventional research that begins with proposing a new module, this project first investigates the computational behavior of modern efficient decoders through a series of empirical analyses.

The objective is to answer one fundamental question:

> **Is decoder computation uniformly useful across the entire image volume?**

If the answer is **no**, then adaptive decoder computation becomes both reasonable and necessary.

---

# Observation O1

## Prediction Error is Spatially Heterogeneous

### Motivation

If segmentation errors are uniformly distributed throughout the volume, allocating identical decoder computation to every voxel is reasonable.

However, if prediction errors concentrate only within certain anatomical regions, a uniform decoder may be inefficient.

Therefore, our first observation investigates the spatial distribution of segmentation errors.

---

## Experimental Design

### Input

* Reproduced EffiDec3D model
* BTCV validation dataset
* FeTA validation dataset

---

### Procedure

For each validation case,

1. Generate prediction

```
Prediction
```

2. Compare with ground truth

```
Prediction != GT
```

3. Generate voxel-wise error map

```
Error(v)=
0  Correct
1  Incorrect
```

---

### Expected Visualization

Figure 2(a)

```
CT Image

Ground Truth

Prediction

Error Map
```

---

### Expected Observation

Prediction errors are expected to be highly concentrated around

* anatomical boundaries
* thin organs
* small anatomical structures
* low-contrast regions

rather than uniformly distributed.

---

### Scientific Implication

Decoder computation should primarily focus on these difficult regions instead of the entire image.

---

# Observation O2

## Prediction Uncertainty is Spatially Heterogeneous

### Motivation

Adaptive computation requires an estimate of segmentation difficulty during inference.

Since the ground truth is unavailable during deployment,

prediction uncertainty becomes a natural candidate.

Before adopting uncertainty as the routing signal,

we first verify whether uncertainty indeed correlates with segmentation difficulty.

---

## Experimental Design

For every voxel,

compute predictive entropy

[
U(v)=-\sum_{c=1}^{C}p_c(v)\log p_c(v)
]

where

* (p_c(v)) denotes the predicted probability of class (c).

---

### Generated Outputs

```
Entropy Map
```

---

### Visualization

Figure 2(b)

```
Prediction

Entropy Map

Overlay
```

---

### Statistical Analysis

Divide voxels into entropy bins

| Entropy Range | Number of Voxels |
| ------------- | ---------------- |
| 0.0–0.1       |                  |
| 0.1–0.2       |                  |
| ...           |                  |
| 0.9–1.0       |                  |

---

### Expected Observation

High uncertainty occupies only a relatively small proportion of voxels.

Most voxels should exhibit very low entropy.

---

### Scientific Implication

Prediction difficulty is spatially heterogeneous rather than uniformly distributed.

---

# Observation O3

## Uncertainty Correlates with Prediction Error

Observation O1 and O2 independently analyze

* prediction errors
* uncertainty

The next question is whether they describe the same phenomenon.

---

## Research Question

Does higher uncertainty correspond to higher segmentation error?

---

## Experimental Design

For every voxel,

record

```
Entropy

Prediction Error
```

Then compute

* Pearson correlation
* Spearman correlation

between uncertainty and prediction error.

---

### Expected Figure

Figure 3

```
Prediction Error

↑

│

│

│

└──────────────────────► Entropy
```

---

### Expected Table

| Metric     | Value |
| ---------- | ----- |
| Pearson r  |       |
| Spearman ρ |       |

---

### Expected Observation

Prediction error should increase monotonically with entropy.

---

### Scientific Implication

Entropy can serve as a reliable proxy for segmentation difficulty.

Notice that this experiment **does not prove entropy is optimal**.

Instead,

it demonstrates that entropy is

* parameter-free,
* easily available,
* sufficiently correlated with prediction difficulty.

---

# Observation O4

## Difficult Anatomical Structures Exhibit Higher Uncertainty

### Motivation

Certain anatomical structures are consistently more difficult across public datasets.

We investigate whether these organs also exhibit higher predictive uncertainty.

---

## Experimental Design

Compute

```
Mean Entropy
```

for every anatomical structure.

---

### Expected Table

| Organ         | Mean Entropy | Dice |
| ------------- | ------------ | ---- |
| Liver         |              |      |
| Kidney        |              |      |
| Spleen        |              |      |
| Pancreas      |              |      |
| Esophagus     |              |      |
| Adrenal Gland |              |      |

---

### Expected Observation

Small organs are expected to exhibit

* higher uncertainty
* lower Dice

than large organs.

---

### Scientific Implication

Different anatomical structures require different decoder capacity.

Uniform decoder computation is therefore unlikely to be optimal.

---

# Observation O5

## Decoder Gain is Spatially Heterogeneous

> **This is the most important observation in the entire project.**

The previous observations only analyze segmentation difficulty.

However,

adaptive computation is meaningful only if

larger decoder capacity actually improves these difficult regions.

---

## Research Question

Who truly benefits from stronger decoder computation?

---

## Experimental Design

Train

* Full Decoder
* EffiDec3D

using identical encoder settings.

Generate

```
Prediction_full

Prediction_effi
```

Then compute

```
Improvement Map
=
Prediction_full
−
Prediction_effi
```

or equivalently,

the voxel-wise performance gain obtained by the stronger decoder.

---

### Visualization

Figure 4

```
Entropy Map

Improvement Map

Overlay
```

---

### Statistical Analysis

Group voxels according to entropy.

For each entropy interval,

compute the average Dice improvement obtained by the larger decoder.

---

### Expected Figure

```
Decoder Gain

↑

│

│

│

└────────────────────────► Entropy
```

---

### Expected Observation

Large decoder capacity mainly improves

* high-uncertainty voxels
* boundary regions
* small anatomical structures

while providing little benefit for easy regions.

---

### Scientific Insight

Decoder redundancy is **not uniformly distributed**.

Instead,

decoder computation is valuable only for a relatively small subset of difficult voxels.

This observation directly motivates adaptive decoder computation.

---

# Observation O6

## Difficulty Evolution During Training

### Research Question

Does prediction difficulty persist throughout training, or is it a transient artifact of early training?

### Motivation

If high-entropy regions disappear as the model trains, difficulty is unstable and cannot support reliable routing. If difficulty persists and concentrates at boundaries and hard organs, it is a stable signal suitable for adaptive computation.

### Method

Save model checkpoints at training epochs {5, 10, 20, 30, 50}. Compute mean entropy of the validation set at each checkpoint and visualize its spatial distribution over time.

### Expected Finding

Mean entropy decreases over training but stabilizes. Residual high-entropy voxels at convergence are concentrated at anatomical boundaries and small organs (Pancreas, Adrenal), confirming that difficulty is a persistent, stable signal.

### Role

Supports Paper A's claim that entropy-based routing is reliable rather than opportunistic.

---

# Observation O7

## Cross-Dataset Consistency

### Research Question

Do the O1–O5 findings generalize from CT (BTCV) to MRI (FeTA)?

### Motivation

If decoder gain concentrates on difficult voxels only in BTCV, the observation may be dataset-specific. Cross-dataset replication on a different modality (fetal brain MRI) is essential for a general claim.

### Method

Repeat the O1–O5 analysis pipeline on the FeTA 2021 dataset using EffiDec3D trained on FeTA.

### Expected Finding

Gain–Entropy Pearson r > 0.40 on FeTA, confirming the relationship is modality-agnostic.

### Role

Cross-dataset replication is a standard criterion for MIDL/MLMI acceptance.

---

# Observation O8

## Backbone Consistency

### Research Question

Does the O5 Gain–Entropy correlation hold when the backbone changes from UXNET to SwinUNETR?

### Motivation

If the finding depends on the specific backbone, AdaDec3D cannot claim general applicability to other efficient decoders.

### Method

Train SwinUNETR_EffiDec3D on BTCV. Run the O5 Gain–Entropy analysis using SwinUNETR predictions.

### Expected Finding

Gain–Entropy r > 0.45 for SwinUNETR, confirming backbone-agnostic signal.

### Role

Backbone consistency strengthens the architectural generality claim in Paper A and Paper B.

---

# Observation O9

## Pareto Analysis — Headline Finding for Paper A

### Research Question

What fraction of voxels account for the majority of decoder gain?

### Motivation

A Pareto distribution of gain (top 20% of uncertain voxels → 80% of gain) is the quantitative foundation for selective decoder allocation. This is the headline number that justifies AdaDec3D's design philosophy.

### Method

Sort all validation voxels by entropy (descending). Plot cumulative decoder gain as a function of voxel percentile (Lorenz-style curve). Identify the minimal voxel percentage covering 80% of total gain.

### Expected Finding

The top ~20% of uncertain voxels account for ~80% of total decoder gain.

### Go Criterion for Paper A

Top ≤ 30% of uncertain voxels cover ≥ 80% of total decoder gain.

### Role

This is the headline finding and primary figure of Paper A.

---

# Observation O10

## Organ Size vs Difficulty

### Research Question

Is difficulty driven by organ size, or is it an independent signal?

### Motivation

Reviewers will ask: "Is your entropy-based difficulty just a proxy for small organs?" This observation directly addresses this concern. If difficulty is heterogeneous even within organ classes and large organs also exhibit localized difficulty, entropy is richer than a simple size-based routing rule.

### Method

For each organ, compute mean volume (size proxy) and mean entropy. Measure Spearman correlation between size and difficulty.

### Expected Finding

Weak-to-moderate negative correlation (smaller organs are harder on average), but high residual variance: some large organs (liver boundary, stomach) also exhibit high difficulty. This demonstrates entropy captures difficulty beyond size.

### Role

Defends entropy-based routing against the "size proxy" objection in Paper A peer review.

---

# Summary of Empirical Observations

The eleven observations collectively answer six scientific questions.

| Observation | Scientific Question | Expected Finding | Paper |
| ----------- | ------------------- | ---------------- | ----- |
| O1 | Where do errors occur? | Errors concentrate in difficult anatomical regions | A |
| O2 | Where is uncertainty located? | High uncertainty occupies only a small proportion of voxels | A |
| O3 | Does uncertainty represent difficulty? | Prediction error increases with uncertainty | A |
| O4 | Which organs are difficult? | Small organs exhibit higher uncertainty and lower Dice | A |
| O5 | Who benefits from larger decoders? | Large decoder capacity mainly benefits high-uncertainty regions | A |
| O6 | Is difficulty persistent? | High-entropy voxels stabilize at boundaries by epoch 30 | A |
| O7 | Does this generalize across datasets? | Replicated on FeTA (MRI), confirming modality-agnostic finding | A |
| O8 | Does this generalize across backbones? | Replicated with SwinUNETR, confirming backbone-agnostic signal | A |
| O9 | How concentrated is the gain? (headline) | Top 20% uncertain voxels → 80% of decoder gain | A |
| O10 | Is difficulty just a size proxy? | No — entropy captures difficulty beyond organ size | A |
| O11 | Which routing signal is best? | Entropy: best correlation, lowest overhead | B |

If O1–O9 are experimentally validated,

they collectively support the central hypothesis:

> **Decoder redundancy is spatially heterogeneous, and decoder computation should therefore be allocated adaptively rather than uniformly.**

Only after validating this hypothesis do we proceed to design the proposed AdaDec3D framework.
# 5. Scientific Insight

The empirical observations presented in the previous section are not independent findings.

Instead, they collectively reveal a previously overlooked property of efficient 3D medical image segmentation.

> **Decoder redundancy is spatially heterogeneous.**

This section summarizes the scientific insight derived from the observation study and formulates the central hypothesis that motivates AdaDec3D.

---

# 5.1 From Observation to Insight

Existing efficient segmentation methods—including EffiDec3D—implicitly assume that decoder computation is equally valuable across the entire image volume.

Mathematically,

[
C(v)=C_{fixed}
]

where

* (C(v)) denotes the decoder computation allocated to voxel (v),
* (C_{fixed}) is constant for every voxel.

This assumption significantly simplifies decoder design and enables aggressive architectural optimization.

However, the empirical observations suggest that this assumption does not hold.

Instead,

different voxels exhibit dramatically different levels of segmentation difficulty.

Similarly,

different voxels benefit differently from stronger decoder capacity.

Therefore,

decoder computation should no longer be regarded as a globally shared resource,

but rather as a computational budget that should be allocated selectively.

---

# 5.2 Spatially Heterogeneous Decoder Redundancy

The observations suggest three important characteristics.

---

## Observation A

Prediction difficulty is spatially heterogeneous.

Most voxels are segmented with extremely high confidence.

Only a relatively small subset of voxels exhibits

* high uncertainty,
* ambiguous boundaries,
* inconsistent predictions.

---

## Observation B

Decoder gain is also spatially heterogeneous.

Increasing decoder capacity does not uniformly improve segmentation quality.

Instead,

performance improvement is concentrated around

* anatomical boundaries,
* thin structures,
* small organs,
* difficult tissue interfaces.

---

## Observation C

The spatial distributions of

* uncertainty,
* segmentation error,
* decoder gain

are highly correlated.

These three phenomena are therefore likely to describe the same underlying computational property.

---

Taken together,

these observations suggest that

> **decoder redundancy itself is spatially heterogeneous.**

---

# 5.3 A New Perspective

Traditional decoder optimization attempts to answer

> How can we build a smaller decoder?

Examples include

* channel reduction,
* lightweight convolution,
* efficient attention,
* decoder pruning.

These approaches optimize **decoder architecture**.

---

This project instead asks a different question.

> **Where should decoder computation be allocated?**

Rather than designing another lightweight decoder,

we propose to allocate decoder computation according to prediction difficulty.

This changes the optimization objective from

> designing a smaller decoder

to

> allocating decoder computation more intelligently.

---

# 5.4 Research Hypotheses

Based on the above observations,

we formulate the following hypotheses.

---

## H1

### Prediction uncertainty reflects segmentation difficulty.

Prediction uncertainty is expected to correlate strongly with

* segmentation errors,
* anatomical ambiguity,
* decoder improvement.

Therefore,

uncertainty can serve as a practical routing signal during inference.

Notice that

this hypothesis does **not** claim that entropy is theoretically optimal.

Instead,

it assumes entropy is

* simple,
* parameter-free,
* sufficiently informative.

Alternative routing signals will also be investigated experimentally.

---

## H2

### Larger decoder capacity is only necessary for difficult regions.

Increasing decoder capacity is expected to improve

* uncertain voxels,
* boundary regions,
* small organs,

while providing minimal improvement for easy voxels.

Therefore,

uniform decoder computation is computationally inefficient.

---

## H3

### Adaptive decoder computation provides a better efficiency–accuracy trade-off.

If decoder computation is concentrated only on difficult regions,

the overall segmentation accuracy can be improved with only a modest increase in computational cost.

Consequently,

adaptive decoder computation is expected to dominate static decoder architectures in terms of

* Dice,
* HD95,
* GFLOPs,
* latency.

---

# 5.5 Research Philosophy

This work differs fundamentally from conventional architecture design.

Instead of proposing a new module first,

we first establish a scientific observation,

then design a method to exploit that observation.

The logical chain of this work is

```text
Observation
        │
        ▼
Scientific Insight
        │
        ▼
Research Hypothesis
        │
        ▼
Adaptive Decoder Computation
        │
        ▼
Experimental Validation
```

This observation-driven paradigm provides a stronger scientific foundation than introducing architectural modifications without first understanding the underlying computational behavior.

---

# 5.6 Design Principles of AdaDec3D

The proposed framework follows three design principles.

---

## Principle 1

### Estimate Difficulty Before Allocating Computation

Adaptive computation should be guided by predicted segmentation difficulty rather than predefined anatomical priors.

Difficulty estimation must therefore

* require no additional annotation,
* be available during inference,
* introduce minimal computational overhead.

---

## Principle 2

### Allocate Computation Instead of Increasing Capacity

The objective is not to build a larger decoder.

Instead,

the same computational budget should be redistributed according to voxel difficulty.

Easy voxels should consume minimal computation,

while difficult voxels receive additional decoder capacity.

---

## Principle 3

### Maintain Compatibility with Existing Efficient Decoders

The proposed framework should not depend on a specific backbone.

Instead,

it should be applicable to existing efficient segmentation architectures such as

* EffiDec3D
* SwinUNETR
* SwinUNETRv2
* 3D UX-Net
* MedNeXt

requiring only decoder-side modifications.

This enables AdaDec3D to serve as a general adaptive decoding framework rather than a model-specific implementation.

---

# 5.7 Transition to the Proposed Method

The previous sections establish the motivation for adaptive computation through empirical evidence.

The remaining question is

> **How can decoder computation be allocated adaptively during inference?**

To answer this question,

the next section introduces **AdaDec3D**, a difficulty-aware adaptive decoding framework composed of three cooperative components:

1. **Difficulty Estimation**

   * Estimate voxel-wise segmentation difficulty from coarse predictions.

2. **Adaptive Decoder Computation**

   * Dynamically allocate decoder capacity according to predicted difficulty.

3. **Selective ROI Refinement**

   * Perform high-resolution refinement only in regions expected to benefit from additional computation.

Unlike previous efficient segmentation methods that optimize decoder architecture statically,

AdaDec3D optimizes **where computation is spent**, enabling decoder resources to be concentrated where they provide the greatest benefit.

# 6. AdaDec3D Framework

Unlike existing efficient segmentation methods that optimize decoder architectures statically, AdaDec3D aims to optimize **decoder computation allocation**.

The proposed framework is built upon one central idea:

> **Different voxels should receive different decoder capacity according to their predicted segmentation difficulty.**

Rather than treating every voxel equally,

AdaDec3D dynamically allocates computational resources only to regions expected to benefit from stronger decoding.

---

# 6.1 Design Philosophy

Existing decoder optimization can be summarized as

```text
Large Decoder
        │
Architecture Compression
        ▼
Small Decoder
```

EffiDec3D belongs to this category.

The decoder architecture is simplified once during training,

and remains fixed during inference.

---

AdaDec3D instead introduces

```text
Lightweight Decoder
        │
Difficulty Estimation
        │
Adaptive Computation Allocation
        ▼
Dynamic Decoder
```

The decoder architecture itself is no longer fixed.

Instead,

decoder computation becomes an adaptive resource allocated according to segmentation difficulty.

---

# 6.2 Overall Framework

The overall pipeline is illustrated below.

```text
Input Volume
      │
      ▼
Backbone Encoder
      │
      ▼
Lightweight Coarse Decoder
      │
      ▼
Coarse Prediction
      │
      ▼
Difficulty Estimation
      │
      ▼
Difficulty Map
      │
      ▼
Adaptive Computation Controller
      ├──────────────┐
      ▼              ▼
Easy Region     Difficult Region
Light Decoder   Strong Decoder
      └──────────────┘
             │
             ▼
ROI Refinement
             │
             ▼
Final Prediction
```

Compared with EffiDec3D,

AdaDec3D introduces only one additional concept:

> **Adaptive Decoder Computation**

The remaining modules are implementation choices rather than independent contributions.

---

# 6.3 Framework Components

The proposed framework consists of three sequential stages.

---

## Stage 1

### Difficulty Estimation

Purpose

Estimate the segmentation difficulty of every voxel before allocating decoder computation.

Input

* coarse prediction
* intermediate decoder features

Output

```text
Difficulty Map
```

representing the expected segmentation difficulty.

---

### Candidate Difficulty Signals

The framework itself does **not** assume a specific routing signal.

Instead,

multiple difficulty estimators will be investigated.

| Signal               | Training | Inference | Complexity |
| -------------------- | -------- | --------- | ---------- |
| Entropy              | ✓        | ✓         | Very Low   |
| Confidence           | ✓        | ✓         | Very Low   |
| MC Dropout           | ✓        | ✓         | High       |
| Feature Variance     | ✓        | ✓         | Medium     |
| Boundary Probability | ✓        | ✓         | Medium     |

The initial implementation adopts predictive entropy because it is

* parameter-free,
* differentiable,
* directly available from coarse prediction.

However,

the framework is independent of this particular choice.

---

### Expected Property

Difficulty estimation should satisfy

* high correlation with prediction error,
* low computational overhead,
* stable across datasets,
* compatible with different segmentation backbones.

---

## Stage 2

### Adaptive Decoder Computation

This is the core component of AdaDec3D.

Instead of increasing decoder capacity globally,

decoder computation is allocated according to the estimated difficulty map.

Conceptually,

```text
Easy Voxels
        │
Small Decoder

Medium Difficulty
        │
Medium Decoder

High Difficulty
        │
Large Decoder
```

Only voxels predicted to be difficult receive stronger decoder computation.

---

### Candidate Implementations

Adaptive computation may be implemented using several strategies.

| Strategy                | Supported |
| ----------------------- | --------- |
| Dynamic Width           | ✓         |
| Mixture-of-Experts      | ✓         |
| Conditional Convolution | Future    |
| Dynamic Depth           | Future    |

For the first implementation,

Mixture-of-Experts is adopted because

* implementation is relatively straightforward,
* routing is differentiable,
* decoder capacity is easily adjustable.

Importantly,

MoE is **not** regarded as the contribution itself.

It is merely one implementation of adaptive computation.

---

### Design Principle

Rather than asking

> Which decoder is best?

AdaDec3D asks

> Which decoder should be used **here**?

This shifts optimization from architecture design to computation allocation.

---

## Stage 3

### Selective ROI Refinement

After adaptive decoding,

remaining difficult regions are refined at higher spatial resolution.

Unlike conventional two-stage segmentation,

ROI refinement is guided entirely by the estimated difficulty map.

Only regions expected to benefit from additional computation are refined.

---

### Pipeline

```text
Coarse Prediction
        │
Difficulty Map
        │
ROI Selection
        │
High-resolution Refinement
        │
Prediction Fusion
```

This design avoids expensive full-volume refinement while preserving high-resolution information around difficult anatomical structures.

---

### Expected Benefit

Compared with global refinement,

ROI refinement is expected to

* reduce FLOPs,
* reduce memory consumption,
* preserve small-organ accuracy.

---

# 6.4 Why This Framework?

AdaDec3D differs fundamentally from previous decoder optimization methods.

| Previous Efficient Decoders | AdaDec3D                        |
| --------------------------- | ------------------------------- |
| Optimize architecture       | Optimize computation allocation |
| Static decoder              | Dynamic decoder                 |
| Uniform computation         | Difficulty-aware computation    |
| Global optimization         | Spatially adaptive optimization |

The proposed framework therefore changes the optimization objective from

> **How can we build a smaller decoder?**

to

> **How should decoder computation be distributed?**

---

# 6.5 Relationship to EffiDec3D

AdaDec3D is designed as a direct extension of EffiDec3D rather than a replacement.

The lightweight decoder proposed in EffiDec3D serves as the computational baseline.

Adaptive computation is introduced only after this efficient baseline has been established.

Conceptually,

```text
Original Network
        │
        ▼
EffiDec3D
(Static Efficient Decoder)
        │
        ▼
AdaDec3D
(Adaptive Efficient Decoder)
```

Therefore,

AdaDec3D inherits

* lightweight decoder design,
* efficient backbone compatibility,
* low computational cost,

while introducing adaptive computation allocation.

---

# 6.6 Expected Advantages

Compared with static decoder architectures,

AdaDec3D is expected to provide several advantages.

### Accuracy

Additional decoder capacity is concentrated on difficult regions,

leading to improved segmentation of

* pancreas,
* adrenal glands,
* esophagus,
* vessels,
* organ boundaries.

---

### Efficiency

Instead of globally increasing decoder complexity,

additional computation is applied only where necessary.

Therefore,

the increase in FLOPs is expected to be substantially smaller than using a uniformly larger decoder.

---

### Generality

Because adaptive computation is implemented entirely on the decoder side,

the framework can be integrated into multiple existing segmentation architectures with minimal modification.

Potential backbones include

* EffiDec3D
* SwinUNETR
* SwinUNETRv2
* 3D UX-Net
* MedNeXt

---

# 6.7 Key Difference from the Original Proposal

The original proposal described three seemingly independent modules:

* Uncertainty Estimation
* MoE Decoder
* ROI Refinement

In the revised framework,

these are no longer treated as separate contributions.

Instead,

they are organized as three sequential stages within a unified Adaptive Decoder Computation framework.

```text
Difficulty Estimation
        │
        ▼
Adaptive Computation Allocation
        │
        ▼
Selective Refinement
```

This unified perspective emphasizes the central scientific contribution:

> **Adaptive allocation of decoder computation according to segmentation difficulty.**

All architectural modules—including entropy estimation, MoE routing, and ROI refinement—are implementation choices that serve this overarching objective rather than standalone innovations.

---

# 6.8 Concrete Implementation

This section records the specific design decisions made for the initial implementation of AdaDec3D on top of the 3D UX-Net / EffiDec3D backbone. These are starting-point choices subject to ablation.

---

## Backbone

EffiDec3D with 3D UX-Net encoder:

* Encoder channels: [48, 96, 192, 384]
* Compressed decoder channels: 48 (uniform, matching EffiDec3D)
* Resolution factor: 2 (decoder outputs at D/2 × H/2 × W/2)
* Skip aggregation: addition

---

## Difficulty Estimation

Predictive entropy on the coarse decoder softmax output:

```
U(v) = -Σ p_c(v) log p_c(v)
```

Zero parameters. Computed from the existing coarse decoder output at no additional forward-pass cost.

---

## Adaptive Router

```
router = Linear(n_decoder_channels + 1, n_experts)
```

Input: concatenation of global average-pooled bottleneck feature [B, n_ch] and mean uncertainty scalar [B, 1].

Output: softmax routing weights [B, n_experts].

Uses soft routing during training (weighted sum of expert outputs). Hard routing (argmax) can be used at inference to avoid redundant forward passes.

---

## Expert Decoders

Three experts with different hidden-channel widths:

| Expert | Hidden Channels | Scenario |
|--------|----------------|----------|
| S      | 32             | Large organs, clear boundaries, low uncertainty |
| M      | 64             | Medium complexity structures |
| L      | 96             | Small organs, ambiguous boundaries, high uncertainty |

Each expert is a two-layer residual conv block followed by a segmentation output head:

```
conv(in_ch → hidden_ch) → IN → ReLU
conv(hidden_ch → in_ch) → IN
residual add → ReLU
conv(in_ch → out_classes, 1×1)
```

---

## ROI-Aware Refinement

Uncertainty mask at quantile q = 0.50 (top 50% most uncertain voxels):

```
roi_mask = (U > quantile(U, 0.50))
```

The residual refinement block applies only within this mask:

```
residual = conv_block(feat) × roi_mask
output = ReLU(feat + residual)
```

Voxels outside the ROI pass through unchanged (zero residual update).

---

## Loss Function

Four-term training objective:

```
L = L_seg + λ_coarse · L_coarse + λ_unc · L_unc + λ_res · L_res + λ_router · L_router
```

| Term | Default λ | Purpose |
|------|-----------|---------|
| L_seg | 1.0 | DiceCE on final prediction (upsampled to full resolution) |
| L_coarse | 0.5 | Auxiliary DiceCE on coarse decoder — prevents backbone from degrading |
| L_unc | 0.1 | Calibration: high-entropy voxels should correlate with actual errors |
| L_res | 0.05 | Resource penalty: encourages lighter experts when accuracy allows |
| L_router | 0.1 | Load balancing: prevents all samples collapsing to a single expert |

All λ values are ablated in Stage 4 experiments.

---

## Two-Stage Training Protocol

**Stage 1 — New modules only (backbone frozen)**

Frozen modules: encoder, coarse decoder (uxnet_3d, encoder2-5, decoder3-5, coarse_out)

Trainable: router, experts (×3), ROI refiner only (~0.5M of 3.7M total params)

Duration: 20,000 iterations, lr = 5×10⁻⁴

**Stage 2 — End-to-end fine-tune**

All parameters unfrozen. Layered learning rates:

* New modules: lr = 5×10⁻⁴
* Backbone: lr × 0.1 = 5×10⁻⁵

Duration: 25,000 iterations

**Weight initialization**: EffiDec3D trained weights loaded into encoder and coarse decoder. The EffiDec3D output head (out.*) maps to AdaDec3D's coarse_out.* — all other key names match exactly.

---

## Expected Computational Cost

| Configuration | GFLOPs |
|--------------|--------|
| Full 3DUXNET (E0) | 632 |
| EffiDec3D (E1) | 51.47 |
| AdaDec3D — 80% easy (Expert-S) + 20% hard (Expert-L) | ~60–80 (estimated) |

Actual cost depends on routing distribution discovered during training. O5 (Decoder Gain Analysis) will reveal what proportion truly requires Expert-L.

# 7. Training Strategy and Experimental Plan

This chapter describes the complete research workflow for AdaDec3D.

Unlike conventional model development, this project follows an **observation-driven development pipeline**. The proposed method will **not** be implemented until the underlying scientific hypotheses have been experimentally verified.

---

# 7.1 Overall Development Roadmap

The project follows a two-paper strategy with sequential stages.

```text
Stage 0
Literature Review
        │
        ▼
Stage 1
Reproduce EffiDec3D
        │
        ▼
Stage 2
Observation Study O1–O11
  (O1–O5 critical gate)
  (O6–O10 extended analysis)
        │
        ▼
Go / No-Go Decision (Paper A)
        │
        ▼
Paper A Draft
MIDL / MLMI / ISBI Submission
        │
        ▼
Stage 3
AdaDec3D Development
  (parallel to Paper A review)
        │
        ▼
Stage 4
Ablation & Analysis
        │
        ▼
Paper A Accepted
        │
        ▼
Paper B Draft
MICCAI 2026 / TMI Submission
```

Each stage has explicit deliverables and acceptance criteria.

**Paper A** (observation study) is the Go/No-Go gate for Paper B. AdaDec3D development begins only after O1–O5 pass their thresholds.

**Paper A and Stage 3 run in parallel**: once O1–O5 pass, Paper A writing and AdaDec3D implementation proceed simultaneously. This avoids idle time during the Paper A review period (typically 3–6 months).

---

# 7.2 Stage 1 — Baseline Reproduction

## Objective

Faithfully reproduce EffiDec3D before introducing any modifications.

The reproduced baseline will serve as the reference model for all subsequent experiments.

---

## Datasets

Primary datasets

* BTCV
* FeTA

Optional validation datasets

* AMOS
* TotalSegmentator (selected organs)

---

## Evaluation Metrics

Segmentation

* Dice
* HD95
* Surface Dice (optional)

Efficiency

* GFLOPs
* Parameters
* GPU Memory
* Inference Latency

---

## Acceptance Criteria

| Metric          | Target                         |
| --------------- | ------------------------------ |
| BTCV Dice       | Within ±0.3% of paper          |
| FeTA Dice       | Within ±0.3% of paper          |
| GFLOPs          | Consistent with reported value |
| Inference Speed | Comparable                     |

If the reproduced baseline deviates significantly,

the project will **not** proceed to Stage 2.

---

# 7.3 Stage 2 — Observation Study

This stage validates the scientific hypotheses established in Chapter 4.

No adaptive decoder will be implemented before completing this stage.

> **Execution guide**: see `2_Observation_Study.md` for notebook filenames, specific output formats, and per-observation deliverables.
> **Baseline commands** (E0 full decoder, E1 EffiDec3D) needed for O5: see `Experiment-Design.md` Part 3.

---

## O1

Prediction Error Distribution

Question

Where do segmentation errors occur?

Output

* Error map
* Error heatmap
* Organ-wise error statistics

Expected Figure

Figure 2(a)

---

## O2

Difficulty Distribution

Question

Where are difficult voxels located?

Output

* Difficulty map
* Difficulty histogram
* Spatial visualization

Expected Figure

Figure 2(b)

---

## O3

Difficulty vs Prediction Error

Question

Does predicted difficulty correlate with segmentation error?

Metrics

* Pearson Correlation
* Spearman Correlation
* Reliability Diagram (optional)

Expected Figure

Figure 3

---

## O4

Difficulty vs Organ

Question

Which anatomical structures are inherently difficult?

Output

Organ-wise

* Dice
* Mean Difficulty
* Boundary Difficulty

Expected Table

Table 2

---

## O5

Decoder Gain Analysis

Question

Who actually benefits from larger decoder computation?

Output

Voxel-wise

* Improvement Map
* Gain Histogram
* Gain vs Difficulty

Expected Figure

Figure 4

This experiment provides the strongest motivation for adaptive decoder computation.

---

## O6

Routing Signal Comparison

Candidate signals

* Entropy
* Confidence
* Feature Variance
* Boundary Probability
* MC Dropout

Metrics

* Correlation with prediction error
* Computational overhead
* Routing stability

Output

Table comparing different routing signals.

---

# 7.4 Go / No-Go Decision

Adaptive decoder development proceeds **only if** the following observations are validated.

---

## Criterion A

Difficulty correlates with prediction error.

Expected

Pearson correlation

> 0.6

---

## Criterion B

Difficulty is concentrated around difficult anatomical regions.

Expected

Boundary overlap significantly exceeds random distribution.

---

## Criterion C

Large decoder capacity mainly benefits high-difficulty voxels.

Expected

Improvement increases monotonically with predicted difficulty.

---

If one or more criteria are not satisfied,

the adaptive routing strategy should be reconsidered before implementing AdaDec3D.

---

# 7.5 Stage 3 — AdaDec3D Development

Only after the observation study is validated.

Development proceeds incrementally.

---

## Experiment E1

Difficulty Estimation

Goal

Implement voxel-wise difficulty estimation.

Candidates

* Entropy
* Confidence
* Feature Variance

Evaluation

Prediction-error correlation.

---

## Experiment E2

Adaptive Decoder Computation

Goal

Replace static decoder with adaptive computation.

Candidate implementations

* Dynamic Width
* MoE
* Conditional Routing

Evaluation

Dice

GFLOPs

Latency

---

## Experiment E3

Selective ROI Refinement

Goal

Improve segmentation around difficult regions.

Evaluation

Small-organ Dice

Boundary Dice

Memory

Latency

---

# 7.6 Stage 4 — Ablation Study

The proposed framework will be analyzed component by component.

---

## A1

Difficulty Estimator

Compare

* Entropy
* Confidence
* MC Dropout
* Feature Variance

Question

Which signal best predicts segmentation difficulty?

---

## A2

Routing Strategy

Compare

* Static Decoder
* Two-Level Routing
* Three-Level Routing
* Continuous Routing (optional)

Question

How much adaptivity is necessary?

---

## A3

Decoder Capacity

Compare

Small

Medium

Large

Question

What is the optimal computation allocation?

---

## A4

ROI Refinement

Compare

Without refinement

Global refinement

Adaptive refinement

Question

Is selective refinement worthwhile?

---

## A5

Routing Threshold

Investigate

Different routing thresholds.

Question

How sensitive is the framework to routing decisions?

---

## A6

Computational Cost

Measure

* FLOPs
* GPU Memory
* Runtime
* Throughput

rather than reporting FLOPs alone.

---

# 7.7 Main Comparison

AdaDec3D will be compared against

## Lightweight Models

* EffiDec3D
* nnUNet (lightweight setting if available)

---

## Standard Models

* SwinUNETR
* SwinUNETRv2
* MedNeXt
* 3D UX-Net

---

## Dynamic Computation Methods (if applicable)

Any available adaptive inference methods compatible with medical segmentation.

---

# 7.8 Visualization

The paper will contain extensive qualitative analysis.

---

## Figure A

Prediction

Ground Truth

Difference

---

## Figure B

Difficulty Map

---

## Figure C

Routing Map

Display

Easy

Medium

Hard

regions.

---

## Figure D

ROI Refinement

Show

Before

↓

After

---

## Figure E

Failure Cases

Analyze

* false positives
* false negatives
* routing mistakes

---

# 7.9 Expected Paper Figures

| Figure | Purpose                                     |
| ------ | ------------------------------------------- |
| Fig.1  | Overall AdaDec3D framework                  |
| Fig.2  | Observation study (Error & Difficulty maps) |
| Fig.3  | Difficulty vs Prediction Error              |
| Fig.4  | Decoder Gain Analysis                       |
| Fig.5  | Adaptive Decoder Pipeline                   |
| Fig.6  | Qualitative Comparison                      |
| Fig.7  | Routing Visualization                       |

---

# 7.10 Risks and Mitigation

## Risk 1

Difficulty estimation is not sufficiently correlated with prediction error.

Mitigation

Evaluate alternative routing signals.

---

## Risk 2

Adaptive routing increases latency.

Mitigation

Optimize routing implementation and minimize synchronization overhead.

---

## Risk 3

Improvement is too small.

Mitigation

Focus evaluation on

* small organs,
* difficult anatomical structures,
* boundary accuracy,

rather than only average Dice.

---

## Risk 4

Reviewer questions the necessity of adaptive computation.

Mitigation

Use Observation Study (Chapter 4) as empirical evidence demonstrating that decoder gain is spatially heterogeneous.

---

# 7.11 Expected Contributions

Upon completion, the project is expected to provide:

### Scientific Contribution

Reveal that decoder redundancy in efficient 3D medical image segmentation is spatially heterogeneous.

---

### Methodological Contribution

Introduce a general framework for adaptive decoder computation based on voxel-wise difficulty estimation.

---

### Experimental Contribution

Provide the first systematic empirical study analyzing

* segmentation difficulty,
* decoder gain,
* computation allocation,

in efficient 3D medical image segmentation.

---

# End of Research Proposal

The proposed research is expected to transform decoder optimization from

> **"Designing smaller decoders"**

to

> **"Allocating decoder computation intelligently."**

This perspective is independent of any particular routing signal or decoder implementation and therefore has the potential to serve as a general framework for future efficient medical image segmentation research.
