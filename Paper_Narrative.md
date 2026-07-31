# Paper Narrative (Observation Study)

# Working Title

**Characterizing Decoder Computation in 3D Medical Image Segmentation: A Characterization Study**

> Alternative (if emphasizing redundancy):
>
> **Rethinking Decoder Redundancy in Efficient 3D Medical Image Segmentation: A Characterization Study**

---

# Core Scientific Question

Modern medical image segmentation models devote substantial computation to the decoder, yet almost all work evaluates decoder designs only through **final Dice, HD95, FLOPs, and parameter count**.

These aggregate metrics answer **how much** performance changes, but not:

- Where does decoder computation actually matter?
- Is its contribution spatially uniform?
- Is useful decoder computation concentrated?
- Can useful decoder computation be predicted before executing the full decoder?

This paper answers these questions through a **systematic characterization study**, not a new segmentation method.

---

# Central Hypothesis

Decoder computation is **not uniformly useful**.

Instead, decoder benefit exhibits three intrinsic properties:

1. **Heterogeneity**
2. **Concentration**
3. **Predictability**

Together these properties characterize how decoder computation contributes to segmentation performance.

---

# Scientific Narrative

## Stage 1 — Heterogeneity

### Research Question

**Is decoder computation uniformly useful?**

Method:

Compare a parameter-matched Full Decoder and Efficient Decoder under the same training pipeline.

Measure

- Positive Flip
- Negative Flip
- Net Flip

Analyze

- Global
- Boundary distance
- Organ-wise

### Main Finding

Decoder computation is highly heterogeneous.

Some regions improve.

Some regions deteriorate.

Many regions are unaffected.

The decoder is therefore **not uniformly beneficial**.

---

## Stage 2 — Concentration

### Research Question

**Where is decoder computation actually useful?**

Method

Construct oracle benefit maps.

Rank regions by decoder benefit.

Evaluate

- Oracle coverage curve
- Top-k benefit
- K80

### Main Finding

Most decoder benefit is concentrated in a small spatial fraction.

The majority of image regions contribute little additional benefit.

This reveals strong spatial concentration of decoder utility.

---

## Stage 3 — Predictability

### Research Question

**Can useful decoder computation be predicted before executing the full decoder?**

Method

Compare

- Entropy
- Confidence
- Lightweight predictor
- Random
- Oracle

Evaluate

- Any Flip AUROC
- Positive Flip AUROC
- Positive-vs-Negative AUROC
- Region-selection curves

### Main Finding

Useful decoder regions are partially predictable.

Prediction signals identify where decoder refinement is likely to matter, although improvement direction remains substantially harder than instability detection.

---

# Overall Scientific Conclusion

Decoder computation is characterized by

- heterogeneous contribution,
- concentrated benefit,
- partial predictability.

This characterization explains why decoder computation is an attractive target for adaptive computation, without proposing a new adaptive method.

---

# Contributions

### Contribution 1

We present the first systematic characterization of decoder computation in 3D medical image segmentation using matched Full/Efficient decoder pairs.

### Contribution 2

We demonstrate that decoder benefit is spatially heterogeneous rather than uniformly distributed.

### Contribution 3

We show that decoder benefit is highly concentrated, with most useful refinement arising from a small fraction of spatial locations.

### Contribution 4

We demonstrate that useful decoder computation is partially predictable using inexpensive inference-time uncertainty signals.

---

# What This Paper Is NOT

This paper does **not**

- propose a new segmentation network;
- propose adaptive decoding;
- claim new state-of-the-art accuracy.

Instead, it provides a scientific understanding of decoder behavior.

---

# Why Controlled Reproduction Is Sufficient

The paper studies the **relative effect** between Full and Efficient decoders.

Therefore the essential requirement is

- identical dataset,
- identical optimization,
- identical evaluation,

rather than exact reproduction of the published Dice.

The study depends on internally controlled comparisons rather than absolute benchmark performance.

---

# Relation to Future Work

This paper establishes a characterization framework.

Future adaptive methods (e.g., AdaDec3D) may exploit these observations to reduce executed computation while preserving segmentation quality.

The present paper intentionally stops at the observation level.
