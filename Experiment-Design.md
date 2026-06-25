# AdaDec3D Experiment Design — Index

This document is the index. The full experiment design is split into two files by research phase.

---

## Phase 1 — Observation Study (Paper A)

**File**: [Experiment-Design-Observation.md](Experiment-Design-Observation.md)

**Contents**:
- Part 0: Hardware (Kaggle P100)
- Part 1: Environment Setup
- Part 2: Dataset Setup (BTCV + FeTA)
- Part 3: Reproduce EffiDec3D (E0 full decoder, E1 EffiDec3D)
- Part 3.5: Observation study O1–O9 with full code
- Go/No-Go table for Paper A submission
- Timeline: Weeks 1–7

**Goal**: Empirically confirm that decoder redundancy is spatially heterogeneous
and that the top 20% of uncertain voxels account for ≥80% of decoder gain.
Submit as Paper A to MIDL / MLMI / ISBI before starting AdaDec3D development.

---

## Phase 2 — AdaDec3D Training (Paper B)

**File**: [Experiment-Design-AdaDec3D.md](Experiment-Design-AdaDec3D.md)

**Contents**:
- Part 4: AdaDec3D experiments (E2 MoE-only, E3 ROI-only, E4 full)
- Part 5: Metrics (DICE, HD95, GFLOPs, latency, expert routing, ROI coverage)
- Part 6: Ablation studies (module contribution + hyperparameter)
- Go/No-Go criteria for Paper B
- Timeline: Weeks 8–14

**Goal**: Implement, train, and evaluate AdaDec3D on BTCV and FeTA.
Submit as Paper B to MICCAI 2026 / TMI.

---

## Two-Paper Strategy Summary

| | Paper A | Paper B |
|---|---|---|
| **Claim** | Decoder redundancy is spatially heterogeneous | AdaDec3D exploits this heterogeneity |
| **Contribution** | Empirical observation (O1–O11) | New adaptive decoder method |
| **Venue** | MIDL / MLMI / ISBI | MICCAI 2026 / TMI |
| **Prerequisite** | E0 + E1 trained | Paper A accepted |
| **Key figure** | O9 Pareto curve | Efficiency-accuracy Pareto curve |

---

## Quick Reference

| Document | Purpose |
|---|---|
| [Observation_Study.md](Observation_Study.md) | O1–O11 code, Go/No-Go criteria, deliverable list |
| [Research_Proposal.md](Research_Proposal.md) | Full scientific motivation, architecture design |
| [Experiment-Design-Observation.md](Experiment-Design-Observation.md) | Step-by-step commands for Phase 1 |
| [Experiment-Design-AdaDec3D.md](Experiment-Design-AdaDec3D.md) | Step-by-step commands for Phase 2 |
