# Paper B: Utility-Aware Adaptive Decoding — Can Flip Direction Be Learned?

> **Prerequisite**: Paper A ([Observation_Study.md](Observation_Study.md)) is complete.
> This proposal **supersedes** the old efficiency-first AdaDec3D design and its
> `Research_Proposal.md §6` architecture.

---

## 0. Why the old AdaDec3D was abandoned

The old plan (input-adaptive decoder for **lower executed MACs**) is a weak paper *because
of Paper A's own results*:

- **Net-neutral** (Full ≈ Effi; gain ~0.05–0.07%) → no accuracy story.
- **EffiDec3D already at the floor** (49.5 GMac; encoder backbone 7.7 GMac is irreducible) →
  the MAC prize is only ~20–30%, and net-neutrality means a *static* thin decoder likely
  matches, so adaptivity buys little.
- **Direction unpredictable** (AUROC 0.52) → you cannot route compute to where it *helps*.

So the efficiency angle is, at best, a conditional mid-tier result. **This proposal pivots to
the one question that actually decides whether adaptive decoding is possible at all.**

---

## 1. Thesis

Paper A established a sharp asymmetry in what predictive uncertainty knows about decoder
computation:

| | Paper A result | Meaning |
|---|---|---|
| **Location** — *where* the decoder acts | AUROC **0.865–0.913** | entropy finds active regions |
| **Direction** — *whether* it helps | AUROC **0.520–0.591** (TTA-MI 0.539–0.570) | ≈ chance; entropy is blind to sign |

Every adaptive-decoding method dies on that 0.52: you can localize decoder activity but cannot
tell corrections from degradations, so you cannot allocate compute to net-positive regions.
But Paper A only tested **unsupervised uncertainty proxies** (entropy, TTA-MI). It never asked:

> **Central question.** Can a *learned, supervised* predictor recover flip **direction** from
> the efficient model's own features, beating the 0.52 uncertainty ceiling — without ever
> running the full decoder at inference?

### The fork (both outcomes are publishable)

| Learned direction AUROC | Paper B becomes | Venue |
|---|---|---|
| **≥ ~0.70** + net-positive selective routing | **Utility-aware adaptive decoding**: apply Full only where predicted to *help*, skip where it *hurts* → **net-positive accuracy** (breaks net-neutrality) at a fraction of Full compute. A genuine method. | MICCAI / TMI |
| **~0.55–0.65** | Marginal routing improvement over entropy; efficiency-only fallback. | JBHI / poster |
| **~0.52** | **Fundamental-limit negative result**: decoder-benefit direction is not recoverable from single-model features → adaptive decoding is inherently capped. Closes Paper A's central gap. | MIDL / short paper |

**This is a fail-fast program.** One number (the held-out direction AUROC) decides the whole
direction before any architecture engineering.

---

## 2. The learned direction predictor

### 2.1 Target labels (reuse Paper A)

At each voxel `v`, from the frozen (E0, E1, GT) triple that `run_observations.py` already
computes:

- `P(v) = 1[ŷ_e(v) ≠ y(v) ∧ ŷ_f(v) = y(v)]` — **positive correction** (Full fixes an Effi error)
- `N(v) = 1[ŷ_e(v) = y(v) ∧ ŷ_f(v) ≠ y(v)]` — **negative degradation** (Full breaks an Effi hit)
- else — **no-flip** (ignored for the direction task)

### 2.2 The non-negotiable constraint — no leakage

The head may consume **only the efficient model's representations**. `ŷ_f`, `y`, `P`, `N` are
**training targets only**; none may enter the head at inference. This is the entire point: if
E1's features carried direction, entropy (a scalar summary of E1's logits) should already do
better than 0.52 — so the head must extract something entropy discards. **Assert** in code that
head inputs derive only from E1.

### 2.3 Granularity

Evaluate at **block level** (the Paper A block partition), so the number is directly comparable
to the 0.52 baseline: aggregate per-voxel head scores to blocks, AUROC on **non-zero-gain
blocks** predicting `sign(G(r))>0`. Train per-voxel, pool to blocks.

### 2.4 Architecture (small on purpose)

Input `F_e` = the efficient decoder's **last feature map** (~48 ch at output resolution),
captured with a forward hook on the frozen E1. The head is deliberately small — the claim is
about *information in the features*, not head capacity.

```python
class DirectionHead(nn.Module):          # ~0.1–0.5M params
    def __init__(self, in_ch=48, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, h, 3, padding=1), nn.InstanceNorm3d(h), nn.LeakyReLU(0.01),
            nn.Conv3d(h, h, 3, padding=1),     nn.InstanceNorm3d(h), nn.LeakyReLU(0.01),
            nn.Conv3d(h, 1, 1))              # per-voxel direction logit u(v)
    def forward(self, feat_e): return self.net(feat_e)     # [B,1,D,H,W]
```

---

## 3. Loss design

### 3.1 Discriminative loss `L_dir` — answers "is direction predictable at all?"

Binary CE on **flip voxels only** (target = `P`, i.e. 1 for correction, 0 for degradation).
Net-neutrality makes |P|≈|N|, so classes are ~balanced; keep weights for safety, add focal
modulation if easy flips dominate.

```python
def L_dir(u, P, N, w_pos=1., w_neg=1.):
    m = (P + N).clamp(max=1)                       # flip mask
    p = torch.sigmoid(u)
    bce = F.binary_cross_entropy(p, P.float(), reduction='none')
    w = torch.where(P > 0, w_pos, w_neg)
    return (bce * w * m).sum() / m.sum().clamp(min=1)
```

`L_dir` → the **direction AUROC** that decides the program.

### 3.2 Decision/utility loss `L_util` — the deployable "penalty"

BCE ignores the asymmetric cost of routing: **routing a degradation (N) is actively harmful
(−1), while skipping a correction (P) is a mere missed gain (0)**. The deployment loss must
encode that. Let `r(v)=σ(u(v))` be the route probability; realized correctness change from
routing is exactly `G(v)=P(v)−N(v)`. Maximize net gain under a budget `B`, with a
**harm-aversion penalty β>1** over-weighting routes to degradations:

```python
def L_util(u, P, N, budget=0.2, lam=1.0, beta=2.0):
    r = torch.sigmoid(u)
    G = P.float() - beta * N.float()               # +1 for P, -beta for N, 0 else
    gain = (r * G).mean()                          # want high
    frac = r.mean()                                # executed budget fraction
    return -gain + lam * (frac - budget)**2        # -gain + budget penalty
```

The three knobs **are** the design:
- **`β` (harm aversion)** — core penalty. Given net-neutrality (P≈N), you *need* `β>1` or the
  expected routing gain is ~0. Sweep β; read the risk–coverage curve.
- **`λ_budget`** — pins executed compute to the routing budget.
- **`B`** — the 5/10/20% operating points from Paper A.

Deployment operates at **block/tile** level (routing is per-tile): pool `u` over the block,
`G(r)=Σ_v G(v)`, run `L_util` on blocks. Full objective: `L = L_dir + α·L_util`
(`L_dir` teaches direction, `L_util` teaches budget-aware, harm-averse routing).

---

## 4. Dataset construction (one obs pass, reuses Paper A)

For each training subject, run frozen E0 & E1 once and cache:
`F_e` (E1 last feature), `entropy(E1)`, `P`, `N`, and the Paper A block index.
This extends `run_observations.py` with a `--dump_direction_cache <dir>` flag (writes
`.npz` per subject). Split **by subject** into train/val/test.

Sanity controls baked into the cache build:
- **Shuffled-label control**: train on permuted P/N → must give AUROC ≈ 0.50 (rules out leakage/overfit artifacts).
- **Entropy-only head**: input = entropy scalar map → must reproduce ≈ 0.52 (confirms the setup and the baseline).

---

## 5. Feature-source ablation (half the paper)

Which representation, if any, carries direction? Report held-out block-direction AUROC (subject
bootstrap CI) for each input:

| Input to head | Expectation |
|---|---|
| entropy scalar (baseline) | ≈ 0.52 (sanity) |
| E1 logits (C classes) | ? |
| **E1 last decoder feature `F_e`** | the headline candidate |
| `F_e` + input image patch | more context |
| multi-scale E1 features | richest |

Whichever first beats 0.52 localizes *where* the direction signal lives (if anywhere) — a
finding in its own right, independent of the method.

---

## 6. Evaluation protocol

1. **Sanity**: entropy-only head ≈ 0.52; shuffled-label ≈ 0.50.
2. **Headline**: block-level **direction AUROC / AUPRC** with `F_e`, vs 0.52 and vs oracle (=1.0),
   subject-bootstrap CI. Run on **held-out test subjects**.
3. **Generalization**: repeat on SwinUNETR & MedNeXt (Paper A backbones) and MSD01/MSD08 — is
   the direction signal architecture/modality-general or specific?
4. **Downstream selective-routing curve** (the method test): at 5/10/20% budgets, net Dice from
   **head-routing** vs entropy vs random vs boundary vs **oracle (C7)**. Net-positive Dice
   (beating *both* Full and Effi) is the accuracy win that breaks net-neutrality.

---

## 7. The method — utility-gated adaptive decoding (only if §6.2 passes)

If the learned head clears the AUROC bar, wire it as the routing signal:

1. Efficient decoder E1 runs densely → base prediction + `F_e`.
2. **DirectionHead(`F_e`)** → per-block utility `u(r)`.
3. Route the **full-capacity decoder path** to the top-`B` blocks by `u(r)` (harm-averse: only
   positive-predicted blocks), fuse back.
4. Report **net Dice** and **executed MACs** vs all controls below.

This reuses the routing/ROI/budget backend from the old `adadec3d.py` (the *only* salvageable
part) — but the gate is now a **learned utility predictor**, not entropy.

---

## 8. Controls & baselines (every final table)

| ID | Control | Isolates |
|---|---|---|
| B-entropy | Route by entropy (Paper A signal, 0.52 direction) | value of the learned head over uncertainty |
| B-random | Random blocks, matched budget | whether routing matters at all |
| B-boundary | Boundary-distance prior, matched budget | learned direction beyond a geometric prior |
| B-oracle | Route by true `G(r)` (C7) | analysis-only upper bound |
| B-static | Static decoder at matched executed MACs | adaptation vs fixed capacity |
| B-shuffle | Head trained on permuted labels | leakage / overfit artifact (must ≈ chance) |

Subject-split; ≥3 matched seeds; hyperparameters on a dev fold, report on held-out fold.

---

## 9. Go / No-Go

**Go — utility-aware method (strong):**
- Held-out block-direction AUROC **≥ 0.70** with `F_e` (CI excludes 0.52), on ≥2 backbones.
- Selective head-routing yields **net-positive Dice** (> Full and > Effi) at ≤20% budget.
- Beats B-entropy, B-random, B-boundary at matched budget; approaches B-oracle.

**Go — negative-result / limit paper (still valuable):**
- AUROC stays ≈ 0.52 across all feature sources and backbones, with tight CIs and the
  shuffled/entropy sanity controls passing → *direction is fundamentally unrecoverable*.

**No-Go — do not ship:**
- AUROC 0.55–0.65 *and* no net-positive routing → thin, inconclusive; fold the number into
  Paper A as a short "we also tried a learned predictor" paragraph and stop.

---

## 10. Fail-fast timeline

```
Week 1  Build direction cache (--dump_direction_cache) on BTCV/UX-Net; sanity controls
        (entropy≈0.52, shuffle≈0.50).
Week 2  Train DirectionHead + feature-source ablation → HELD-OUT DIRECTION AUROC.
        *** DECISION POINT: the fork in §1 is resolved here. ***
Week 3  If ≥0.70: generalization (Swin/MedNeXt, MSD01/08) + selective-routing curves.
        If ≈0.52: lock the negative result, expand feature/robustness controls.
Week 4  If method alive: wire utility gate into the routing backend; controls B-*.
Week 5  Ablations (β, budget, feature source); Pareto/risk-coverage figures.
Week 6+ Write-up.
```

The Week-2 AUROC is the cheapest decisive experiment in the whole two-paper arc: it reuses the
frozen Paper A checkpoints and flip labels, trains a <0.5M-param head, and needs no new
segmentation training.

---

## 11. Reused assets & what changes

| Asset | Reuse |
|---|---|
| E0/E1 checkpoints (Paper A, all cells) | frozen pair — the head's supervision source |
| `run_observations.py` P/N labels + block index + direction AUROC | dataset build + eval (comparable to 0.52) |
| `adadec3d.py` routing/ROI/budget backend | **only if the method proceeds** — gate swapped to the learned head |
| old efficiency thesis, MoE experts, recovery targets | **dropped** |

> **Note on legacy code**: `EffiDec3D/networks/adadec3d.py` and `main_train_adadec3d.py`
> are kept for now as the routing backend, but their v1/v2 architecture and loss terms are
> superseded. They should be refactored (not extended) if §7 proceeds.
