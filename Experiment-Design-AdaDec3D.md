# Paper B: AdaDec3D — Complete Experiment Guide

> **Prerequisites**: all Paper A Go/No-Go criteria in [Observation_Study.md](Observation_Study.md) must pass.
> **Scientific motivation and architecture**: [Research_Proposal.md §6](Research_Proposal.md).

---

## Two-Paper Context

| | Paper A | Paper B — this document |
|---|---|---|
| **Output** | Empirical evidence that decoder benefit is spatially heterogeneous | AdaDec3D: adaptive decoder that realizes selective allocation |
| **Gate** | O1–O5 + O9 pass | Paper A accepted; O7, O8, O11 pass |
| **Venue** | MIDL / MLMI / ISBI | MICCAI 2026 / TMI |
| **Key metric** | Opportunity curve (O9) | DICE ≥ EffiDec3D + 0.3% at matched executed MACs |

E0 and E1 checkpoints from Paper A are reused here as baselines.

---

## Part 1: Implementation Files

| File | Purpose |
|---|---|
| [EffiDec3D/networks/adadec3d.py](EffiDec3D/networks/adadec3d.py) | `AdaDec3D_UXNET` model |
| [EffiDec3D/main_train_adadec3d.py](EffiDec3D/main_train_adadec3d.py) | Two-stage training script |

---

## Part 2: Experiment Groups

### Primary experiments

| ID | Name | `--use_moe` | `--use_roi` | Notes |
|---|---|---|---|---|
| E0 | Full 3DUXNET | — | — | Upper bound (from Paper A) |
| E1 | EffiDec3D | — | — | Baseline (from Paper A) |
| E2 | +MoE only | `True` | `False` | Ablation: no ROI |
| E3 | +ROI only | `False` | `True` | Ablation: no MoE |
| **E4** | **AdaDec3D** | `True` | `True` | Full method |

### Required causal controls

Every final table must include these controls to separate adaptation from added parameters or training time.

| ID | Control | Controls for |
|---|---|---|
| C0 | E1 continued training | Additional optimizer steps |
| C1 | Static Expert-S / M / L | Fixed capacity at each executed cost point |
| C2 | Static decoder, param-matched to E4 | Added parameters |
| C3 | Static decoder, FLOP/latency-matched | Added computation |
| C4 | Random ROI, matched crop fraction | Whether localization matters |
| C5 | Boundary ROI, matched crop fraction | Entropy beyond a boundary prior |
| C6 | Dense refinement | Accuracy ceiling without conditional savings |
| C7 | Oracle positive-gain ROI | Analysis-only upper bound (not deployable) |

Use identical splits, optimizer schedules, augmentations, and checkpoint rules.
Run ≥ 3 matched seeds. Select hyperparameters on a dev fold; report on held-out fold.

---

## Part 3: Training Commands

All commands run from `/root/AdaDec3D/EffiDec3D`.
Common args for every run:

```
--root /root/autodl-tmp/btcv-synapse --dataset BTCV8
--cache_rate 1.0 --num_workers 8 --gpu 0
```

### Stage 1 — freeze backbone, train new modules only

```bash
python main_train_adadec3d.py \
  --effidec3d_weights /root/output/E1/.../best_metric_model.pth \
  --output /root/output/E4 \
  --stage 1 --max_iter 20000 --eval_step 500 --lr 5e-4 \
  --root /root/autodl-tmp/btcv-synapse --dataset BTCV8 \
  --cache_rate 1.0 --num_workers 8 --gpu 0
```

Expected at startup:

```
[AdaDec3D] Loaded 312 shared keys from EffiDec3D checkpoint
[AdaDec3D] Missing (new modules, expected): ['roi_refiner.conv.0.weight', ...]
[Stage 1] Trainable: 0.52M / 3.68M params
```

### Stage 2 — end-to-end fine-tune

```bash
python main_train_adadec3d.py \
  --stage1_ckpt /root/output/E4/stage1/.../best_metric_model.pth \
  --output /root/output/E4 \
  --stage 2 --max_iter 25000 --eval_step 500 --lr 5e-4 --backbone_lr_factor 0.1 \
  --root /root/autodl-tmp/btcv-synapse --dataset BTCV8 \
  --cache_rate 1.0 --num_workers 8 --gpu 0
```

Backbone LR = 5e-4 × 0.1 = 5e-5. New modules LR = 5e-4.

### Ablation runs E2 and E3

```bash
# E2: MoE only
python main_train_adadec3d.py \
  --effidec3d_weights /root/output/E1/best_metric_model.pth \
  --output /root/output/E2 \
  --stage 1 --use_moe True --use_roi False \
  --max_iter 20000 --lr 5e-4 \
  --root /root/autodl-tmp/btcv-synapse --dataset BTCV8 \
  --cache_rate 1.0 --num_workers 8 --gpu 0

python main_train_adadec3d.py \
  --stage1_ckpt /root/output/E2/stage1/best_metric_model.pth \
  --output /root/output/E2 \
  --stage 2 --use_moe True --use_roi False \
  --max_iter 25000 --lr 5e-4 \
  --root /root/autodl-tmp/btcv-synapse --dataset BTCV8 \
  --cache_rate 1.0 --num_workers 8 --gpu 0

# E3: ROI only
python main_train_adadec3d.py \
  --effidec3d_weights /root/output/E1/best_metric_model.pth \
  --output /root/output/E3 \
  --stage 1 --use_moe False --use_roi True \
  --max_iter 20000 --lr 5e-4 \
  --root /root/autodl-tmp/btcv-synapse --dataset BTCV8 \
  --cache_rate 1.0 --num_workers 8 --gpu 0

python main_train_adadec3d.py \
  --stage1_ckpt /root/output/E3/stage1/best_metric_model.pth \
  --output /root/output/E3 \
  --stage 2 --use_moe False --use_roi True \
  --max_iter 25000 --lr 5e-4 \
  --root /root/autodl-tmp/btcv-synapse --dataset BTCV8 \
  --cache_rate 1.0 --num_workers 8 --gpu 0
```

---

## Part 4: Loss Terms

| Term | Default λ | Purpose |
|---|---|---|
| `L_seg` | 1.0 | DiceCE on final prediction (upsampled to full resolution) |
| `L_coarse` | 0.5 | Auxiliary DiceCE on coarse decoder — prevents backbone degradation |
| `L_uncertainty` | 0.1 | Calibration: high-entropy voxels correlate with actual errors |
| `L_resource` | 0.05 | Pushes router toward lighter experts when accuracy allows |
| `L_router` | 0.1 | Expected-cost constraint: penalizes exceeding declared expert budget |

All adjustable via `--lambda_uncertainty`, `--lambda_resource`, `--lambda_router`, `--lambda_coarse`.

### Monitor in TensorBoard

```python
%load_ext tensorboard
%tensorboard --logdir /root/output/E4/stage1/BTCV13/tensorboard
```

| Signal | Healthy | Action if unhealthy |
|---|---|---|
| `Loss/router` | Decreasing, stabilises | Expected cost misses budget → tune `--lambda_router` |
| `Loss/unc` | Decreasing | Uncertainty not calibrating → check loss formulation |
| `Loss/seg` stalls + low `Loss/resource` | — | Reduce `--lambda_resource` |

---

## Part 5: Metrics

### 5.1 Segmentation (DICE + HD95)

```python
from monai.metrics import DiceMetric, HausdorffDistanceMetric

dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95)

per_dice = dice_metric.aggregate()
per_hd95 = hd95_metric.aggregate()
BTCV_NAMES = ["Aorta","Gallbladder","Spleen","L.Kidney","R.Kidney",
              "Liver","Stomach","IVC","Port.Vein","Pancreas","R.Adrenal","L.Adrenal","Duodenum"]
for name, d, h in zip(BTCV_NAMES, per_dice, per_hd95):
    print(f"{name:15s}: DICE={d:.4f}  HD95={h:.2f}mm")
```

### 5.2 Efficiency (executed cost — primary)

`ptflops` reports the **dense static graph** (upper bound only). Primary efficiency
results are: hard-routing latency, selected expert, ROI crop fraction, and
executed expert/ROI MACs per subject. Report mean, median, p95 on the same GPU
after warm-up, including routing, crop extraction, and scatter/fusion overhead.
Do not label MACs as FLOPs; state the convention explicitly.

```python
from ptflops import get_model_complexity_info
import time, torch

model.eval()
dummy = torch.randn(1, 1, 96, 96, 96).cuda()

macs, params = get_model_complexity_info(model, (1,96,96,96),
    as_strings=True, print_per_layer_stat=False, verbose=False)
print(f"Static upper bound: {macs}  params: {params}")

# End-to-end latency (50 runs after 10 warm-up)
times = []
with torch.no_grad():
    for i in range(60):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        _ = model(dummy)
        torch.cuda.synchronize()
        if i >= 10: times.append(time.perf_counter() - t0)
print(f"Latency: {np.mean(times)*1000:.1f} ms  p95: {np.percentile(times,95)*1000:.1f} ms")

torch.cuda.reset_peak_memory_stats()
with torch.no_grad(): _ = model(dummy)
print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()//1024**2:.0f} MB")
```

### 5.3 AdaDec3D-specific diagnostics

#### Expert routing distribution

```python
expert_hist = [0, 0, 0]
with torch.no_grad():
    for batch in val_loader:
        pred, extras = model(batch["image"].cuda(), return_router=True)
        expert_hist[extras["router_weights"].argmax(dim=1).item()] += 1

total = sum(expert_hist)
print(f"Expert-S (32ch): {expert_hist[0]/total:.1%}")
print(f"Expert-M (64ch): {expert_hist[1]/total:.1%}")
print(f"Expert-L (96ch): {expert_hist[2]/total:.1%}")
```

**Target**: Expert-L preferentially activated for hard samples (Pancreas, Adrenal).
If all samples route to Expert-S → increase `--lambda_router`.

#### ROI coverage of small organs

```python
import torch.nn.functional as F

SMALL_ORGANS = {"Gallbladder":2, "Pancreas":10, "R.Adrenal":11,
                "L.Adrenal":12, "Duodenum":13}
coverage, missed = {n: [] for n in SMALL_ORGANS}, {n: 0 for n in SMALL_ORGANS}

with torch.no_grad():
    for batch in val_loader:
        lbl = batch["label"]
        pred, extras = model(batch["image"].cuda(), return_roi=True)
        roi_full = F.interpolate(extras["roi_mask"].float(), size=lbl.shape[2:], mode="nearest")
        for name, organ_id in SMALL_ORGANS.items():
            organ_vox = (lbl == organ_id).float()
            if organ_vox.sum() == 0:
                missed[name] += 1   # completely absent from this scan
                continue
            covered = (organ_vox * roi_full.cpu()).sum() / organ_vox.sum()
            coverage[name].append(covered.item())

for name in SMALL_ORGANS:
    cov = np.mean(coverage[name]) if coverage[name] else float('nan')
    print(f"{name:15s}: coverage={cov:.1%}  missed_scans={missed[name]}")
```

**Target**: coverage > 80%. Also report `missed_scans` — confidently missed organs
receive no ROI and represent the most important failure mode.

#### Uncertainty–error calibration

```python
from scipy.stats import pearsonr

unc_vals, err_vals = [], []
with torch.no_grad():
    for batch in val_loader:
        lbl = batch["label"].cuda().squeeze(1).long()
        pred, extras = model(batch["image"].cuda(), return_uncertainty=True)
        unc = extras["uncertainty"]
        err = (pred.argmax(1) != F.interpolate(
            lbl.unsqueeze(1).float(), size=pred.shape[2:], mode="nearest"
        ).squeeze(1).long()).float()
        unc_vals.append(unc.mean().item())
        err_vals.append(err.mean().item())

r, p = pearsonr(unc_vals, err_vals)
print(f"Uncertainty-Error Pearson r={r:.3f}  p={p:.4f}")
```

Primary routing metric is counterfactual net decoder benefit (risk–coverage curves,
AUROC/AUPRC against positive-transition regions). The uncertainty–error correlation
above is a secondary diagnostic. Also compare entropy against random, boundary,
foreground, confidence, and organ-size signals at identical selection budgets.

### 5.4 Final result table format

```
Table: BTCV 13-Organ Segmentation (Synapse Dataset)

Method        Params  GFLOPs*  Mean   Aorta  Gallb  Splen  LKid  RKid
Full 3DUXNET  53M     632.0    79.74
EffiDec3D     3.16M   51.47    79.25
E2 +MoE       —       —        —
E3 +ROI       —       —        —
E4 AdaDec3D   X.XM    XX.X†    XX.XX
C0 E1+steps   —       51.47    —
C1 StaticL    —       —        —

* static upper-bound MACs
† executed MACs vary per sample; report mean ± std
```

Key claims:
- `Pancreas`: target ≥ +1% over EffiDec3D
- `R.Adrenal`, `L.Adrenal`: target ≥ +1%
- Mean DICE: ≥ EffiDec3D + 0.3%
- Executed MACs: ≤ EffiDec3D × 1.3 on average

---

## Part 6: Hyperparameter Ablations

Run after E4 is stable.

```bash
# ROI coverage threshold: fraction of voxels refined
python main_train_adadec3d.py --roi_quantile 0.25 ...  # top 75% (broad)
python main_train_adadec3d.py --roi_quantile 0.50 ...  # default: top 50%
python main_train_adadec3d.py --roi_quantile 0.75 ...  # top 25% (tight)

# Resource penalty
python main_train_adadec3d.py --lambda_resource 0.01 ...  # accuracy priority
python main_train_adadec3d.py --lambda_resource 0.05 ...  # default
python main_train_adadec3d.py --lambda_resource 0.20 ...  # efficiency priority
```

Plot executed MACs vs mean DICE for the three `lambda_resource` values →
**efficiency-accuracy Pareto curve** (key figure for Paper B).

### Post-training analysis (all extras in one pass)

```python
from networks.adadec3d import AdaDec3D_UXNET

model = AdaDec3D_UXNET(out_chans=14).cuda()
ckpt = torch.load("/path/to/E4/stage2/best_metric_model.pth")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

with torch.no_grad():
    pred, extras = model(img,
                         return_router=True,
                         return_roi=True,
                         return_uncertainty=True)
    router_w = extras["router_weights"]   # [B, 3]
    roi_mask  = extras["roi_mask"]        # [B, 1, D/2, H/2, W/2]
    unc_map   = extras["uncertainty"]     # [B, D/2, H/2, W/2]
```

---

## Part 7: Timeline — Phase 2 (Paper B)

```
Week 8-9: Stage 1 training
  [ ] C0: E1 continued training (controls for extra steps)
  [ ] C1: Static Expert-S / M / L (controls for fixed capacity)
  [ ] E2: +MoE only (20 000 iter — verify expert distribution not collapsed)
  [ ] E3: +ROI only (20 000 iter — verify ROI coverage > 80%)

Week 10-11: Stage 2 fine-tuning
  [ ] E4: Full AdaDec3D (25 000 iter)
  [ ] C2, C3: param-matched and FLOP-matched static decoders
  [ ] C4, C5: random and boundary ROI controls

Week 12: Hyperparameter ablations
  [ ] ROI quantile ablation (0.25 / 0.50 / 0.75)
  [ ] lambda_resource ablation → Pareto curve
  [ ] FeTA: run E1_feta and E4_feta

Week 13-14: Paper B writing
  [ ] Table 1: E0–E4 + C0–C7 quantitative comparison (BTCV)
  [ ] Table 2: Hyperparameter ablations
  [ ] Table 3: FeTA cross-dataset results
  [ ] Fig: Executed-MACs vs DICE Pareto curve
  [ ] Fig: Uncertainty map + ROI mask + routing distribution
  [ ] MICCAI 2026 submission (typically January deadline)
```

---

## Part 8: Go / No-Go Criteria

### Go: MICCAI oral

- Mean DICE ≥ EffiDec3D + 0.5%
- Pancreas + Adrenal mean DICE ≥ +1.5%
- Executed MACs ≤ EffiDec3D × 1.3 (measured, not inferred from soft routing)
- E4 > C0, C1, C2, C3 on at least 2 of 3 matched seeds

### Go: MICCAI poster / JBHI

- Mean DICE ≥ EffiDec3D + 0.2%
- ≥ 2 small organ classes with ≥ +1% DICE
- ROI coverage > 80% for small organs

### No-Go: debug first

| Symptom | Likely cause | Fix |
|---|---|---|
| DICE < EffiDec3D | Expert collapse | Increase `--lambda_router` |
| All samples → Expert-S | Load imbalance | Increase `--lambda_router` |
| ROI coverage < 50% | ROI threshold too tight | Decrease `--roi_quantile` |
| E4 ≤ C0 (continued training) | Gains from steps, not adaptation | Re-examine routing effectiveness |
| No latency saving | Routing overhead dominates | Profile crop extraction and scatter ops |
