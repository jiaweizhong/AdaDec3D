# Experiment Design — AdaDec3D Training

> **Scope**: AdaDec3D training experiments, metrics, ablations, and Paper B timeline.
> **Prerequisites**: All Go/No-Go criteria in [Experiment-Design-Observation.md](Experiment-Design-Observation.md) must pass.
> For AdaDec3D architecture see [Research_Proposal.md §6](Research_Proposal.md).
> For observation study analysis see [Observation_Study.md](Observation_Study.md).

---

## Part 4: AdaDec3D Experiments

**Implemented files** (ready to use):
- [EffiDec3D/networks/adadec3d.py](EffiDec3D/networks/adadec3d.py) — `AdaDec3D_UXNET` model
- [EffiDec3D/main_train_adadec3d.py](EffiDec3D/main_train_adadec3d.py) — two-stage training script

### Step 4.1: Experiment groups

| ID | Name | `--use_moe` | `--use_roi` | Notes |
|---|---|---|---|---|
| E0 | Full 3DUXNET | — | — | Upper bound (from observation study) |
| E1 | EffiDec3D | — | — | Baseline (from observation study) |
| E2 | +MoE only | `True` | `False` | Ablation: ROI disabled |
| E3 | +ROI only | `False` | `True` | Ablation: MoE disabled |
| **E4** | **AdaDec3D** | `True` | `True` | Full method |

### Step 4.1a: Mandatory causal controls

E2–E4 alone cannot establish that adaptation, rather than added parameters or
training time, causes improvement. Every final table must also include:

| ID | Control | Purpose |
|---|---|---|
| C0 | E1 continued training | Matches E4's additional optimizer steps |
| C1 | Static Expert-S/M/L | Tests fixed capacity at each executed cost |
| C2 | Static decoder, parameter-matched | Controls for added parameters |
| C3 | Static decoder, FLOP/latency-matched | Controls for added computation |
| C4 | Random ROI, matched crop fraction | Tests whether localization matters |
| C5 | Boundary ROI, matched crop fraction | Tests entropy beyond a simple boundary prior |
| C6 | Dense refinement | Measures the accuracy ceiling without conditional savings |
| C7 | Oracle positive-gain ROI | Analysis-only upper bound; never a deployable baseline |

Use the same split, optimizer schedule, augmentations, checkpoint rule, and total
data exposure. Run at least three matched seeds. Select hyperparameters on a
development fold and report the locked configuration on a held-out fold or via
nested cross-validation.

### Step 4.2: Stage 1 — train new modules, backbone frozen

Requires E1's `best_metric_model.pth`. Freezes
`uxnet_3d / encoder2-5 / decoder3-5 / coarse_out`, trains only the router, experts, and ROI refiner.

```bash
cd /kaggle/input/adadec3d-code/EffiDec3D

python main_train_adadec3d.py \
  --root /kaggle/input/btcv-synapse \
  --output /kaggle/working/output/E4 \
  --dataset BTCV13 \
  --effidec3d_weights /kaggle/input/e1-checkpoint/best_metric_model.pth \
  --stage 1 \
  --max_iter 20000 \
  --eval_step 500 \
  --lr 5e-4 \
  --cache_rate 0.5 \
  --num_workers 2 \
  --gpu 0
```

Expected console output at startup:

```
[AdaDec3D] Loaded 312 shared keys from EffiDec3D checkpoint
[AdaDec3D] Missing (new modules, expected): ['roi_refiner.conv.0.weight', ...]
[Stage 1] Trainable: 0.52M / 3.68M params
```

Trainable params ≈ 0.5M (router + experts + roi_refiner only).

### Step 4.3: Stage 2 — end-to-end fine-tune

```bash
python main_train_adadec3d.py \
  --root /kaggle/input/btcv-synapse \
  --output /kaggle/working/output/E4 \
  --dataset BTCV13 \
  --stage1_ckpt /kaggle/input/e4-stage1/best_metric_model.pth \
  --stage 2 \
  --max_iter 25000 \
  --eval_step 500 \
  --lr 5e-4 \
  --backbone_lr_factor 0.1 \
  --cache_rate 0.5 \
  --num_workers 2 \
  --gpu 0
```

Backbone trains at `5e-4 × 0.1 = 5e-5`. New modules train at `5e-4`.

### Step 4.4: Ablation runs (E2 and E3)

```bash
# E2: MoE only — disable ROI refinement
# Stage 1
python main_train_adadec3d.py \
  --effidec3d_weights /path/to/E1/best_metric_model.pth \
  --output /kaggle/working/output/E2 \
  --stage 1 --use_moe True --use_roi False \
  --max_iter 20000 --lr 5e-4 \
  --root /kaggle/input/btcv-synapse --dataset BTCV13 \
  --cache_rate 0.5 --num_workers 2 --gpu 0

# Stage 2 for E2
python main_train_adadec3d.py \
  --stage1_ckpt /path/to/E2/stage1/best_metric_model.pth \
  --output /kaggle/working/output/E2 \
  --stage 2 --use_moe True --use_roi False \
  --max_iter 25000 --lr 5e-4 \
  --root /kaggle/input/btcv-synapse --dataset BTCV13 \
  --cache_rate 0.5 --num_workers 2 --gpu 0

# E3: ROI only — disable MoE (always uses Expert-M)
# Stage 1
python main_train_adadec3d.py \
  --effidec3d_weights /path/to/E1/best_metric_model.pth \
  --output /kaggle/working/output/E3 \
  --stage 1 --use_moe False --use_roi True \
  --max_iter 20000 --lr 5e-4 \
  --root /kaggle/input/btcv-synapse --dataset BTCV13 \
  --cache_rate 0.5 --num_workers 2 --gpu 0

# Stage 2 for E3
python main_train_adadec3d.py \
  --stage1_ckpt /path/to/E3/stage1/best_metric_model.pth \
  --output /kaggle/working/output/E3 \
  --stage 2 --use_moe False --use_roi True \
  --max_iter 25000 --lr 5e-4 \
  --root /kaggle/input/btcv-synapse --dataset BTCV13 \
  --cache_rate 0.5 --num_workers 2 --gpu 0
```

### Step 4.5: Loss terms

| Term | Weight | Purpose |
|---|---|---|
| `L_seg` | 1.0 | DiceCE on final prediction (interpolated to full resolution) |
| `L_coarse` | 0.5 | Auxiliary DiceCE on coarse decoder output — prevents backbone degradation |
| `L_uncertainty` | 0.1 | Calibration: high-entropy voxels should correlate with actual errors |
| `L_resource` | 0.05 | Pushes router toward lighter experts when accuracy allows |
| `L_router` | 0.1 | Expected-cost constraint: penalizes exceeding a declared normalized expert budget without forcing uniform use |

All weights adjustable via `--lambda_uncertainty`, `--lambda_resource`, `--lambda_router`, `--lambda_coarse`.

### Step 4.6: Monitor training loss terms

```python
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/output/E4/stage1/BTCV13/tensorboard
```

Watch for:
- `Loss/router` should decrease and stabilise — expected cost stays at or below the declared budget
- `Loss/unc` should decrease — uncertainty is calibrating
- If expected cost misses its target → tune `--lambda_router`; do not force uniform expert use
- If `Loss/seg` stalls with low `Loss/resource` → reduce `--lambda_resource`

---

## Part 5: Metrics — What to Measure and Why

### 5.1 Standard segmentation metrics

```python
from monai.metrics import DiceMetric, HausdorffDistanceMetric

dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95)

per_class_dice = dice_metric.aggregate()   # tensor [n_classes]
per_class_hd95 = hd95_metric.aggregate()  # tensor [n_classes]

BTCV_CLASS_NAMES = [
    "Aorta", "Gallbladder", "Spleen", "L.Kidney", "R.Kidney",
    "Liver", "Stomach", "IVC", "Port.Vein",
    "Pancreas", "R.Adrenal", "L.Adrenal", "Duodenum"
]
for name, d, h in zip(BTCV_CLASS_NAMES, per_class_dice, per_class_hd95):
    print(f"{name:15s}: DICE={d:.4f}  HD95={h:.2f}mm")
```

### 5.2 Efficiency metrics

`ptflops` reports the dense static graph and is retained only as a reproducible
upper bound. The primary efficiency results are hard-routing inference latency,
selected expert, ROI crop fraction, and executed expert/ROI MACs per subject.
Report mean, median, and 95th percentile on the same GPU after warm-up, including
sliding-window inference, routing, crop extraction, and scatter/fusion overhead.
Do not label MACs as FLOPs; state the convention explicitly.

```python
from ptflops import get_model_complexity_info
import time, torch

model.eval()
dummy = torch.randn(1, 1, 96, 96, 96).cuda()

macs, params = get_model_complexity_info(
    model, (1, 96, 96, 96),
    as_strings=True, print_per_layer_stat=False, verbose=False
)
print(f"GFLOPs: {macs}")
print(f"Params: {params}")

times = []
with torch.no_grad():
    for i in range(60):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(dummy)
        torch.cuda.synchronize()
        if i >= 10:
            times.append(time.perf_counter() - t0)
print(f"Latency: {sum(times)/len(times)*1000:.1f} ms (mean over 50 runs)")

torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    _ = model(dummy)
peak_mem = torch.cuda.max_memory_allocated() / 1024**2
print(f"Peak GPU Memory: {peak_mem:.0f} MB")
```

### 5.3 AdaDec3D-specific metrics

#### A: Expert activation distribution

```python
expert_hist = [0, 0, 0]  # Expert-S, M, L

with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].cuda()
        lbl = batch["label"].cuda()
        pred, extras = model(img, return_router=True)
        router_w = extras["router_weights"]   # [B, 3]
        expert_id = router_w.argmax(dim=1).item()
        expert_hist[expert_id] += 1

total = sum(expert_hist)
print(f"Expert-S (32ch): {expert_hist[0]/total:.1%}")
print(f"Expert-M (64ch): {expert_hist[1]/total:.1%}")
print(f"Expert-L (96ch): {expert_hist[2]/total:.1%}")
```

**Target**: Expert-L activated more for hard samples (gallbladder, pancreas, adrenal). If all samples use Expert-S, increase `--lambda_router`.

#### B: ROI coverage of small organs

```python
import torch.nn.functional as F

SMALL_ORGAN_LABELS = {
    "Gallbladder": 2, "Pancreas": 10, "R.Adrenal": 11,
    "L.Adrenal": 12, "Duodenum": 13
}
coverage_stats = {name: [] for name in SMALL_ORGAN_LABELS}

with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].cuda()
        lbl = batch["label"]
        pred, extras = model(img, return_roi=True)
        roi_mask = extras["roi_mask"]   # [B, 1, D/2, H/2, W/2]
        roi_full = F.interpolate(roi_mask.float(), size=lbl.shape[2:], mode="nearest")
        for name, organ_id in SMALL_ORGAN_LABELS.items():
            organ_vox = (lbl == organ_id).float()
            if organ_vox.sum() == 0:
                continue
            covered = (organ_vox * roi_full.cpu()).sum()
            coverage_stats[name].append((covered / organ_vox.sum()).item())

for name, vals in coverage_stats.items():
    if vals:
        print(f"{name:15s}: ROI coverage = {sum(vals)/len(vals):.1%}")
```

**Target**: Coverage > 80% for all small organs.

Also report complete-organ miss rate and the fraction of positive decoder
transitions outside the ROI. High coverage among detected organs can hide the
most important failure mode: a confidently missed small organ receives no ROI.

#### C: Uncertainty–error calibration

```python
from scipy.stats import pearsonr
import numpy as np

unc_vals, err_vals = [], []

with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].cuda()
        lbl = batch["label"].cuda().squeeze(1).long()
        pred, extras = model(img, return_uncertainty=True)
        unc_map = extras["uncertainty"]   # [B, D/2, H/2, W/2]
        error = (pred.argmax(1) != F.interpolate(
            lbl.unsqueeze(1).float(), size=pred.shape[2:], mode="nearest"
        ).squeeze(1).long()).float()
        unc_vals.append(unc_map.mean().item())
        err_vals.append(error.mean().item())

r, p = pearsonr(unc_vals, err_vals)
print(f"Uncertainty-Error Pearson r = {r:.3f}, p-value = {p:.4f}")
```

**Target**: r > 0.60.

This correlation is secondary. The primary routing metric is prediction of
counterfactual net decoder benefit, evaluated with risk–coverage curves and
AUROC/AUPRC against positive net-benefit regions. Report ECE and Brier score for
calibration, and compare entropy with random, boundary, foreground, confidence,
and organ-size signals at identical selection budgets.

### 5.4 Final result table format

```
Table: BTCV 13-Organ Segmentation on Synapse Dataset

Method         Params  GFLOPs  Mean   Aorta  Gallb  Splen  LKid  RKid  Liver  Stom  IVC  PVein  Pancr  RAdG  LAdG  Duod
Full 3DUXNET   53M     632.0   79.74
EffiDec3D      3.16M   51.47   79.25
AdaDec3D(E4)   X.XM    XX.X    XX.XX
```

Key columns for AdaDec3D's contribution:
- `Pancr`: EffiDec3D ~54%, target ≥55%
- `RAdG`, `LAdG`: EffiDec3D ~63%, target ≥64%
- `Mean`: AdaDec3D ≥ EffiDec3D + 0.3% minimum

---

## Part 6: Ablation Studies

### 6.1 Module contribution ablation

See Step 4.4 above for full commands. Summary of expected outcomes:

| Experiment | MoE | ROI | Expected mean DICE | Interpretation |
|---|---|---|---|---|
| E1 EffiDec3D | ✗ | ✗ | 79.25% | Baseline |
| E2 +MoE | ✓ | ✗ | +0.1-0.3% | Routing helps for hard samples |
| E3 +ROI | ✗ | ✓ | +0.1-0.3% | Refinement helps for small organs |
| E4 AdaDec3D | ✓ | ✓ | +0.3-0.7% | Both modules together |

### 6.2 Hyperparameter ablation (run after E4 is stable)

```bash
# ROI coverage threshold
python main_train_adadec3d.py --roi_quantile 0.25 ...   # refine top 75% (broad)
python main_train_adadec3d.py --roi_quantile 0.50 ...   # default: top 50%
python main_train_adadec3d.py --roi_quantile 0.75 ...   # refine top 25% (tight)

# Resource penalty weight — accuracy vs efficiency
python main_train_adadec3d.py --lambda_resource 0.01 ...  # accuracy priority
python main_train_adadec3d.py --lambda_resource 0.05 ...  # default
python main_train_adadec3d.py --lambda_resource 0.20 ...  # efficiency priority
```

Plot GFLOPs vs mean DICE for the three `lambda_resource` values → efficiency-accuracy Pareto curve.

### 6.3 Post-training analysis

```python
import torch, torch.nn.functional as F
from networks.adadec3d import AdaDec3D_UXNET

model = AdaDec3D_UXNET(out_chans=14).cuda()
ckpt = torch.load("/path/to/E4/stage2/best_metric_model.pth")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# All analysis extras in one pass
with torch.no_grad():
    pred, extras = model(img,
                         return_router=True,
                         return_roi=True,
                         return_uncertainty=True)
    router_w = extras["router_weights"]   # [B, 3]
    roi_mask = extras["roi_mask"]         # [B, 1, D/2, H/2, W/2]
    unc_map  = extras["uncertainty"]      # [B, D/2, H/2, W/2]
```

---

## Part 7 (Phase 2): Execution Timeline

```
Week 8-9: Stage 1 training
  [ ] E2: +MoE only (20000 iter — check expert distribution not collapsed)
  [ ] E3: +ROI only (20000 iter — check ROI coverage > 80% for small organs)
  [ ] If expert collapse: increase lambda_router from 0.1 to 0.3

Week 10-11: Stage 2 fine-tuning + ablations
  [ ] E4 Stage 2: Full AdaDec3D (25000 iter)
  [ ] Run ROI quantile ablation (0.25 / 0.50 / 0.75)
  [ ] Run lambda_resource ablation (Pareto curve — key figure for Paper B)

Week 12: FeTA + visualization
  [ ] Prepare FeTA dataset, run E1 and E4 on FeTA
  [ ] Visualize: uncertainty map, ROI mask, expert activation per organ
  [ ] Compute Uncertainty–Error Pearson r on FeTA

Week 13-14: Paper B writing
  [ ] Table 1: E0–E4 quantitative comparison (BTCV)
  [ ] Table 2: Ablation studies (modules + hyperparameters)
  [ ] Table 3: FeTA cross-dataset results
  [ ] Figure: Efficiency-accuracy Pareto curve
  [ ] Figure: Uncertainty map + ROI mask visualization
  [ ] Figure: Expert routing distribution per organ
  [ ] MICCAI 2026 submission (typically January deadline)
```

---

## Part 8: Go / No-Go Criteria for Paper B

### Go: submit to MICCAI (oral-level)

- AdaDec3D mean DICE ≥ EffiDec3D + 0.5%
- Small organs (Pancreas + Adrenal) mean DICE ≥ +1.5%
- GFLOPs ≤ EffiDec3D × 1.3
- Expert-L activated preferentially for hard samples

### Go: submit to MICCAI (poster) or JBHI

- AdaDec3D mean DICE ≥ EffiDec3D + 0.2%
- At least 2 small organ classes with ≥ +1% DICE improvement
- ROI coverage > 80% for small organs

### No-Go: needs debugging

| Symptom | Likely cause | Fix |
|---|---|---|
| DICE below EffiDec3D | Expert collapse | Increase `--lambda_router` from 0.1 to 0.3 |
| All samples → Expert-S | Load imbalance | Increase `--lambda_router` |
| ROI coverage < 50% | ROI threshold too tight | Decrease `--roi_quantile` |
| Uncertainty-Error r < 0.3 | Uncertainty not calibrating | Check loss formulation |
