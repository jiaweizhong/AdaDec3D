# AdaDec3D Experiment Design

---

## 0. Hardware: Kaggle P100

**Hardware**: Kaggle P100 (16 GB VRAM, CUDA). All code is CUDA-native (PyTorch + MONAI + ptflops); no changes needed.

### Timing estimates

```
EffiDec3D (3.16M params, 51.47 GFLOPs):
  ~1-2 sec per iter (4 patches of 96^3)
  45000 iter × 1.5s ≈ 18-20 hours → 2-3 Kaggle sessions

Full 3DUXNET (53M params, 632 GFLOPs):
  ~5-8 sec per iter
  20000 iter × 6s ≈ 33 hours → 4 sessions (use max_iter=20000 for upper-bound baseline)
```

The training script saves `last_model.pth` after every eval step and auto-resumes — download it before each 9-hour session ends, re-upload as a dataset for the next session.

---

## Part 1: Environment Setup

### Step 1.1: Kaggle Notebook file layout

```
/kaggle/
  input/
    btcv-synapse/          <- upload as Kaggle Dataset
      imagesTr/
      labelsTr/
      imagesVal/
      labelsVal/
    adadec3d-code/         <- upload repo as Kaggle Dataset
      EffiDec3D/
      networks/
      ...
  working/
    output/                <- training outputs, download last_model.pth before session ends
```

### Step 1.2: Install dependencies

```bash
pip install monai==1.3.0
pip install batchgenerators
pip install medpy
pip install ptflops
pip install scikit-learn scipy nibabel tqdm
```

Verify GPU:

```python
import torch
print(torch.cuda.is_available())       # True
print(torch.cuda.get_device_name(0))   # Tesla P100-PCIE-16GB
print(torch.cuda.get_device_properties(0).total_memory // 1024**3, "GB")  # 16 GB
```

### Step 1.3: Verify environment before first run

```bash
cd /kaggle/input/adadec3d-code/EffiDec3D

python -c "
import monai, torch
from networks.UXNet_3D.network_backbone import UXNET_EffiDec3D
from networks.swin_unetr_effidec3d import SwinUNETR_EffiDec3D
from monai_utils.inferers.utils import sliding_window_inference_1out
print('All imports OK')
print('MONAI:', monai.__version__)
print('PyTorch:', torch.__version__)
"
```

---

## Part 2: Dataset Setup (Step by Step)

### 2.1 BTCV / Synapse Dataset (primary, directly comparable to paper)

#### Where to get the data

**Option A: Official Synapse (requires registration)**

1. Register at [synapse.org](https://www.synapse.org)
2. Go to Multi-Atlas Abdomen Labeling Challenge: `syn3193805`
3. Download `RawData.zip` (~1.5 GB)

**Option B: TransUNet preprocessed version (recommended)**

The 3D UX-Net / SwinUNet papers all use this preprocessed version.
Search for "Synapse multi-organ segmentation dataset" on Kaggle — several public datasets are available with the correct preprocessing already applied.

#### Expected directory structure

The training script (`load_datasets_transforms.py` line 113-123) expects exactly:

```
/path/to/btcv/
  imagesTr/
    img0001.nii.gz
    img0002.nii.gz
    ...                    <- 18 training cases
  labelsTr/
    label0001.nii.gz       <- must match imagesTr filenames 1:1 when sorted
    label0002.nii.gz
    ...
  imagesVal/
    img0021.nii.gz
    ...                    <- 12 validation cases
  labelsVal/
    label0021.nii.gz
    ...
```

The code uses `sorted(glob(...))` so filenames must sort in the same order for images and labels.

#### Standard BTCV train/val split (same as 3D UX-Net paper)

```python
# Cases used for training (18 cases)
TRAIN = ["0001","0002","0003","0004","0005","0006","0007","0008",
         "0009","0010","0021","0022","0023","0024","0025","0026",
         "0027","0028"]
# Cases used for validation (12 cases)
VAL   = ["0029","0030","0031","0032","0033","0034","0035","0036",
         "0037","0038","0039","0040"]
```

#### Verify data loading

```bash
cd /kaggle/input/adadec3d-code/EffiDec3D

python -c "
import argparse
from load_datasets_transforms import data_loader

args = argparse.Namespace(
    root='/kaggle/input/btcv-synapse',
    dataset='BTCV13',
    mode='train'
)
tr, val, nc = data_loader(args)
print('Train images:', len(tr['images']))   # expect: 18
print('Val images:', len(val['images']))     # expect: 12
print('Output classes:', nc)                # expect: 14 (13 organs + background)
print('First train image:', tr['images'][0])
print('First train label:', tr['labels'][0])
"
```

#### BTCV13 label mapping

```
Label 0:  Background
Label 1:  Aorta
Label 2:  Gallbladder
Label 3:  Spleen
Label 4:  Left Kidney
Label 5:  Right Kidney
Label 6:  Liver
Label 7:  Stomach
Label 8:  Inferior Vena Cava (IVC)
Label 9:  Portal and Splenic Vein
Label 10: Pancreas            <- small structure, key metric
Label 11: Right Adrenal Gland <- small structure, key metric
Label 12: Left Adrenal Gland  <- small structure, key metric
Label 13: Duodenum
```

**Small structures** (labels 2, 10, 11, 12, 13) are where AdaDec3D must outperform EffiDec3D.

### 2.2 FeTA 2021 Dataset (second priority, small structures)

#### Where to get the data

1. Register at [fetachallenge.github.io](https://fetachallenge.github.io)
2. Download `feta_2.2.tar.gz` (~2 GB, 80 fetal brain MRI subjects)

#### Convert to expected format

```python
# run once: prepare_feta.py
import glob, shutil, os

src = "/path/to/feta_2.2"
dst = "/kaggle/input/feta-processed"

subjects = sorted(glob.glob(f"{src}/sub-*/"))
train_subs = subjects[:70]
val_subs   = subjects[70:]

for split, subs in [("Tr", train_subs), ("Val", val_subs)]:
    os.makedirs(f"{dst}/images{split}", exist_ok=True)
    os.makedirs(f"{dst}/labels{split}", exist_ok=True)
    for sub in subs:
        sid = os.path.basename(sub.rstrip("/"))
        shutil.copy(f"{sub}/anat/{sid}_T2w.nii.gz",
                    f"{dst}/images{split}/{sid}.nii.gz")
        shutil.copy(f"{sub}/anat/{sid}_dseg.nii.gz",
                    f"{dst}/labels{split}/{sid}.nii.gz")

print("Train:", len(os.listdir(f"{dst}/imagesTr")))  # 70
print("Val:  ", len(os.listdir(f"{dst}/imagesVal"))) # 10
```

#### FeTA label mapping (7 structures)

```
Label 0: Background
Label 1: Intracranial space (IS)
Label 2: White matter (WM)
Label 3: Cortical grey matter (CGM)
Label 4: Deep grey matter (DGM)     <- small, key metric
Label 5: Cerebellum (CE)
Label 6: Brainstem (BS)
Label 7: Cerebrospinal fluid (CSF)
```

---

## Part 3: Reproduce EffiDec3D Baseline (Step by Step)

### Step 3.1: Train Full Baseline (E0, upper bound)

Run first — establishes the accuracy ceiling to show EffiDec3D's tradeoff.

```bash
cd /kaggle/input/adadec3d-code/EffiDec3D

python main_train_BTCV_TU.py \
  --root /kaggle/input/btcv-synapse \
  --output /kaggle/working/output/E0_full \
  --dataset BTCV13 \
  --img_size 96 96 96 \
  --n_channels 1 \
  --network 3DUXNET \
  --channels 48 96 192 384 \
  --feature_size 48 \
  --ds False \
  --mode train \
  --pretrain False \
  --batch_size 1 \
  --crop_sample 4 \
  --lr 0.001 \
  --optim AdamW \
  --max_iter 20000 \
  --eval_step 500 \
  --val_batch 1 \
  --gpu 0 \
  --cache_rate 0.5 \
  --num_workers 2 \
  --overlap 0.7
```

Notes:
- `max_iter 20000` instead of 45000 — sufficient for upper-bound reference, fits in ~2 Kaggle sessions
- `cache_rate 0.5` — do NOT use 1.0 on Kaggle (RAM is limited to ~13GB, OOM risk)
- `eval_step 500` — less frequent validation to maximize training throughput

### Step 3.2: Train EffiDec3D (E1, the baseline to beat)

```bash
python main_train_BTCV_TU.py \
  --root /kaggle/input/btcv-synapse \
  --output /kaggle/working/output/E1_effidec3d \
  --dataset BTCV13 \
  --img_size 96 96 96 \
  --n_channels 1 \
  --network 3DUXNET_EffiDec3D \
  --channels 48 96 192 384 \
  --n_decoder_channels 48 \
  --resolution_factor 2 \
  --skip_aggregation addition \
  --ds False \
  --mode train \
  --pretrain False \
  --batch_size 1 \
  --crop_sample 4 \
  --lr 0.001 \
  --optim AdamW \
  --max_iter 45000 \
  --eval_step 250 \
  --val_batch 1 \
  --gpu 0 \
  --cache_rate 0.5 \
  --num_workers 2 \
  --overlap 0.7
```

Critical parameters (wrong values = wrong results):

| Parameter | Correct value | What goes wrong if incorrect |
|---|---|---|
| `resolution_factor` | **2** (not 1) | 1 = full resolution = not EffiDec3D |
| `n_decoder_channels` | **48** | Different channel count = different model |
| `skip_aggregation` | **addition** | concatenation doubles channels, changes FLOPs |
| `overlap` | **0.7** | Lower overlap = worse validation DICE |
| `ds` | **False** | True enables deep supervision, changes loss |

### Step 3.3: Multi-session checkpoint resumption

The script auto-saves `last_model.pth` and auto-resumes. Before each Kaggle session ends:

```python
# In notebook, copy checkpoint to Kaggle output for download
import shutil, glob

# Find the output directory (auto-named by the script)
output_dirs = glob.glob("/kaggle/working/output/E1_effidec3d/**/last_model.pth", recursive=True)
print("Checkpoint found at:", output_dirs)

# Copy to /kaggle/working/ for easy download
shutil.copy(output_dirs[0], "/kaggle/working/last_model_E1.pth")
```

Next session: upload `last_model_E1.pth` as a Kaggle Dataset input, then set `--output` to the same path and the script will find and resume from it automatically.

### Step 3.4: Monitor training

```python
# In a separate notebook cell
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/output/E1_effidec3d
```

Or check the script's stdout (redirect to file with `| tee log.txt`).

### Step 3.5: Verify reproduction success

At the start of training the script prints:

```
Computational complexity:   51.47 GMac   <- must match paper
Number of parameters:       3.16 M        <- must match paper
```

At validation the script prints per-step DICE. Final mean DICE should be:

```
BTCV13 mean DICE: 79.0% - 79.5%   (paper reports 79.25%)
```

If DICE is below 78%, check:
1. Is `resolution_factor=2` (not 1)?
2. Is `overlap=0.7` during inference?
3. Are imagesTr and labelsTr sorted in matching order?
4. Did training complete enough iterations (need ~30000+ for convergence)?

---

## Part 3.5: Observation Study — Validate Before Building

**Purpose**: Empirically confirm that adaptive decoder computation is worth pursuing before investing GPU time in AdaDec3D. This is the Go/No-Go gate. See `1_Research_Proposal.md §4` for scientific motivation and `2_Observation_Study.md` for deliverable formats.

**Prerequisites**: E0 (`best_metric_model.pth`) and E1 (`best_metric_model.pth`) must be trained first.

Run all analyses in a Kaggle notebook. Save figures to `/kaggle/working/obs/`.

```python
# Setup (run once at top of each notebook)
import torch, torch.nn.functional as F
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from load_datasets_transforms import data_loader, data_transforms
import argparse

def load_model(network_name, ckpt_path, device="cuda"):
    import argparse
    from monai_utils.inferers.utils import sliding_window_inference_1out
    if network_name == "3DUXNET_EffiDec3D":
        from networks.UXNet_3D.network_backbone import UXNET_EffiDec3D
        model = UXNET_EffiDec3D(
            in_chans=1, out_chans=14,
            depths=[2,2,2,2], feat_size=[48,96,192,384],
            n_decoder_channels=48, resolution_factor=2,
            skip_aggregation="addition"
        ).to(device)
    else:
        from networks.UXNet_3D.network_backbone import UXNET
        model = UXNET(in_chans=1, out_chans=14,
                      depths=[2,2,2,2], feat_size=[48,96,192,384]).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

BTCV_CLASS_NAMES = [
    "Aorta","Gallbladder","Spleen","L.Kidney","R.Kidney",
    "Liver","Stomach","IVC","Port.Vein",
    "Pancreas","R.Adrenal","L.Adrenal","Duodenum"
]

args = argparse.Namespace(root="/kaggle/input/btcv-synapse", dataset="BTCV13", mode="val")
_, val_files, n_cls = data_loader(args)
val_transform = data_transforms(args)
from monai.data import DataLoader, Dataset
val_ds = Dataset(data=val_files, transform=val_transform)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
```

### O1: Prediction Error Distribution

**Question**: Are segmentation errors uniformly distributed, or concentrated in specific regions?

```python
from monai_utils.inferers.utils import sliding_window_inference_1out

effi_model = load_model("3DUXNET_EffiDec3D", "/kaggle/input/e1-ckpt/best_metric_model.pth")
post_pred = AsDiscrete(argmax=True, to_onehot=14)
post_lbl  = AsDiscrete(to_onehot=14)

error_rates = []  # per case
organ_error = {name: [] for name in BTCV_CLASS_NAMES}
boundary_error_rates, interior_error_rates = [], []

with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].cuda()
        lbl = batch["label"].squeeze(1).long()  # [1, D, H, W]

        logits = sliding_window_inference_1out(img, (96,96,96), 4, effi_model, overlap=0.7)
        pred = logits.argmax(1).cpu()  # [1, D, H, W]

        error = (pred != lbl).float()  # [1, D, H, W]
        error_rates.append(error.mean().item())

        # Per-organ error rate
        for c, name in enumerate(BTCV_CLASS_NAMES, start=1):
            mask = (lbl == c)
            if mask.sum() > 0:
                organ_error[name].append((error[mask]).mean().item())

        # Boundary vs interior (erode label to get interior)
        from scipy.ndimage import binary_erosion
        lbl_np = (lbl > 0).squeeze().numpy()
        interior = torch.from_numpy(binary_erosion(lbl_np, iterations=3).astype(np.float32))
        boundary = torch.from_numpy(lbl_np.astype(np.float32)) - interior
        if boundary.sum() > 0:
            boundary_error_rates.append((error.squeeze() * boundary).sum() / boundary.sum())
        if interior.sum() > 0:
            interior_error_rates.append((error.squeeze() * interior).sum() / interior.sum())

print(f"Mean voxel error rate: {np.mean(error_rates):.3f}")
print(f"Boundary error rate:   {np.mean(boundary_error_rates):.3f}")
print(f"Interior error rate:   {np.mean(interior_error_rates):.3f}")
print("\nPer-organ error rates:")
for name, vals in organ_error.items():
    if vals: print(f"  {name:15s}: {np.mean(vals):.3f}")

# Expected: boundary >> interior; Pancreas/Adrenal >> Liver/Spleen
```

**Target result**: Boundary error rate should be 3-5× interior error rate. Small organs (Pancreas, Adrenal) should have the highest organ-wise error rates.

---

### O2: Uncertainty (Entropy) Distribution

**Question**: Are high-uncertainty voxels a small minority of the total volume?

```python
entropy_per_voxel = []  # store all entropy values (subsampled)
high_unc_fraction = []  # fraction of voxels with entropy > 0.5

with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].cuda()

        logits = sliding_window_inference_1out(img, (96,96,96), 4, effi_model, overlap=0.7)
        prob = logits.softmax(1).cpu()  # [1, C, D, H, W]
        entropy = -(prob * torch.log(prob + 1e-8)).sum(1)  # [1, D, H, W]

        flat = entropy.flatten().numpy()
        entropy_per_voxel.append(flat[::10])  # subsample 1/10 to save memory
        high_unc_fraction.append((entropy > 0.5).float().mean().item())

all_entropy = np.concatenate(entropy_per_voxel)
print(f"Entropy percentiles:")
for p in [50, 75, 90, 95, 99]:
    print(f"  p{p}: {np.percentile(all_entropy, p):.4f}")
print(f"Fraction with entropy > 0.5: {np.mean(high_unc_fraction):.2%}")

# Plot entropy histogram
plt.figure(figsize=(8,4))
plt.hist(all_entropy, bins=50, log=True)
plt.xlabel("Entropy"); plt.ylabel("Voxel count (log scale)")
plt.title("O2: Entropy Distribution")
plt.savefig("/kaggle/working/obs/O2_entropy_histogram.png", dpi=150)
plt.show()

# Expected: >90% of voxels have entropy < 0.1 (very low); tail extends to ~log(C)≈2.64
```

**Target result**: High-uncertainty voxels (entropy > 0.5) should comprise <10% of the total volume. If they're >30%, adaptive computation loses its efficiency advantage.

---

### O3: Uncertainty–Error Correlation (Key Metric)

**Question**: Does high entropy reliably predict where errors occur?

```python
from scipy.stats import pearsonr, spearmanr

case_entropy, case_error = [], []

with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].cuda()
        lbl = batch["label"].squeeze(1).long().cpu()

        logits = sliding_window_inference_1out(img, (96,96,96), 4, effi_model, overlap=0.7)
        prob = logits.softmax(1).cpu()
        pred = logits.argmax(1).cpu()
        entropy = -(prob * torch.log(prob + 1e-8)).sum(1).squeeze()

        error = (pred.squeeze() != lbl.squeeze()).float()

        # Bin entropy and compute mean error per bin
        n_bins = 20
        for b in range(n_bins):
            lo = b / n_bins * entropy.max().item()
            hi = (b + 1) / n_bins * entropy.max().item()
            mask = (entropy >= lo) & (entropy < hi)
            if mask.sum() > 100:
                case_entropy.append(entropy[mask].mean().item())
                case_error.append(error[mask].mean().item())

r_pearson, p_val = pearsonr(case_entropy, case_error)
r_spearman, _ = spearmanr(case_entropy, case_error)
print(f"Pearson r  = {r_pearson:.3f}  (p={p_val:.4f})")
print(f"Spearman ρ = {r_spearman:.3f}")

plt.figure(figsize=(6,5))
plt.scatter(case_entropy, case_error, alpha=0.6)
plt.xlabel("Mean Entropy"); plt.ylabel("Error Rate")
plt.title(f"O3: Entropy vs Error  (r={r_pearson:.2f})")
plt.savefig("/kaggle/working/obs/O3_entropy_error_scatter.png", dpi=150)

# GO criterion: Pearson r > 0.60
print(f"\n{'GO ✓' if r_pearson > 0.60 else 'NO-GO ✗'}: Pearson r = {r_pearson:.3f} (threshold: 0.60)")
```

---

### O4: Per-Organ Difficulty

**Question**: Which anatomical structures are inherently harder, and do they have higher entropy?

```python
organ_dice, organ_entropy = {n: [] for n in BTCV_CLASS_NAMES}, {n: [] for n in BTCV_CLASS_NAMES}
from monai.metrics import DiceMetric
dice_metric = DiceMetric(include_background=False, reduction="none")

with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].cuda()
        lbl = batch["label"].cpu()

        logits = sliding_window_inference_1out(img, (96,96,96), 4, effi_model, overlap=0.7)
        prob = logits.softmax(1).cpu()
        entropy = -(prob * torch.log(prob + 1e-8)).sum(1).squeeze()  # [D, H, W]

        pred_onehot = post_pred(logits.squeeze(0))
        lbl_onehot  = post_lbl(lbl.squeeze(0))
        dice_vals   = dice_metric(pred_onehot.unsqueeze(0), lbl_onehot.unsqueeze(0))[0]

        for c, name in enumerate(BTCV_CLASS_NAMES):
            organ_dice[name].append(dice_vals[c].item())
            mask = (lbl.squeeze() == c + 1)
            if mask.sum() > 0:
                organ_entropy[name].append(entropy[mask].mean().item())

print(f"{'Organ':15s} {'Dice':>6} {'Entropy':>8}")
print("-" * 32)
for name in BTCV_CLASS_NAMES:
    d = np.nanmean(organ_dice[name])
    e = np.nanmean(organ_entropy[name]) if organ_entropy[name] else float('nan')
    print(f"{name:15s} {d:6.3f}  {e:8.4f}")

# Expected: Pancreas + Adrenal = highest entropy + lowest DICE
```

---

### O5: Decoder Gain Analysis (Most Critical)

**Question**: Do difficult voxels (high entropy) benefit more from the full decoder (E0) than easy voxels?

This experiment directly validates the adaptive computation hypothesis.

```python
full_model  = load_model("3DUXNET",         "/kaggle/input/e0-ckpt/best_metric_model.pth")
effi_model  = load_model("3DUXNET_EffiDec3D", "/kaggle/input/e1-ckpt/best_metric_model.pth")

bin_entropy, bin_gain = [], []

with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].cuda()
        lbl = batch["label"].squeeze(1).long().cpu()

        # Full decoder prediction
        logits_full = sliding_window_inference_1out(img, (96,96,96), 4, full_model, overlap=0.7)
        pred_full = logits_full.argmax(1).cpu().squeeze()

        # EffiDec3D prediction + uncertainty
        logits_effi = sliding_window_inference_1out(img, (96,96,96), 4, effi_model, overlap=0.7)
        prob_effi = logits_effi.softmax(1).cpu()
        pred_effi = logits_effi.argmax(1).cpu().squeeze()
        entropy = -(prob_effi * torch.log(prob_effi + 1e-8)).sum(1).squeeze()

        lbl_sq = lbl.squeeze()
        # Gain: voxels where full decoder is correct but EffiDec3D is wrong
        gain = ((pred_full == lbl_sq) & (pred_effi != lbl_sq)).float()

        # Bin by entropy, compute mean gain per bin
        n_bins = 20
        for b in range(n_bins):
            lo = b / n_bins
            hi = (b + 1) / n_bins
            q_lo = entropy.quantile(lo).item()
            q_hi = entropy.quantile(hi).item()
            mask = (entropy >= q_lo) & (entropy < q_hi)
            if mask.sum() > 100:
                bin_entropy.append(entropy[mask].mean().item())
                bin_gain.append(gain[mask].mean().item())

# Sort by entropy for plotting
pairs = sorted(zip(bin_entropy, bin_gain))
x, y = zip(*pairs)

plt.figure(figsize=(7,5))
plt.plot(x, y, "o-", markersize=4)
plt.xlabel("Mean Entropy (EffiDec3D uncertainty)")
plt.ylabel("Decoder Gain (fraction of voxels fixed by full decoder)")
plt.title("O5: Decoder Gain vs Uncertainty")
plt.savefig("/kaggle/working/obs/O5_decoder_gain.png", dpi=150)
plt.show()

from scipy.stats import pearsonr
r, p = pearsonr(x, y)
print(f"Gain-Entropy Pearson r = {r:.3f}  (p={p:.4f})")
print(f"\n{'GO ✓' if r > 0.50 else 'NO-GO ✗'}: gain concentrates in high-entropy regions (threshold: r > 0.50)")
```

**Target result**: The gain curve should slope upward — high-entropy voxels get 5-10× more benefit from the full decoder than low-entropy voxels. If the curve is flat, adaptive computation won't help.

---

### Observation Study Go/No-Go

| Check | Criterion | Result | Decision |
|-------|-----------|--------|----------|
| O3 | Entropy–Error Pearson r > 0.60 | | |
| O5 | Gain–Entropy Pearson r > 0.50 | | |
| O2 | High-uncertainty voxels < 15% of volume | | |
| O4 | Small organs (Pancreas/Adrenal) rank top-3 in entropy | | |

**If all 4 pass** → proceed to Part 4 (AdaDec3D training).

**If O3 or O5 fail** → entropy may not be a reliable routing signal. Consider boundary probability or feature variance as alternatives before proceeding (see `1_Research_Proposal.md §7.4`).

---

## Part 4: AdaDec3D Experiments (Step by Step)

**Implemented files** (ready to use):
- [networks/adadec3d.py](EffiDec3D/networks/adadec3d.py) — AdaDec3D_UXNET model
- [main_train_adadec3d.py](EffiDec3D/main_train_adadec3d.py) — two-stage training script

### Step 4.1: Experiment groups

| ID | Name | `--use_moe` | `--use_roi` | Notes |
|---|---|---|---|---|
| E0 | Full 3DUXNET | — | — | Upper bound, `main_train_BTCV_TU.py --network 3DUXNET` |
| E1 | EffiDec3D | — | — | Baseline, `main_train_BTCV_TU.py --network 3DUXNET_EffiDec3D` |
| E2 | +MoE only | `True` | `False` | Ablation: ROI removed |
| E3 | +ROI only | `False` | `True` | Ablation: MoE removed |
| **E4** | **AdaDec3D** | `True` | `True` | Full method |

### Step 4.2: Stage 1 — train new modules, backbone frozen

Requires E1's `best_metric_model.pth` first. The script loads it, freezes
`uxnet_3d / encoder2-5 / decoder3-5 / coarse_out`, and trains only the
router, experts, and ROI refiner.

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

Backbone (`uxnet_3d`, encoders, coarse decoder) trains at `5e-4 × 0.1 = 5e-5`.
New modules train at `5e-4`.

### Step 4.4: Ablation runs (E2 and E3)

```bash
# E2: MoE only — disable ROI refinement
python main_train_adadec3d.py \
  --effidec3d_weights /path/to/E1/best_metric_model.pth \
  --stage 1 --use_moe True --use_roi False \
  --output /kaggle/working/output/E2 --max_iter 20000 [... same as above]

# Then stage 2 for E2
python main_train_adadec3d.py \
  --stage1_ckpt /path/to/E2/stage1/best_metric_model.pth \
  --stage 2 --use_moe True --use_roi False \
  --output /kaggle/working/output/E2 --max_iter 25000 [...]

# E3: ROI only — disable MoE (uses fixed Expert-M)
python main_train_adadec3d.py \
  --effidec3d_weights /path/to/E1/best_metric_model.pth \
  --stage 1 --use_moe False --use_roi True \
  --output /kaggle/working/output/E3 --max_iter 20000 [...]
```

### Step 4.5: Loss terms — what each one does

| Term | Weight | Purpose |
|---|---|---|
| `L_seg` | 1.0 | DiceCE on final prediction (interpolated to full resolution) |
| `L_coarse` | 0.5 | Auxiliary DiceCE on coarse decoder output — keeps backbone from degrading |
| `L_uncertainty` | 0.1 | Calibration: high-entropy voxels should correlate with actual errors |
| `L_resource` | 0.05 | Pushes router toward lighter experts when accuracy allows |
| `L_router` | 0.1 | Load balancing: prevents all samples collapsing to one expert |

All weights are adjustable via `--lambda_uncertainty`, `--lambda_resource`, `--lambda_router`, `--lambda_coarse`.

### Step 4.6: Monitor training loss terms

```python
# In Kaggle notebook
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/output/E4/stage1/BTCV13/tensorboard
```

Watch for:
- `Loss/router` should decrease and stabilise near 0 — experts are being used
- `Loss/unc` should decrease — uncertainty is calibrating
- If `Loss/router` stays high → increase `--lambda_router`
- If `Loss/seg` stops improving but `Loss/resource` is low → `lambda_resource` is too aggressive, reduce it

---

## Part 5: Metrics — What to Measure and Why

### 5.1 Standard segmentation metrics

```python
from monai.metrics import DiceMetric, HausdorffDistanceMetric

dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95)

# After each validation pass:
per_class_dice = dice_metric.aggregate()  # tensor [n_classes]
per_class_hd95 = hd95_metric.aggregate()  # tensor [n_classes]
mean_dice = per_class_dice.mean().item()

BTCV_CLASS_NAMES = [
    "Aorta", "Gallbladder", "Spleen", "L.Kidney", "R.Kidney",
    "Liver", "Stomach", "IVC", "Port.Vein",
    "Pancreas", "R.Adrenal", "L.Adrenal", "Duodenum"
]
# Print per-class
for name, d, h in zip(BTCV_CLASS_NAMES, per_class_dice, per_class_hd95):
    print(f"{name:15s}: DICE={d:.4f}  HD95={h:.2f}mm")
```

### 5.2 Efficiency metrics

```python
from ptflops import get_model_complexity_info
import time, torch

model.eval()
dummy = torch.randn(1, 1, 96, 96, 96).cuda()

# FLOPs and param count
macs, params = get_model_complexity_info(
    model, (1, 96, 96, 96),
    as_strings=True, print_per_layer_stat=False, verbose=False
)
print(f"GFLOPs: {macs}")
print(f"Params: {params}")

# Inference latency (average over 50 runs, after 10 warmup)
times = []
with torch.no_grad():
    for i in range(60):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(dummy)
        torch.cuda.synchronize()
        if i >= 10:  # skip warmup
            times.append(time.perf_counter() - t0)
print(f"Latency: {sum(times)/len(times)*1000:.1f} ms (mean over 50 runs)")

# Peak GPU memory
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    _ = model(dummy)
peak_mem = torch.cuda.max_memory_allocated() / 1024**2
print(f"Peak GPU Memory: {peak_mem:.0f} MB")
```

### 5.3 AdaDec3D-specific metrics (prove the adaptive mechanism works)

#### A: Expert activation distribution

```python
expert_hist = [0, 0, 0]  # counts for Expert-S, M, L
per_organ_expert = {name: [0,0,0] for name in BTCV_CLASS_NAMES}

model.eval()
with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].cuda()
        lbl = batch["label"].cuda()
        
        # Model returns router weights alongside prediction
        pred, router_w = model(img, return_router=True)
        expert_id = router_w.argmax(dim=1).item()  # 0, 1, or 2
        expert_hist[expert_id] += 1
        
        # Which organs are present in this sample?
        for c, name in enumerate(BTCV_CLASS_NAMES, start=1):
            if (lbl == c).any():
                per_organ_expert[name][expert_id] += 1

total = sum(expert_hist)
print(f"Expert-S (32ch): {expert_hist[0]/total:.1%}")
print(f"Expert-M (64ch): {expert_hist[1]/total:.1%}")
print(f"Expert-L (96ch): {expert_hist[2]/total:.1%}")
```

**What to look for**: Expert-L should be activated more for samples containing gallbladder, pancreas, adrenal glands. If all samples use Expert-S, load balancing loss needs to be increased.

#### B: ROI coverage of small organs

```python
SMALL_ORGAN_LABELS = {
    "Gallbladder": 2, "Pancreas": 10, "R.Adrenal": 11,
    "L.Adrenal": 12, "Duodenum": 13
}

coverage_stats = {name: [] for name in SMALL_ORGAN_LABELS}

with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].cuda()
        lbl = batch["label"]
        
        _, _, roi_mask = model(img, return_roi=True)
        # roi_mask: [B, 1, D/2, H/2, W/2] binary
        roi_full = F.interpolate(
            roi_mask.float(), size=lbl.shape[2:], mode="nearest"
        )  # upsample to full resolution
        
        for name, organ_id in SMALL_ORGAN_LABELS.items():
            organ_vox = (lbl == organ_id).float()
            if organ_vox.sum() == 0:
                continue  # organ not present in this scan
            covered = (organ_vox * roi_full.cpu()).sum()
            coverage_stats[name].append((covered / organ_vox.sum()).item())

for name, vals in coverage_stats.items():
    if vals:
        print(f"{name:15s}: ROI coverage = {sum(vals)/len(vals):.1%}")
```

**Target**: Coverage > 80% for all small organs. If coverage < 60%, decrease `roi_quantile` (expand ROI) or decrease the threshold.

#### C: Uncertainty-error correlation

```python
from scipy.stats import pearsonr
import numpy as np

unc_vals, err_vals = [], []

with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].cuda()
        lbl = batch["label"].cuda().squeeze(1).long()
        
        pred, extras = model(img, return_uncertainty=True)
        unc_map = extras["uncertainty"]  # [B, D/2, H/2, W/2]
        
        coarse_lbl = F.interpolate(
            lbl.unsqueeze(1).float(), size=unc_map.shape[1:], mode="nearest"
        ).squeeze(1).long()
        error = (pred.argmax(1) != F.interpolate(
            lbl.unsqueeze(1).float(), size=pred.shape[2:], mode="nearest"
        ).squeeze(1).long()).float()
        
        unc_vals.append(unc_map.mean().item())
        err_vals.append(error.mean().item())

r, p = pearsonr(unc_vals, err_vals)
print(f"Uncertainty-Error Pearson r = {r:.3f}, p-value = {p:.4f}")
```

**Target**: r > 0.60 (strong positive correlation). This proves the uncertainty estimate is meaningful, not random.

### 5.4 Final result table format

```
Table: BTCV 13-Organ Segmentation on Synapse Dataset

Method         Params  GFLOPs   Mean    Aorta  Gallb  Spleen LKid  RKid  Liver  Stom  IVC   PVein Pancr  RAdG  LAdG  Duod
Full 3DUXNET   53M     632.0    79.74   ...
EffiDec3D      3.16M   51.47    79.25   ...
AdaDec3D(E4)   X.XM    XX.X     XX.XX   ...
```

The columns that matter most for proving AdaDec3D's contribution:
- `Pancr` (Pancreas): EffiDec3D typically scores ~54%, AdaDec3D should recover toward 55-56%
- `RAdG`, `LAdG` (Adrenal glands): EffiDec3D ~63%, target 64-65%
- `Gallb` (Gallbladder): variable but important
- `Mean`: AdaDec3D >= EffiDec3D + 0.3% minimum to claim improvement

---

## Part 6: Ablation Studies

All ablation runs use `main_train_adadec3d.py`. Run Stage 1 (20000 iter) then Stage 2 (25000 iter) for each configuration.
Full common args: `--root /kaggle/input/btcv-synapse --dataset BTCV13 --cache_rate 0.5 --num_workers 2 --gpu 0`.

### 6.1 Module contribution ablation (E0-E4)

```bash
# E2: MoE only — ROI refinement disabled
# Stage 1
python main_train_adadec3d.py \
  --effidec3d_weights /path/to/E1/best_metric_model.pth \
  --output /kaggle/working/output/E2 \
  --stage 1 --use_moe True --use_roi False --max_iter 20000 --lr 5e-4

# Stage 2
python main_train_adadec3d.py \
  --stage1_ckpt /kaggle/input/e2-stage1/best_metric_model.pth \
  --output /kaggle/working/output/E2 \
  --stage 2 --use_moe True --use_roi False --max_iter 25000 --lr 5e-4

# E3: ROI only — MoE disabled (always uses Expert-M)
# Stage 1
python main_train_adadec3d.py \
  --effidec3d_weights /path/to/E1/best_metric_model.pth \
  --output /kaggle/working/output/E3 \
  --stage 1 --use_moe False --use_roi True --max_iter 20000 --lr 5e-4

# Stage 2
python main_train_adadec3d.py \
  --stage1_ckpt /kaggle/input/e3-stage1/best_metric_model.pth \
  --output /kaggle/working/output/E3 \
  --stage 2 --use_moe False --use_roi True --max_iter 25000 --lr 5e-4

# E4: Full AdaDec3D (see Part 4.2 and 4.3 above)
```

### 6.2 Hyperparameter ablation (run after E4 is stable)

```bash
# ROI coverage threshold — how much volume is marked as uncertain
python main_train_adadec3d.py --roi_quantile 0.25 ...  # refine top 75% (broad)
python main_train_adadec3d.py --roi_quantile 0.50 ...  # default: top 50%
python main_train_adadec3d.py --roi_quantile 0.75 ...  # refine top 25% (tight)

# Resource penalty weight — accuracy vs efficiency trade-off
python main_train_adadec3d.py --lambda_resource 0.01 ...  # accuracy priority
python main_train_adadec3d.py --lambda_resource 0.05 ...  # default
python main_train_adadec3d.py --lambda_resource 0.20 ...  # efficiency priority
```

Plot GFLOPs vs mean DICE for the three `lambda_resource` values → **efficiency-accuracy Pareto curve** (key figure in the paper).

### 6.3 Post-training analysis commands (eval only, no retraining)

```python
# In a Kaggle notebook cell — analysis of a trained E4 checkpoint
import torch, torch.nn.functional as F
from networks.adadec3d import AdaDec3D_UXNET

model = AdaDec3D_UXNET(out_chans=14).cuda()
ckpt = torch.load("/path/to/E4/stage2/best_metric_model.pth")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Expert routing distribution
with torch.no_grad():
    pred, extras = model(img, return_router=True)
    router_w = extras["router_weights"]   # [B, 3]

# ROI coverage of small organs
with torch.no_grad():
    pred, extras = model(img, return_roi=True)
    roi_mask = extras["roi_mask"]          # [B, 1, D/2, H/2, W/2]

# Uncertainty calibration
with torch.no_grad():
    pred, extras = model(img, return_uncertainty=True)
    unc_map = extras["uncertainty"]        # [B, D/2, H/2, W/2]

# All three at once
with torch.no_grad():
    pred, extras = model(img, return_router=True, return_roi=True, return_uncertainty=True)
    router_w = extras["router_weights"]
    roi_mask = extras["roi_mask"]
    unc_map  = extras["uncertainty"]
```

See Part 5.3 for the full analysis loops (expert histogram, ROI coverage %, uncertainty-error Pearson r).

---

## Part 7: Execution Timeline

```
Week 1: Setup + data
  [ ] Kaggle environment, install deps, verify GPU
  [ ] Download BTCV Synapse dataset
  [ ] Run data_loader sanity check (18 train, 12 val images found)
  [ ] Sanity run: 100 iterations, confirm no import errors or OOM

Week 2-3: Baseline reproduction
  [ ] E0: Full 3DUXNET (max_iter=20000, ~2 sessions)
  [ ] E1: EffiDec3D (max_iter=45000, ~3 sessions, checkpoint across sessions)
  [ ] Confirm: GFLOPs=51.47, Params=3.16M, mean DICE in 79%-79.5%
  [ ] Record per-class DICE for all 13 organs

Week 3-4: Observation study (Part 3.5) — Go/No-Go gate
  [ ] O1: Error map analysis — confirm boundary >> interior error rate
  [ ] O2: Entropy histogram — confirm high-unc voxels < 15% of volume
  [ ] O3: Entropy-error Pearson r > 0.60 → entropy is a valid routing signal
  [ ] O4: Per-organ table — confirm Pancreas/Adrenal have highest entropy
  [ ] O5: Decoder gain analysis — Gain-Entropy r > 0.50 → adaptive decoding is justified
  [ ] Save all figures to /kaggle/working/obs/ (these become paper Figure 2-4)
  [ ] --- GO / NO-GO DECISION HERE ---

Week 4-5: Implement AdaDec3D modules ✅ DONE
  [x] UncertaintyHead (entropy computation, no params)   → adadec3d.py:_uncertainty()
  [x] AdaptiveRouter (Linear(feat_dim+1, n_experts))     → adadec3d.py:AdaptiveRouter
  [x] ExpertDecoder x3 (S=32, M=64, L=96 channels)      → adadec3d.py:ExpertHead
  [x] ROIRefiner (residual conv × uncertainty mask)      → adadec3d.py:ROIRefineBlock
  [x] Training script with 4-term loss, 2-stage          → main_train_adadec3d.py
  [ ] Unit test: python -c "from networks.adadec3d import AdaDec3D_UXNET; ..."

Week 6-7: Stage 1 training
  [ ] E2: +MoE only (20000 iter, check expert distribution not collapsed)
  [ ] E3: +ROI only (20000 iter, check ROI coverage > 80% for small organs)
  [ ] If expert collapse: increase lambda_router from 0.1 to 0.3

Week 8-9: Stage 2 fine-tuning + ablations
  [ ] E4: Full AdaDec3D (25000 iter)
  [ ] Run ROI quantile ablation (0.25 / 0.50 / 0.75)
  [ ] Run lambda_resource ablation (Pareto curve)

Week 10: FeTA + visualization
  [ ] Prepare FeTA dataset, run E1 and E4 on FeTA
  [ ] Visualize: uncertainty map, ROI mask, expert activation per organ
  [ ] Compute Uncertainty-Error Pearson r

Week 11-12: Paper writing
  [ ] Table 1: E0-E4 quantitative comparison
  [ ] Table 2: Ablation studies (modules + hyperparams)
  [ ] Figure: Efficiency-accuracy Pareto curve
  [ ] Figure: Uncertainty map + ROI mask visualization
  [ ] MICCAI 2026 submission (typically January deadline)
```

---

## Part 8: Go / No-Go Criteria

### Go: submit to MICCAI (oral-level)
- AdaDec3D mean DICE >= EffiDec3D + 0.5%
- Small organs (Pancreas + Adrenal glands) mean DICE >= +1.5%
- GFLOPs <= EffiDec3D x 1.3
- Expert distribution: Expert-L is activated preferentially for hard samples

### Go: submit to MICCAI (poster) or JBHI
- AdaDec3D mean DICE >= EffiDec3D + 0.2%
- At least 2 small organ classes with >= +1% DICE improvement
- ROI coverage > 80% for small organs

### No-Go: needs debugging
- DICE below EffiDec3D -> check load balancing loss (expert collapse is common)
- All samples route to Expert-S -> increase lambda_router
- ROI coverage < 50% for small organs -> decrease roi_quantile
- Uncertainty-Error r < 0.3 -> uncertainty head not calibrating, check loss formulation
