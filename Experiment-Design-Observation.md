# Experiment Design — Observation Study

> **Scope**: Environment setup, dataset preparation, baseline reproduction (E0/E1),
> and the full observation study (O1–O11) that gates Paper A submission.
> For AdaDec3D training experiments see [Experiment-Design-AdaDec3D.md](Experiment-Design-AdaDec3D.md).
> For observation deliverables and Go/No-Go criteria see [Observation_Study.md](Observation_Study.md).
> For scientific motivation see [Research_Proposal.md §4](Research_Proposal.md).

---

## Part 0: Hardware — Kaggle P100

**Hardware**: Kaggle P100 (16 GB VRAM, CUDA). All code is CUDA-native (PyTorch + MONAI + ptflops); no changes needed.

### Timing estimates

```
EffiDec3D (3.16M params, 51.47 GFLOPs):
  ~1-2 sec per iter (4 patches of 96^3)
  20000 iter × 1.5s ≈ 8-10 hours for the matched pilot

Full 3DUXNET (53M params, 632 GFLOPs):
  ~5-8 sec per iter
  20000 iter × 6s ≈ 33 hours → 4 sessions
```

The pilot uses exactly 20,000 optimizer steps for both models. Comparing a
20,000-step full decoder against a 45,000-step EffiDec3D model would confound
decoder capacity with training exposure. A confirmatory run may extend both
models to 45,000 steps.

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
    feta-processed/        <- upload FeTA after conversion (O7)
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
    obs/                   <- observation study figures
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

## Part 2: Dataset Setup

### 2.1 BTCV / Synapse Dataset (primary)

#### Where to get the data

**Option A: Official Synapse (requires registration)**

1. Register at [synapse.org](https://www.synapse.org)
2. Go to Multi-Atlas Abdomen Labeling Challenge: `syn3193805`
3. Download `RawData.zip` (~1.5 GB)

**Option B: TransUNet preprocessed version (recommended)**

Search for "Synapse multi-organ segmentation dataset" on Kaggle — several public datasets are available with the correct preprocessing already applied.

#### Expected directory structure

```
/path/to/btcv/
  imagesTr/
    img0001.nii.gz  ...  (18 training cases)
  labelsTr/
    label0001.nii.gz ...
  imagesVal/
    img0021.nii.gz  ...  (12 validation cases)
  labelsVal/
    label0021.nii.gz ...
```

#### Standard BTCV train/val split

```python
TRAIN = ["0001","0002","0003","0004","0005","0006","0007","0008",
         "0009","0010","0021","0022","0023","0024","0025","0026",
         "0027","0028"]
VAL   = ["0029","0030","0031","0032","0033","0034","0035","0036",
         "0037","0038","0039","0040"]
```

This fixed split is suitable for an exploratory reproduction only. Do not tune
thresholds, select checkpoints, formulate the headline, and report final evidence
on the same 12 cases. For confirmatory results, use repeated/nested subject-level
cross-validation or reserve a held-out test fold before examining O1–O11. All
confidence intervals resample subjects, never individual voxels.

For the causal decoder test, add a shared-encoder condition: initialize both
models from the same encoder checkpoint, freeze the encoder, and train only the
full and compressed decoders with identical schedules. The ordinary end-to-end
comparison remains secondary because decoder changes can alter encoder features.

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
Label 10: Pancreas            <- key small structure
Label 11: Right Adrenal Gland <- key small structure
Label 12: Left Adrenal Gland  <- key small structure
Label 13: Duodenum
```

#### Verify data loading

```bash
python -c "
import argparse
from load_datasets_transforms import data_loader

args = argparse.Namespace(root='/kaggle/input/btcv-synapse', dataset='BTCV13', mode='train')
tr, val, nc = data_loader(args)
print('Train images:', len(tr['images']))   # expect: 18
print('Val images:', len(val['images']))     # expect: 12
print('Output classes:', nc)                # expect: 14
"
```

### 2.2 FeTA 2021 Dataset (for O7 cross-dataset)

#### Where to get the data

1. Register at [fetachallenge.github.io](https://fetachallenge.github.io)
2. Download `feta_2.2.tar.gz` (~2 GB, 80 fetal brain MRI subjects)

#### Convert to expected format

```python
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

## Part 3: Reproduce EffiDec3D Baseline

### Step 3.1: Train Full Baseline (E0, upper bound)

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
- `max_iter 20000` — sufficient for upper-bound reference, fits in ~2 Kaggle sessions
- `cache_rate 0.5` — do NOT use 1.0 on Kaggle (RAM ~13 GB, OOM risk)

### Step 3.2: Train EffiDec3D (E1, the baseline to beat)

```bash
python main_train_BTCV_TU.py \
  --root /kaggle/input/btcv-synapse \
  --output /kaggle/working/output/E1_effidec3d \
  --dataset BTCV13 \
  --img_size 96 96 96\
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
  --max_iter 20000 \
  --eval_step 500 \
  --val_batch 1 \
  --gpu 0 \
  --cache_rate 0.5 \
  --num_workers 2 \
  --overlap 0.7
```

Critical parameters:

| Parameter | Correct value | What goes wrong if incorrect |
|---|---|---|
| `resolution_factor` | **2** | 1 = full resolution = not EffiDec3D |
| `n_decoder_channels` | **48** | Different channel count = different model |
| `skip_aggregation` | **addition** | concatenation doubles channels, changes FLOPs |
| `overlap` | **0.7** | Lower overlap = worse validation DICE |

### Step 3.3: Multi-session checkpoint resumption

Before each Kaggle session ends:

```python
import shutil, glob
output_dirs = glob.glob("/kaggle/working/output/E1_effidec3d/**/last_model.pth", recursive=True)
shutil.copy(output_dirs[0], "/kaggle/working/last_model_E1.pth")
```

Next session: upload `last_model_E1.pth` as a Kaggle Dataset input; the script auto-resumes.

### Step 3.4: Verify reproduction success

At startup the script prints:

```
Computational complexity:   51.47 GMac   <- must match paper
Number of parameters:       3.16 M        <- must match paper
```

Final validation DICE target:

```
BTCV13 mean DICE: 79.0% - 79.5%   (paper reports 79.25%)
```

### Step 3.5: Training with checkpoint milestones (for O6)

Add `--save_milestones 5,10,20,30,50` to both E0 and E1 runs,
or manually save checkpoints during training by adding to the main loop:

```python
MILESTONES = [5, 10, 20, 30, 50]  # epochs
if epoch in MILESTONES:
    torch.save({"model_state_dict": model.state_dict()},
               f"{output_dir}/epoch_{epoch:03d}.pth")
```

These milestone checkpoints are required for O6 (Difficulty Evolution Analysis).

### Step 3.6: Train on FeTA (for O7)

O7 requires a matched full-decoder FeTA run as well as the EffiDec3D run below.
Use the same steps, seeds, optimizer, preprocessing, and checkpoint rule for both;
otherwise this tests only EffiDec3D uncertainty, not marginal decoder utility.

```bash
python main_train_BTCV_TU.py \
  --root /kaggle/input/feta-processed \
  --output /kaggle/working/output/E1_feta \
  --dataset FeTA \
  --network 3DUXNET_EffiDec3D \
  --n_decoder_channels 48 \
  --resolution_factor 2 \
  --skip_aggregation addition \
  --max_iter 45000 --eval_step 250 --lr 0.001 \
  --cache_rate 0.5 --num_workers 2 --gpu 0
```

### Step 3.7: Train SwinUNETR_EffiDec3D (for O8)

O8 likewise requires the corresponding full SwinUNETR decoder trained with the
same budget. A single compressed model cannot establish backbone consistency of
the full-versus-efficient decoder effect.

```bash
python main_train_BTCV_TU.py \
  --root /kaggle/input/btcv-synapse \
  --output /kaggle/working/output/E1_swin \
  --dataset BTCV13 \
  --network SwinUNETR_EffiDec3D \
  --n_decoder_channels 48 \
  --resolution_factor 2 \
  --skip_aggregation addition \
  --max_iter 45000 --eval_step 250 --lr 0.001 \
  --cache_rate 0.5 --num_workers 2 --gpu 0
```

---

## Part 3.5: Observation Study — Go/No-Go Gate for Paper A

**Purpose**: Empirically confirm that adaptive decoder computation is worth pursuing,
and generate the empirical evidence for Paper A.
**Prerequisites**: E0 and E1 `best_metric_model.pth` must be trained first.

Run all analyses as Kaggle notebooks. Save figures to `/kaggle/working/obs/`.

### Common setup (run once at top of each notebook)

```python
import torch, torch.nn.functional as F
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from monai.transforms import AsDiscrete
from monai_utils.inferers.utils import sliding_window_inference_1out
from load_datasets_transforms import data_loader, data_transforms
import argparse

def load_model(network_name, ckpt_path, device="cuda"):
    if network_name == "3DUXNET_EffiDec3D":
        from networks.UXNet_3D.network_backbone import UXNET_EffiDec3D
        model = UXNET_EffiDec3D(
            in_chans=1, out_chans=14,
            depths=[2,2,2,2], feat_size=[48,96,192,384],
            n_decoder_channels=48, resolution_factor=2,
            skip_aggregation="addition"
        ).to(device)
    elif network_name == "SwinUNETR_EffiDec3D":
        from networks.swin_unetr_effidec3d import SwinUNETR_EffiDec3D
        model = SwinUNETR_EffiDec3D(
            in_channels=1, out_channels=14,
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

args = argparse.Namespace(
    root="/kaggle/input/btcv-synapse", dataset="BTCV13",
    mode="validation", crop_sample=4, img_size=[96, 96, 96]
)
_, val_samples, n_cls = data_loader(args)
_, val_transform = data_transforms(args)
from monai.data import DataLoader, Dataset
val_files = [
    {"image": image, "label": label}
    for image, label in zip(val_samples["images"], val_samples["labels"])
]
val_ds = Dataset(data=val_files, transform=val_transform)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
```

---

### O1: Prediction Error Distribution

**Question**: Are segmentation errors uniformly distributed, or concentrated in specific regions?

```python
from monai_utils.inferers.utils import sliding_window_inference_1out

effi_model = load_model("3DUXNET_EffiDec3D", "/kaggle/input/e1-ckpt/best_metric_model.pth")
post_pred = AsDiscrete(argmax=True, to_onehot=14)
post_lbl  = AsDiscrete(to_onehot=14)

error_rates = []
organ_error = {name: [] for name in BTCV_CLASS_NAMES}
boundary_error_rates, interior_error_rates = [], []

with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].cuda()
        lbl = batch["label"].squeeze(1).long()

        logits = sliding_window_inference_1out(img, (96,96,96), 4, effi_model, overlap=0.7)
        pred = logits.argmax(1).cpu()
        error = (pred != lbl).float()
        error_rates.append(error.mean().item())

        for c, name in enumerate(BTCV_CLASS_NAMES, start=1):
            mask = (lbl == c)
            if mask.sum() > 0:
                organ_error[name].append((error[mask]).mean().item())

        from scipy.ndimage import binary_erosion
        lbl_np = (lbl > 0).squeeze().numpy()
        interior = torch.from_numpy(binary_erosion(lbl_np, iterations=3).astype(np.float32))
        boundary = torch.from_numpy(lbl_np.astype(np.float32)) - interior
        if boundary.sum() > 0:
            boundary_error_rates.append((error.squeeze() * boundary).sum() / boundary.sum())
        if interior.sum() > 0:
            interior_error_rates.append((error.squeeze() * interior).sum() / interior.sum())

print(f"Mean voxel error rate:  {np.mean(error_rates):.3f}")
print(f"Boundary error rate:    {np.mean(boundary_error_rates):.3f}")
print(f"Interior error rate:    {np.mean(interior_error_rates):.3f}")
for name, vals in organ_error.items():
    if vals: print(f"  {name:15s}: {np.mean(vals):.3f}")
```

**Target**: Boundary error rate 3-5× interior error rate.

---

### O2: Uncertainty (Entropy) Distribution

**Question**: Are high-uncertainty voxels a small minority of the total volume?

```python
entropy_per_voxel = []
high_unc_fraction = []

with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].cuda()
        logits = sliding_window_inference_1out(img, (96,96,96), 4, effi_model, overlap=0.7)
        prob = logits.softmax(1).cpu()
        entropy = -(prob * torch.log(prob + 1e-8)).sum(1)
        flat = entropy.flatten().numpy()
        entropy_per_voxel.append(flat[::10])
        high_unc_fraction.append((entropy > 0.5).float().mean().item())

all_entropy = np.concatenate(entropy_per_voxel)
for p in [50, 75, 90, 95, 99]:
    print(f"  p{p}: {np.percentile(all_entropy, p):.4f}")
print(f"Fraction with entropy > 0.5: {np.mean(high_unc_fraction):.2%}")
```

**Target**: High-uncertainty voxels (entropy > 0.5) should comprise < 15% of total volume.

---

### O3: Uncertainty–Error Correlation

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
print(f"\n{'GO ✓' if r_pearson > 0.60 else 'NO-GO ✗'}: threshold r > 0.60")
```

---

### O4: Per-Organ Difficulty

**Question**: Which anatomical structures are inherently harder?

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
        entropy = -(prob * torch.log(prob + 1e-8)).sum(1).squeeze()
        pred_onehot = post_pred(logits.squeeze(0))
        lbl_onehot  = post_lbl(lbl.squeeze(0))
        dice_vals   = dice_metric(pred_onehot.unsqueeze(0), lbl_onehot.unsqueeze(0))[0]
        for c, name in enumerate(BTCV_CLASS_NAMES):
            organ_dice[name].append(dice_vals[c].item())
            mask = (lbl.squeeze() == c + 1)
            if mask.sum() > 0:
                organ_entropy[name].append(entropy[mask].mean().item())

print(f"{'Organ':15s} {'Dice':>6} {'Entropy':>8}")
for name in BTCV_CLASS_NAMES:
    d = np.nanmean(organ_dice[name])
    e = np.nanmean(organ_entropy[name]) if organ_entropy[name] else float('nan')
    print(f"{name:15s} {d:6.3f}  {e:8.4f}")
```

---

### O5: Decoder Gain Analysis (Critical Go/No-Go)

**Question**: Do difficult voxels receive greater *net* benefit from the full decoder (E0)?

```python
full_model  = load_model("3DUXNET",           "/kaggle/input/e0-ckpt/best_metric_model.pth")
effi_model  = load_model("3DUXNET_EffiDec3D", "/kaggle/input/e1-ckpt/best_metric_model.pth")

bin_entropy, bin_positive, bin_negative, bin_net = [], [], [], []

with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].cuda()
        lbl = batch["label"].squeeze(1).long().cpu()

        logits_full = sliding_window_inference_1out(img, (96,96,96), 4, full_model, overlap=0.7)
        pred_full = logits_full.argmax(1).cpu().squeeze()

        logits_effi = sliding_window_inference_1out(img, (96,96,96), 4, effi_model, overlap=0.7)
        prob_effi = logits_effi.softmax(1).cpu()
        pred_effi = logits_effi.argmax(1).cpu().squeeze()
        entropy = -(prob_effi * torch.log(prob_effi + 1e-8)).sum(1).squeeze()

        lbl_sq = lbl.squeeze()
        positive = ((pred_full == lbl_sq) & (pred_effi != lbl_sq)).float()
        negative = ((pred_full != lbl_sq) & (pred_effi == lbl_sq)).float()
        net_gain = positive - negative

        n_bins = 20
        for b in range(n_bins):
            q_lo = entropy.quantile(b / n_bins).item()
            q_hi = entropy.quantile((b + 1) / n_bins).item()
            mask = (entropy >= q_lo) & (entropy < q_hi)
            if mask.sum() > 100:
                bin_entropy.append(entropy[mask].mean().item())
                bin_positive.append(positive[mask].mean().item())
                bin_negative.append(negative[mask].mean().item())
                bin_net.append(net_gain[mask].mean().item())

from scipy.stats import pearsonr
pairs = sorted(zip(bin_entropy, bin_net))
x, y = zip(*pairs)
r, p = pearsonr(x, y)
print(f"Net-gain/entropy Pearson r = {r:.3f} (descriptive; bins are correlated)")
print(f"Positive transition rate = {np.mean(bin_positive):.5f}")
print(f"Negative transition rate = {np.mean(bin_negative):.5f}")
print(f"Net transition rate      = {np.mean(bin_net):.5f}")
```

**Target**: Net gain slopes upward with entropy. Always report positive and
negative transitions separately; positive transitions alone overstate benefit.

---

### O9: Selective-computation opportunity analysis

**Question**: At fixed selection budgets, does entropy recover more positive
decoder transitions than a matched random selection?

Treat the curve below as an exploratory ranking diagnostic, not proof of FLOP
savings. Run it per subject and report the mean with a subject-bootstrap 95% CI.
For budgets of 5%, 10%, 20%, 30%, and 50%, compare entropy against 100 random
selections, a foreground mask, and a fixed-width organ-boundary mask. Keep the
oracle positive-transition ranking only as an unattainable upper bound. Do not
pool scans before inference: large volumes would dominate the result.

```python
import numpy as np

case_entropy, case_positive = [], []

with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].cuda()
        lbl = batch["label"].squeeze(1).long().cpu()

        logits_full = sliding_window_inference_1out(img, (96,96,96), 4, full_model, overlap=0.7)
        pred_full = logits_full.argmax(1).cpu().squeeze()

        logits_effi = sliding_window_inference_1out(img, (96,96,96), 4, effi_model, overlap=0.7)
        prob_effi = logits_effi.softmax(1).cpu()
        pred_effi = logits_effi.argmax(1).cpu().squeeze()
        entropy = -(prob_effi * torch.log(prob_effi + 1e-8)).sum(1).squeeze()

        lbl_sq = lbl.squeeze()
        positive = ((pred_full == lbl_sq) & (pred_effi != lbl_sq)).float()
        # Restrict the budget denominator to the union of foreground predictions
        # and labels. Otherwise easy distant background dominates every budget.
        body = (lbl_sq > 0) | (pred_full > 0) | (pred_effi > 0)
        case_entropy.append(entropy[body].numpy())
        case_positive.append(positive[body].numpy())

budgets = np.array([5, 10, 20, 30, 50])
rng = np.random.default_rng(0)
entropy_recovery, random_recovery = [], []
for entropy, positive in zip(case_entropy, case_positive):
    total = positive.sum()
    if total == 0:
        continue
    order = np.argsort(entropy)[::-1]
    entropy_recovery.append([
        positive[order[:max(1, int(len(order) * q / 100))]].sum() / total
        for q in budgets
    ])
    random_recovery.append(np.mean([
        [positive[rng.choice(len(positive), max(1, int(len(positive) * q / 100)),
                             replace=False)].sum() / total for q in budgets]
        for _ in range(100)
    ], axis=0))

entropy_recovery = np.asarray(entropy_recovery)
random_recovery = np.asarray(random_recovery)
mean_entropy = entropy_recovery.mean(0)
mean_random = random_recovery.mean(0)
print("Entropy recovery:", dict(zip(budgets, mean_entropy)))
print("Random recovery: ", dict(zip(budgets, mean_random)))

plt.figure(figsize=(7, 5))
plt.plot(budgets, mean_entropy * 100, marker="o", label="Entropy")
plt.plot(budgets, mean_random * 100, marker="o", label="Random (100 repeats)")
plt.xlabel("Selected union-foreground voxels (%)")
plt.ylabel("Positive decoder transitions recovered (%)")
plt.title("O9: Selective-computation opportunity")
plt.legend()
plt.savefig("/kaggle/working/obs/O9_pareto_curve.png", dpi=150)
plt.show()
```

---

### Additional observations O6–O8, O10–O11

These require additional checkpoints or datasets — refer to [Observation_Study.md](Observation_Study.md) for full code. Quick summary of setup requirements:

| Observation | Requires | When to run |
|---|---|---|
| O6 Difficulty Evolution | Milestone checkpoints from E1 training | After E1 finishes |
| O7 Cross-Dataset | E1 trained on FeTA (Step 3.6) | After E1_feta finishes |
| O8 Backbone Consistency | SwinUNETR_EffiDec3D trained on BTCV (Step 3.7) | After E1_swin finishes |
| O10 Organ Size vs Difficulty | No extra training, uses E1 BTCV | After O4 |
| O11 Routing Signal Comparison | No extra training, uses E1 BTCV | After O5 |

---

### Observation Study Go/No-Go

#### Minimum criteria for a confirmatory study

| Check | Criterion | Result | Decision |
|-------|-----------|--------|----------|
| O3 | Entropy–Error Pearson r > 0.60 | | |
| O5 | Net benefit differs across organs/regions with a subject-level 95% CI | | |
| O9 | Entropy beats matched random selection at 10–30% budgets | | |
| O2 | Uncertainty is concentrated; report observed fraction without a preset cutoff | | |

**If all 4 pass** → preregister/lock the confirmatory protocol, then run the
held-out evaluation. A particular “20% recovers 80%” result is an observation,
not a required threshold.

#### Additional criteria for Paper B (AdaDec3D)

| Check | Criterion | Result | Decision |
|-------|-----------|--------|----------|
| O7 | FeTA replication: Gain-Entropy r > 0.40 | | |
| O8 | SwinUNETR backbone: Gain-Entropy r > 0.45 | | |
| O11 | Entropy is best or tied-best routing signal | | |

**If all 3 pass** → proceed to [Experiment-Design-AdaDec3D.md](Experiment-Design-AdaDec3D.md).

---

## Part 7 (Phase 1): Execution Timeline

```
Week 1: Setup + data
  [ ] Kaggle environment, install deps, verify GPU
  [ ] Download BTCV Synapse dataset
  [ ] Run data_loader sanity check (18 train, 12 val images found)
  [ ] Sanity run: 100 iterations, confirm no OOM

Week 2-3: Baseline reproduction
  [ ] E0: Full 3DUXNET (max_iter=20000, ~2 sessions)
  [ ] E1: EffiDec3D on BTCV (max_iter=20000, matched to E0)
  [ ] Confirm: GFLOPs=51.47, Params=3.16M, mean DICE 79.0-79.5%

Week 4: Observation study O1-O5 (critical gate)
  [ ] O1: Error map — confirm boundary >> interior error rate
  [ ] O2: Entropy histogram — confirm high-unc voxels < 15% of volume
  [ ] O3: Entropy-error Pearson r > 0.60
  [ ] O4: Per-organ table — Pancreas/Adrenal rank top-3 in entropy
  [ ] O5: positive, negative, and net transitions by organ and boundary/interior
  [ ] O9: entropy vs random/boundary selection at matched budgets
  [ ] --- PAPER A GO / NO-GO DECISION ---

Week 5-6: Extended observations for Paper A
  [ ] O6: Difficulty evolution (using milestone checkpoints)
  [ ] O7: FeTA replication (requires E1_feta training)
  [ ] O8: SwinUNETR consistency (requires E1_swin training)
  [ ] O10: Organ size vs difficulty scatter
  [ ] O11: Routing signal comparison table

Week 7: Paper A draft
  [ ] Write observation study paper
  [ ] Target: MIDL / MLMI / ISBI (typically August-October submission)
```
