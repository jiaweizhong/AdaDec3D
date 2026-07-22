# Paper A: Observation Study — Complete Experiment Guide

> **Scientific motivation**: [Research_Proposal.md §3–5](Research_Proposal.md)
> **Architecture and Paper B**: [Experiment-Design-AdaDec3D.md](Experiment-Design-AdaDec3D.md)

---

## Two-Paper Strategy

| | Paper A — this document | Paper B |
|---|---|---|
| **Claim** | Decoder capacity benefit is spatially heterogeneous and predictable | AdaDec3D realizes selective allocation with lower executed cost |
| **Venue** | MIDL / MLMI / ISBI | MICCAI 2026 / TMI |
| **Gate** | O1–O5 + O9 pass Go criteria | Paper A accepted; O7, O8, O11 pass |
| **Key result** | Measure observed selection budget and recovered benefit vs controls | DICE ≥ EffiDec3D + 0.3% at matched executed MACs |

Paper A is self-contained. The selection budget and recovered benefit are **outcomes to measure**, not thresholds to hit in advance.

---

## Part 0: Hardware & Environment

### Hardware

AutoDL RTX 5090 (32 GB VRAM, Blackwell / SM_100). Requires CUDA ≥ 12.8 and PyTorch ≥ 2.6.

Both training scripts auto-detect GPU capability and use BF16 mixed precision on Blackwell/Ampere (`torch.autocast("cuda", dtype=torch.bfloat16)`), falling back to FP16 on older cards. BF16 gives ~1.5–2× wall-clock speedup over FP32 on the 5090 with no loss scaling required.

### Timing estimates (RTX 5090 + BF16)

```
E0 full 3DUXNET (53M params, 632 GFLOPs):
  ~0.4-0.7 s/iter → 20 000 iter ≈ 2-4 h

E1 EffiDec3D (3.16M params, 51.47 GFLOPs):
  ~0.1-0.2 s/iter → 20 000 iter ≈ 30-60 min
```

**Matched pilot**: run both models for exactly 20 000 optimizer steps to avoid
confounding decoder capacity with training exposure. A confirmatory run may
extend both to 45 000 steps.

The training script saves `last_model.pth` after every eval step and auto-resumes on restart. Run training inside `tmux` or `screen` so the session persists if the SSH connection drops.

### File layout

```
/root/
  AdaDec3D/           EffiDec3D/ networks/ ...   (code, cloned from repo)
  autodl-tmp/
    btcv-synapse/     imagesTr/ labelsTr/ imagesVal/ labelsVal/
    feta-processed/   imagesTr/ labelsTr/ imagesVal/ labelsVal/
  output/             training checkpoints
  obs/                observation study figures
```

`/root/autodl-tmp/` is the AutoDL persistent data disk (larger SSD, persists between instances). Put datasets there, not in `/root/` which is on the system disk.

### AutoDL instance setup

Select image: **PyTorch 2.6.0 / CUDA 12.8** (required for RTX 5090 Blackwell support).

```bash
# 1. Clone code
cd /root && git clone https://github.com/<your-repo>/AdaDec3D.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Upload datasets to /root/autodl-tmp/btcv-synapse/ and /root/autodl-tmp/feta-processed/
#    (use AutoDL file upload, scp, or wget)
```

### Verify environment

```bash
cd /root/AdaDec3D/EffiDec3D
python verify_env.py
```

---

## Part 1: Dataset Setup

### 1.0 Kaggle API Setup (one-time on AutoDL)

All datasets are sourced from Kaggle. Run once per AutoDL instance:

```bash
pip install kaggle
mkdir -p ~/.kaggle
# Kaggle → Account → API → Create New Token → paste the JSON below
cat > ~/.kaggle/kaggle.json << 'EOF'
{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}
EOF
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets list --search "synapse" | head -5   # verify credentials
```

---

### 1.1 BTCV / Synapse (primary — CT, 13 organs)

**Source**: TransUNet preprocessed version on Kaggle (search "Synapse multi-organ segmentation").
Official source: [synapse.org](https://www.synapse.org) project `syn3193805`.

**Download to AutoDL**

```bash
# Search on Kaggle to confirm the slug, then:
kaggle datasets download -d tiangexiang/synapse-multi-organ-segmentation \
    -p /root/autodl-tmp/ --unzip
# If the extracted folder name differs, rename it:
mv /root/autodl-tmp/Synapse /root/autodl-tmp/btcv-synapse 2>/dev/null || true
ls /root/autodl-tmp/btcv-synapse/imagesTr/ | head -3   # expect img0001.nii.gz …
```

**Expected layout**

```
/root/autodl-tmp/btcv-synapse/
  imagesTr/   img0001.nii.gz … (18 cases)
  labelsTr/   label0001.nii.gz …
  imagesVal/  img0021.nii.gz … (12 cases)
  labelsVal/  label0021.nii.gz …
```

**Standard split** (same as 3D UX-Net paper)

```python
TRAIN = ["0005","0006","0007","0009","0010","0021","0023","0024",
         "0026","0027","0028","0030","0031","0033","0034","0037","0039","0040"]
VAL   = ["0001","0002","0003","0004","0008","0022","0025","0029",
         "0032","0035","0036","0038"]
```

> **Confirmatory note**: do not tune thresholds, select checkpoints, formulate
> the Paper A headline, and report final evidence on the same 12 cases.
> For confirmatory results reserve a held-out fold or use nested cross-validation.
> All confidence intervals must resample subjects, not individual voxels.

**BTCV13 label mapping**

```
0 Background  1 Aorta      2 Gallbladder  3 Spleen      4 L.Kidney
5 R.Kidney    6 Liver      7 Stomach      8 IVC         9 Port.Vein
10 Pancreas*  11 R.Adrenal* 12 L.Adrenal* 13 Duodenum*
```
\* small structures — primary metric for Paper A/B.

**Verify loading**

```bash
python -c "import argparse; from load_datasets_transforms import data_loader; args = argparse.Namespace(root='/root/autodl-tmp/btcv-synapse', dataset='BTCV13', mode='train'); tr, val, nc = data_loader(args); print('Train:', len(tr['images']), 'Val:', len(val['images']), 'Classes:', nc)"
```

### 1.2 FeTA 2021 (for O7 — MRI, fetal brain, 7 structures)

**Source**: Check Kaggle first (search "FeTA 2021 fetal brain MRI"); if available download with:

```bash
kaggle datasets download -d <feta-dataset-slug> -p /root/autodl-tmp/ --unzip
```

Otherwise download directly: [fetachallenge.github.io](https://fetachallenge.github.io), `feta_2.2.tar.gz` (~2 GB, 80 subjects).

```bash
wget -O /root/autodl-tmp/feta_2.2.tar.gz \
    https://zenodo.org/record/xxxxxx/files/feta_2.2.tar.gz   # use link from site
tar -xzf /root/autodl-tmp/feta_2.2.tar.gz -C /root/autodl-tmp/
```

**Convert to expected format**

```python
import glob, shutil, os

src = "/root/autodl-tmp/feta_2.2"
dst = "/root/autodl-tmp/feta-processed"
subjects = sorted(glob.glob(f"{src}/sub-*/"))

for split, subs in [("Tr", subjects[:70]), ("Val", subjects[70:])]:
    os.makedirs(f"{dst}/images{split}", exist_ok=True)
    os.makedirs(f"{dst}/labels{split}", exist_ok=True)
    for sub in subs:
        sid = os.path.basename(sub.rstrip("/"))
        shutil.copy(f"{sub}/anat/{sid}_T2w.nii.gz", f"{dst}/images{split}/{sid}.nii.gz")
        shutil.copy(f"{sub}/anat/{sid}_dseg.nii.gz", f"{dst}/labels{split}/{sid}.nii.gz")
print("Train:", len(os.listdir(f"{dst}/imagesTr")))   # 70
print("Val:  ", len(os.listdir(f"{dst}/imagesVal")))  # 10
```

**FeTA label mapping**: 0 BG | 1 IS | 2 WM | 3 CGM | 4 DGM\* | 5 CE | 6 BS | 7 CSF

---

## Part 2: Baseline Training (E0 + E1)

Both models use identical optimizers, augmentations, crop sizes, and iteration
counts so that decoder capacity is the only variable. This is the primary causal
comparison for O5.

### E0 — Full 3DUXNET (upper bound)

```bash
cd /root/AdaDec3D/EffiDec3D

python main_train_BTCV_TU.py \
  --root /root/autodl-tmp/btcv-synapse --output /root/output/E0 \
  --dataset BTCV13 --network 3DUXNET \
  --img_size 96 96 96 --n_channels 1 \
  --channels 48 96 192 384 --feature_size 48 \
  --ds False --mode train --pretrain False \
  --batch_size 1 --crop_sample 4 \
  --lr 0.001 --optim AdamW \
  --max_iter 20000 --eval_step 500 \
  --val_batch 1 --gpu 0 \
  --cache_rate 1.0 --num_workers 8 --overlap 0.7
```

### E1 — EffiDec3D (baseline to beat)

```bash
python main_train_BTCV_TU.py \
  --root /root/autodl-tmp/btcv-synapse --output /root/output/E1 \
  --dataset BTCV13 --network 3DUXNET_EffiDec3D \
  --img_size 96 96 96 --n_channels 1 \
  --channels 48 96 192 384 \
  --n_decoder_channels 48 --resolution_factor 2 --skip_aggregation addition \
  --ds False --mode train --pretrain False \
  --batch_size 1 --crop_sample 4 \
  --lr 0.001 --optim AdamW \
  --max_iter 20000 --eval_step 500 \
  --val_batch 1 --gpu 0 \
  --cache_rate 1.0 --num_workers 8 --overlap 0.7
```

**Critical parameters for E1**

| Parameter | Correct | Wrong value effect |
|---|---|---|
| `resolution_factor` | **2** | 1 = full resolution = not EffiDec3D |
| `n_decoder_channels` | **48** | Different channel count |
| `skip_aggregation` | **addition** | concatenation doubles channels |
| `overlap` | **0.7** | Lower → worse DICE |

**Verify E1 prints at startup**:

```
Computational complexity:   51.47 GMac
Number of parameters:       3.16 M
```

**Target BTCV13 mean DICE**: 79.0–79.5% (paper: 79.25%)

### Checkpoint resumption

The script saves `last_model.pth` after every eval step and auto-resumes on restart. Run inside `tmux` to survive SSH disconnection:

```bash
tmux new -s e1_train
# ... run training command ...
# Ctrl-B D to detach; tmux attach -t e1_train to re-attach
```

### Milestone checkpoints (required for O6)

Add to the training loop after each epoch evaluation:

```python
MILESTONES = [5, 10, 20, 30, 50]   # epochs
if epoch in MILESTONES:
    torch.save({"model_state_dict": model.state_dict()},
               f"{output_dir}/epoch_{epoch:03d}.pth")
```

### E1 on FeTA (required for O7)

O7 needs both an E0-FeTA and an E1-FeTA run with identical schedules.

```bash
# E0 FeTA
python main_train_BTCV_TU.py \
  --root /root/autodl-tmp/feta-processed --output /root/output/E0_feta \
  --dataset FeTA --network 3DUXNET \
  --max_iter 20000 --eval_step 500 --lr 0.001 \
  --cache_rate 1.0 --num_workers 8 --gpu 0

# E1 FeTA
python main_train_BTCV_TU.py \
  --root /root/autodl-tmp/feta-processed --output /root/output/E1_feta \
  --dataset FeTA --network 3DUXNET_EffiDec3D \
  --n_decoder_channels 48 --resolution_factor 2 --skip_aggregation addition \
  --max_iter 20000 --eval_step 500 --lr 0.001 \
  --cache_rate 1.0 --num_workers 8 --gpu 0
```

### SwinUNETR_EffiDec3D on BTCV (required for O8)

O8 needs both E0-Swin and E1-Swin with identical schedules.

```bash
# E0 SwinUNETR
python main_train_BTCV_TU.py \
  --root /root/autodl-tmp/btcv-synapse --output /root/output/E0_swin \
  --dataset BTCV13 --network SwinUNETR \
  --max_iter 20000 --eval_step 500 --lr 0.001 \
  --cache_rate 1.0 --num_workers 8 --gpu 0

# E1 SwinUNETR_EffiDec3D
python main_train_BTCV_TU.py \
  --root /root/autodl-tmp/btcv-synapse --output /root/output/E1_swin \
  --dataset BTCV13 --network SwinUNETR_EffiDec3D \
  --n_decoder_channels 48 --resolution_factor 2 --skip_aggregation addition \
  --max_iter 20000 --eval_step 500 --lr 0.001 \
  --cache_rate 1.0 --num_workers 8 --gpu 0
```

---

## Part 3: Observation Study

**Prerequisites**: E0 and E1 `best_metric_model.pth` trained and verified.
Save all figures to `/root/obs/`.

### Common notebook setup

```python
import torch, torch.nn.functional as F
import numpy as np, matplotlib.pyplot as plt
import json, os
from monai.transforms import AsDiscrete
from monai_utils.inferers.utils import sliding_window_inference_1out
from load_datasets_transforms import data_loader, data_transforms
import argparse

os.makedirs("/root/obs", exist_ok=True)
RESULTS_FILE = "/root/obs/results.json"

def save_obs(tag, metrics):
    """Append/update one observation's metrics in the shared results JSON."""
    data = {}
    if os.path.exists(RESULTS_FILE):
        try:
            data = json.load(open(RESULTS_FILE))
        except json.JSONDecodeError:
            pass
    data[tag] = metrics
    json.dump(data, open(RESULTS_FILE, "w"), indent=2)
    print(f"[{tag}] metrics saved → {RESULTS_FILE}")

def load_model(network_name, ckpt_path, device="cuda"):
    if network_name == "3DUXNET_EffiDec3D":
        from networks.UXNet_3D.network_backbone import UXNET_EffiDec3D
        model = UXNET_EffiDec3D(in_chans=1, out_chans=14, depths=[2,2,2,2],
            feat_size=[48,96,192,384], n_decoder_channels=48, resolution_factor=2,
            skip_aggregation="addition").to(device)
    elif network_name == "SwinUNETR_EffiDec3D":
        from networks.swin_unetr_effidec3d import SwinUNETR_EffiDec3D
        model = SwinUNETR_EffiDec3D(in_channels=1, out_channels=14,
            n_decoder_channels=48, resolution_factor=2,
            skip_aggregation="addition").to(device)
    elif network_name == "SwinUNETR":
        from networks.swin_unetr import SwinUNETR
        model = SwinUNETR(in_channels=1, out_channels=14).to(device)
    else:
        from networks.UXNet_3D.network_backbone import UXNET
        model = UXNET(in_chans=1, out_chans=14, depths=[2,2,2,2],
                      feat_size=[48,96,192,384]).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

BTCV_NAMES = ["Aorta","Gallbladder","Spleen","L.Kidney","R.Kidney",
              "Liver","Stomach","IVC","Port.Vein","Pancreas","R.Adrenal","L.Adrenal","Duodenum"]

args = argparse.Namespace(
    root="/root/autodl-tmp/btcv-synapse", dataset="BTCV13",
    mode="validation", crop_sample=4, img_size=[96, 96, 96]
)
_, val_samples, n_cls = data_loader(args)
_, val_transform = data_transforms(args)
from monai.data import DataLoader, Dataset
val_files = [{"image": im, "label": lb}
             for im, lb in zip(val_samples["images"], val_samples["labels"])]
val_ds = Dataset(data=val_files, transform=val_transform)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

effi_model = load_model("3DUXNET_EffiDec3D", "/root/output/E1/.../best_metric_model.pth")
full_model  = load_model("3DUXNET",           "/root/output/E0/.../best_metric_model.pth")
post_pred   = AsDiscrete(argmax=True, to_onehot=14)
post_lbl    = AsDiscrete(to_onehot=14)
```

---

### O1 — Prediction Error Distribution

**Question**: Are errors uniformly distributed or concentrated in specific regions?

```python
from scipy.ndimage import binary_erosion

boundary_err, interior_err = [], []
organ_err = {n: [] for n in BTCV_NAMES}

with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].cuda()
        lbl = batch["label"].squeeze(1).long().cpu()
        logits = sliding_window_inference_1out(img, (96,96,96), 4, effi_model, overlap=0.7)
        pred = logits.argmax(1).cpu()
        error = (pred != lbl).float().squeeze()

        for c, name in enumerate(BTCV_NAMES, start=1):
            mask = (lbl.squeeze() == c)
            if mask.sum() > 0:
                organ_err[name].append(error[mask].mean().item())

        fg = (lbl.squeeze() > 0).numpy()
        interior = torch.from_numpy(binary_erosion(fg, iterations=3).astype(np.float32))
        boundary = torch.from_numpy(fg.astype(np.float32)) - interior
        if boundary.sum() > 0: boundary_err.append((error * boundary).sum() / boundary.sum())
        if interior.sum() > 0: interior_err.append((error * interior).sum() / interior.sum())

b_err = float(np.mean(boundary_err))
i_err = float(np.mean(interior_err))
print(f"Boundary error: {b_err:.3f}")
print(f"Interior error: {i_err:.3f}")
print(f"Ratio: {b_err/i_err:.1f}×")
for name, vals in organ_err.items():
    if vals: print(f"  {name:15s}: {np.mean(vals):.3f}")

# Figure: per-organ error bar chart
names_o1 = [n for n, v in organ_err.items() if v]
vals_o1  = [np.mean(organ_err[n]) for n in names_o1]
plt.figure(figsize=(11, 4))
plt.bar(range(len(names_o1)), vals_o1)
plt.xticks(range(len(names_o1)), names_o1, rotation=45, ha="right")
plt.ylabel("Mean pixel error rate"); plt.title("O1: Per-Organ Error Rate")
plt.tight_layout()
plt.savefig("/root/obs/O1_organ_error.png", dpi=150)
plt.show()

save_obs("O1", {
    "boundary_error": b_err,
    "interior_error": i_err,
    "boundary_interior_ratio": round(b_err / i_err, 2),
    "organ_error": {n: round(float(np.mean(v)), 4) for n, v in organ_err.items() if v},
})
```

**Expected**: boundary error 3–5× interior error; Pancreas/Adrenal highest organ error.

---

### O2 — Entropy Distribution

**Question**: Where is high uncertainty located, and what fraction of voxels does it occupy?

```python
all_entropy, high_unc_frac = [], []

with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].cuda()
        logits = sliding_window_inference_1out(img, (96,96,96), 4, effi_model, overlap=0.7)
        prob = logits.softmax(1).cpu()
        ent = -(prob * torch.log(prob + 1e-8)).sum(1).squeeze()
        all_entropy.append(ent.flatten().numpy()[::10])   # 10% subsample
        high_unc_frac.append((ent > 0.5).float().mean().item())

all_ent = np.concatenate(all_entropy)
pcts = {p: float(np.percentile(all_ent, p)) for p in [50, 75, 90, 95, 99]}
frac_high = float(np.mean(high_unc_frac))
print("Entropy percentiles:", {p: f"{v:.4f}" for p, v in pcts.items()})
print(f"Fraction entropy > 0.5: {frac_high:.2%}")

plt.figure(figsize=(8,4))
plt.hist(all_ent, bins=50, log=True)
plt.xlabel("Entropy"); plt.ylabel("Voxel count (log)")
plt.title("O2: Entropy Distribution")
plt.savefig("/root/obs/O2_entropy.png", dpi=150)
plt.show()

save_obs("O2", {"percentiles": pcts, "fraction_above_0.5": frac_high})
```

**Report**: the observed distribution and fraction. Do not apply a hard threshold.

---

### O3 — Uncertainty–Error Correlation

**Question**: Does high entropy reliably predict where errors occur?

```python
from scipy.stats import pearsonr, spearmanr

x_ent, y_err = [], []
with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].cuda()
        lbl = batch["label"].squeeze(1).long().cpu()
        logits = sliding_window_inference_1out(img, (96,96,96), 4, effi_model, overlap=0.7)
        prob = logits.softmax(1).cpu()
        ent = -(prob * torch.log(prob + 1e-8)).sum(1).squeeze()
        err = (logits.argmax(1).cpu().squeeze() != lbl.squeeze()).float()
        for b in range(20):
            lo, hi = b/20 * ent.max().item(), (b+1)/20 * ent.max().item()
            mask = (ent >= lo) & (ent < hi)
            if mask.sum() > 100:
                x_ent.append(ent[mask].mean().item())
                y_err.append(err[mask].mean().item())

r_p, _ = pearsonr(x_ent, y_err)
r_s, _ = spearmanr(x_ent, y_err)
print(f"Pearson r={r_p:.3f}  Spearman ρ={r_s:.3f}")
print(f"{'GO ✓' if r_p > 0.60 else 'NO-GO ✗'}  (threshold r > 0.60)")

# Figure: entropy bin vs error rate scatter
plt.figure(figsize=(6, 5))
plt.scatter(x_ent, y_err, alpha=0.7)
plt.xlabel("Mean entropy (bin)"); plt.ylabel("Error rate (bin)")
plt.title(f"O3: Uncertainty–Error  r={r_p:.3f}")
plt.tight_layout()
plt.savefig("/root/obs/O3_unc_error_scatter.png", dpi=150)
plt.show()

save_obs("O3", {"pearson_r": float(r_p), "spearman_rho": float(r_s),
                "go": r_p > 0.60})
```

---

### O4 — Per-Organ Difficulty

**Question**: Which anatomical structures are inherently harder, and do they show higher entropy?

```python
from monai.metrics import DiceMetric

organ_dice = {n: [] for n in BTCV_NAMES}
organ_ent  = {n: [] for n in BTCV_NAMES}
dice_metric = DiceMetric(include_background=False, reduction="none")

with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].cuda()
        lbl = batch["label"].cpu()
        logits = sliding_window_inference_1out(img, (96,96,96), 4, effi_model, overlap=0.7)
        prob = logits.softmax(1).cpu()
        ent = -(prob * torch.log(prob + 1e-8)).sum(1).squeeze()
        dice_vals = dice_metric(post_pred(logits.squeeze(0)).unsqueeze(0),
                                post_lbl(lbl.squeeze(0)).unsqueeze(0))[0]
        for c, name in enumerate(BTCV_NAMES):
            organ_dice[name].append(dice_vals[c].item())
            mask = (lbl.squeeze() == c + 1)
            if mask.sum() > 0:
                organ_ent[name].append(ent[mask].mean().item())

print(f"{'Organ':15s} {'DICE':>6} {'Entropy':>8}")
dice_summary, ent_summary = {}, {}
for name in BTCV_NAMES:
    d = float(np.nanmean(organ_dice[name]))
    e = float(np.nanmean(organ_ent[name])) if organ_ent[name] else float('nan')
    dice_summary[name] = round(d, 4)
    ent_summary[name]  = round(e, 4)
    print(f"{name:15s} {d:6.3f}  {e:8.4f}")

# Figure: dual bar chart
fig, ax1 = plt.subplots(figsize=(12, 5))
x = np.arange(len(BTCV_NAMES))
ax1.bar(x - 0.2, [dice_summary[n] for n in BTCV_NAMES], 0.4, label="DICE", color="steelblue")
ax1.set_ylabel("DICE"); ax1.set_ylim(0, 1)
ax2 = ax1.twinx()
ax2.bar(x + 0.2, [ent_summary[n] for n in BTCV_NAMES], 0.4, label="Entropy", color="orange", alpha=0.8)
ax2.set_ylabel("Mean Entropy")
ax1.set_xticks(x); ax1.set_xticklabels(BTCV_NAMES, rotation=45, ha="right")
ax1.set_title("O4: Per-Organ DICE vs Entropy")
fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
plt.tight_layout()
plt.savefig("/root/obs/O4_organ_dice_entropy.png", dpi=150)
plt.show()

save_obs("O4", {"dice": dice_summary, "entropy": ent_summary})
```

---

### O5 — Decoder Gain Analysis *(critical Go/No-Go gate)*

**Question**: Does a stronger decoder produce net benefit primarily in high-entropy voxels?

Report **positive** and **negative** transitions separately — positive alone overstates benefit.

```python
bin_ent, bin_pos, bin_neg, bin_net = [], [], [], []

with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].cuda()
        lbl = batch["label"].squeeze(1).long().cpu().squeeze()

        pred_full = sliding_window_inference_1out(img, (96,96,96), 4, full_model,
                                                   overlap=0.7).argmax(1).cpu().squeeze()
        logits_e  = sliding_window_inference_1out(img, (96,96,96), 4, effi_model, overlap=0.7)
        prob_e    = logits_e.softmax(1).cpu()
        pred_effi = logits_e.argmax(1).cpu().squeeze()
        ent = -(prob_e * torch.log(prob_e + 1e-8)).sum(1).squeeze()

        pos = ((pred_full == lbl) & (pred_effi != lbl)).float()
        neg = ((pred_full != lbl) & (pred_effi == lbl)).float()
        net = pos - neg

        for b in range(20):
            q_lo = ent.quantile(b/20).item()
            q_hi = ent.quantile((b+1)/20).item()
            mask = (ent >= q_lo) & (ent < q_hi)
            if mask.sum() > 100:
                bin_ent.append(ent[mask].mean().item())
                bin_pos.append(pos[mask].mean().item())
                bin_neg.append(neg[mask].mean().item())
                bin_net.append(net[mask].mean().item())

from scipy.stats import pearsonr
pairs = sorted(zip(bin_ent, bin_net))
x, y = zip(*pairs)
r, p = pearsonr(x, y)
print(f"Net-gain/entropy Pearson r={r:.3f} (descriptive; bins are correlated)")
mean_pos = float(np.mean(bin_pos))
mean_neg = float(np.mean(bin_neg))
print(f"Mean positive rate={mean_pos:.5f}  negative rate={mean_neg:.5f}")

# Figure: net gain curve with positive/negative lines
plt.figure(figsize=(8, 5))
plt.plot(list(x), [p_ for p_ in bin_pos], "g--o", markersize=4, label="Positive rate")
plt.plot(list(x), [n_ for n_ in bin_neg], "r--o", markersize=4, label="Negative rate")
plt.plot(list(x), list(y), "b-o", markersize=5, label=f"Net gain (r={r:.2f})")
plt.axhline(0, color="k", linewidth=0.8, linestyle=":")
plt.xlabel("Mean entropy (bin)"); plt.ylabel("Rate")
plt.title("O5: Decoder Gain vs Uncertainty")
plt.legend(); plt.tight_layout()
plt.savefig("/root/obs/O5_decoder_gain.png", dpi=150)
plt.show()

save_obs("O5", {
    "net_gain_entropy_pearson_r": float(r),
    "mean_positive_rate": mean_pos,
    "mean_negative_rate": mean_neg,
    "bin_ent": [float(v) for v in x],
    "bin_net": [float(v) for v in y],
    "go": r > 0.0 and mean_pos > mean_neg,
})
```

**Go criterion**: net benefit rises with entropy AND a deployable signal beats
matched random at 10–30% budgets with a subject-level 95% CI (see O9).

---

### O6 — Difficulty Evolution During Training

**Question**: Does high entropy persist and stabilize at boundaries as training progresses?

*Requires milestone checkpoints saved in Part 2.*

```python
MILESTONES = [5, 10, 20, 30, 50]

epoch_ent = {}
for epoch in MILESTONES:
    m = load_model("3DUXNET_EffiDec3D",
                   f"/root/output/E1/.../epoch_{epoch:03d}.pth")
    mean_ents = []
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].cuda()
            logits = sliding_window_inference_1out(img, (96,96,96), 4, m, overlap=0.7)
            prob = logits.softmax(1).cpu()
            ent = -(prob * torch.log(prob + 1e-8)).sum(1)
            mean_ents.append(ent.mean().item())
    epoch_ent[epoch] = float(np.mean(mean_ents))
    print(f"Epoch {epoch:3d}  mean_entropy={epoch_ent[epoch]:.4f}")

# Figure: entropy vs training epoch
plt.figure(figsize=(7, 4))
plt.plot(list(epoch_ent.keys()), list(epoch_ent.values()), "o-")
plt.xlabel("Epoch"); plt.ylabel("Mean entropy")
plt.title("O6: Entropy Evolution During Training")
plt.tight_layout()
plt.savefig("/root/obs/O6_entropy_evolution.png", dpi=150)
plt.show()

save_obs("O6", {"epoch_mean_entropy": epoch_ent})
```

**Figure**: mean entropy vs epoch (line) + spatial entropy maps at epochs 5, 20, 50.
**Expected**: entropy decreases but stabilizes; residual high-entropy voxels concentrate at boundaries/small organs by epoch 30.

---

### O7 — Cross-Dataset Consistency

**Question**: Do O1–O5 findings replicate on FeTA (fetal brain MRI)?

*Requires E0_feta and E1_feta from Part 2.*

```python
# Repeat O5 analysis with FeTA models and val_loader
FETA_NAMES = ["IS","WM","CGM","DGM","CE","BS","CSF"]

feta_args = argparse.Namespace(
    root="/root/autodl-tmp/feta-processed", dataset="FeTA",
    mode="validation", crop_sample=4, img_size=[96,96,96]
)
_, feta_val, n_cls_feta = data_loader(feta_args)
_, feta_transform = data_transforms(feta_args)
feta_files = [{"image": im, "label": lb}
              for im, lb in zip(feta_val["images"], feta_val["labels"])]
feta_loader = DataLoader(Dataset(data=feta_files, transform=feta_transform),
                         batch_size=1, shuffle=False, num_workers=2)

full_feta  = load_model("3DUXNET",           "/root/output/E0_feta/.../best_metric_model.pth")
effi_feta  = load_model("3DUXNET_EffiDec3D", "/root/output/E1_feta/.../best_metric_model.pth")

# Run identical O5 analysis using feta_loader, full_feta, effi_feta, n_cls=8
# ... (same code as O5 above, replace val_loader / full_model / effi_model)
print(f"FeTA Net-gain/entropy r={r_feta:.3f}")
print(f"{'GO ✓' if r_feta > 0.40 else 'NO-GO ✗'}  (threshold r > 0.40)")

save_obs("O7", {"feta_gain_entropy_pearson_r": float(r_feta), "go": r_feta > 0.40})
```

---

### O8 — Backbone Consistency

**Question**: Does the O5 gain–entropy correlation hold with SwinUNETR instead of UXNET?

*Requires E0_swin and E1_swin from Part 2.*

```python
full_swin = load_model("SwinUNETR",          "/root/output/E0_swin/.../best_metric_model.pth")
effi_swin = load_model("SwinUNETR_EffiDec3D","/root/output/E1_swin/.../best_metric_model.pth")

# Run identical O5 analysis using val_loader, full_swin, effi_swin
# ... (same code as O5 above, replace full_model / effi_model)
print(f"SwinUNETR Net-gain/entropy r={r_swin:.3f}")
print(f"{'GO ✓' if r_swin > 0.45 else 'NO-GO ✗'}  (threshold r > 0.45)")

save_obs("O8", {"swin_gain_entropy_pearson_r": float(r_swin), "go": r_swin > 0.45})
```

---

### O9 — Selective-Allocation Opportunity *(headline result for Paper A)*

**Question**: At fixed selection budgets, does entropy recover more positive decoder
transitions than matched random selection?

This is an **opportunity analysis** — it measures the potential for selective
allocation but does not prove computational savings (Paper B must show that).

```python
rng = np.random.default_rng(0)
budgets = np.array([5, 10, 20, 30, 50])

entropy_recovery, random_recovery = [], []

with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].cuda()
        lbl = batch["label"].squeeze(1).long().cpu().squeeze()

        pred_full = sliding_window_inference_1out(
            img, (96,96,96), 4, full_model, overlap=0.7).argmax(1).cpu().squeeze()
        logits_e  = sliding_window_inference_1out(
            img, (96,96,96), 4, effi_model, overlap=0.7)
        prob_e    = logits_e.softmax(1).cpu()
        pred_effi = logits_e.argmax(1).cpu().squeeze()
        ent = -(prob_e * torch.log(prob_e + 1e-8)).sum(1).squeeze()

        pos = ((pred_full == lbl) & (pred_effi != lbl)).float()
        # Budget denominator = union of foreground predictions and labels
        body = (lbl > 0) | (pred_full > 0) | (pred_effi > 0)
        ent_body = ent[body].numpy()
        pos_body = pos[body].numpy()

        total = pos_body.sum()
        if total == 0:
            continue

        # Entropy ranking
        order = np.argsort(ent_body)[::-1]
        entropy_recovery.append([
            pos_body[order[:max(1, int(len(order)*q/100))]].sum() / total
            for q in budgets
        ])
        # 100 random selections per subject
        random_recovery.append(np.mean([
            [pos_body[rng.choice(len(pos_body), max(1, int(len(pos_body)*q/100)),
                                 replace=False)].sum() / total for q in budgets]
            for _ in range(100)
        ], axis=0))

ent_arr  = np.asarray(entropy_recovery)
rand_arr = np.asarray(random_recovery)
print("Budget (%):       ", budgets)
print("Entropy recovery: ", ent_arr.mean(0).round(3))
print("Random recovery:  ", rand_arr.mean(0).round(3))

# Subject-level 95% CI via bootstrap
B = 2000
diffs = []
for _ in range(B):
    idx = rng.integers(len(ent_arr), size=len(ent_arr))
    diffs.append((ent_arr[idx] - rand_arr[idx]).mean(0))
diffs = np.array(diffs)
lo, hi = np.percentile(diffs, [2.5, 97.5], axis=0)
print("Entropy vs Random 95% CI lower:", lo.round(3))
print("Entropy vs Random 95% CI upper:", hi.round(3))

plt.figure(figsize=(7,5))
plt.plot(budgets, ent_arr.mean(0)*100, "o-", label="Entropy")
plt.fill_between(budgets, (ent_arr.mean(0)+lo)*100, (ent_arr.mean(0)+hi)*100, alpha=0.2)
plt.plot(budgets, rand_arr.mean(0)*100, "o--", label="Random (100 repeats)", color="gray")
plt.xlabel("Selected union-foreground voxels (%)")
plt.ylabel("Positive decoder transitions recovered (%)")
plt.title("O9: Selective-Allocation Opportunity")
plt.legend()
plt.savefig("/root/obs/O9_opportunity_curve.png", dpi=150)
plt.show()

save_obs("O9", {
    "budgets_pct": budgets.tolist(),
    "entropy_recovery_mean": ent_arr.mean(0).round(4).tolist(),
    "random_recovery_mean":  rand_arr.mean(0).round(4).tolist(),
    "ci_lower_95": lo.round(4).tolist(),
    "ci_upper_95": hi.round(4).tolist(),
    "go": bool((lo > 0).any()),   # any budget where lower CI > 0
})
```

**Go criterion**: entropy outperforms matched random at 10–30% budgets and the
lower bound of the 95% CI is above zero. Report the actual budget/recovery pair;
do not assume a specific concentration ratio in advance.

---

### O10 — Organ Size vs Difficulty

**Question**: Is difficulty just a proxy for small organs, or does entropy capture richer signal?

*Requires O4 to have been run (uses `organ_ent` dict from O4).*

```python
from scipy.stats import spearmanr

# Compute mean voxel size per organ from validation labels
organ_sizes_all = {n: [] for n in BTCV_NAMES}
for batch in val_loader:
    lbl = batch["label"].cpu().squeeze()
    for c, name in enumerate(BTCV_NAMES):
        mask = (lbl == c + 1)
        if mask.sum() > 0:
            organ_sizes_all[name].append(mask.float().sum().item())

sizes, diffs, names_o10 = [], [], []
print(f"{'Organ':15s}  {'Size (vx)':>10}  {'Difficulty':>10}")
for name in BTCV_NAMES:
    if organ_sizes_all[name] and organ_ent.get(name):   # organ_ent from O4
        s = float(np.mean(organ_sizes_all[name]))
        d = float(np.nanmean(organ_ent[name]))
        sizes.append(s); diffs.append(d); names_o10.append(name)
        print(f"{name:15s}  {s:10.0f}  {d:10.4f}")

r_size, _ = spearmanr(sizes, diffs)
print(f"\nOrgan size vs difficulty  Spearman ρ={r_size:.3f}")

# Figure: scatter size vs difficulty
plt.figure(figsize=(7, 5))
plt.scatter(sizes, diffs, zorder=3)
for n, s, d in zip(names_o10, sizes, diffs):
    plt.annotate(n, (s, d), fontsize=7, xytext=(4, 2), textcoords="offset points")
plt.xlabel("Mean organ size (voxels)"); plt.ylabel("Mean entropy (difficulty)")
plt.title(f"O10: Organ Size vs Difficulty  ρ={r_size:.2f}")
plt.tight_layout()
plt.savefig("/root/obs/O10_size_vs_difficulty.png", dpi=150)
plt.show()

save_obs("O10", {
    "spearman_rho_size_vs_difficulty": float(r_size),
    "organ_size": {n: round(s, 0) for n, s in zip(names_o10, sizes)},
    "organ_difficulty": {n: round(d, 4) for n, d in zip(names_o10, diffs)},
})
```

**Expected**: weak-to-moderate negative correlation (Spearman ρ ≈ −0.4 to −0.6),
but high residual variance — large organs (stomach, liver boundary) also show high
difficulty. This demonstrates entropy captures difficulty beyond organ size alone.

---

### O11 — Routing Signal Comparison

**Question**: Which test-time difficulty signal best predicts decoder gain?

Run after O5. Evaluate five signals on the BTCV validation set:

| Signal | Implementation | Overhead |
|---|---|---|
| Entropy | `-(p log p).sum(1)` over softmax | ~0 ms |
| Confidence | `1 - max(p)` over softmax | ~0 ms |
| Feature Variance | std of last decoder feature map | low |
| MC Dropout | variance over T=10 stochastic passes | T× latency |
| Boundary Probability | distance-to-foreground-boundary map | moderate |

For each signal compute:
- Pearson correlation with per-bin O5 net gain
- Inference latency overhead (ms/volume vs baseline)
- Stability: BTCV vs FeTA correlation difference

*Requires O5 to have been run (`bin_ent`, `bin_net` populated).*

```python
import time
from scipy.stats import pearsonr

assert len(bin_net) > 0, "Run O5 first to populate bin_ent and bin_net"

signal_results = {}

# ---------- Entropy (from O5 — zero extra compute) ----------
signal_results["Entropy"] = {
    "corr_btcv": float(pearsonr(bin_ent, bin_net)[0]),
    "latency_ms": 0.0,
}

# ---------- Confidence = 1 − max(softmax) ----------
conf_bins_signal, conf_bins_gain = [], []
t0 = time.perf_counter()
with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].cuda()
        lbl = batch["label"].squeeze(1).long().cpu().squeeze()
        logits_e = sliding_window_inference_1out(img, (96,96,96), 4, effi_model, overlap=0.7)
        prob_e   = logits_e.softmax(1).cpu()
        pred_effi = logits_e.argmax(1).cpu().squeeze()
        pred_full = sliding_window_inference_1out(img, (96,96,96), 4, full_model,
                                                   overlap=0.7).argmax(1).cpu().squeeze()
        conf = 1 - prob_e.max(1).values.squeeze()   # high = uncertain
        pos = ((pred_full == lbl) & (pred_effi != lbl)).float()
        neg = ((pred_full != lbl) & (pred_effi == lbl)).float()
        net_c = pos - neg
        for b in range(20):
            q_lo = conf.quantile(b/20).item()
            q_hi = conf.quantile((b+1)/20).item()
            mask = (conf >= q_lo) & (conf < q_hi)
            if mask.sum() > 100:
                conf_bins_signal.append(conf[mask].mean().item())
                conf_bins_gain.append(net_c[mask].mean().item())
lat_conf = (time.perf_counter() - t0) / len(val_loader) * 1000
signal_results["Confidence"] = {
    "corr_btcv": float(pearsonr(conf_bins_signal, conf_bins_gain)[0]),
    "latency_ms": round(lat_conf, 1),
}

# ---------- MC Dropout (T=10 forward passes) ----------
# Requires dropout layers to be active (model.train() mode during inference)
mc_bins_signal, mc_bins_gain = [], []
t0 = time.perf_counter()
effi_model.train()   # enable dropout
with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].cuda()
        lbl = batch["label"].squeeze(1).long().cpu().squeeze()
        T = 10
        preds = torch.stack([
            sliding_window_inference_1out(img, (96,96,96), 4, effi_model, overlap=0.7).softmax(1).cpu()
            for _ in range(T)
        ])
        mc_var = preds.var(0).sum(1).squeeze()   # sum of per-class variance
        pred_effi_mc = preds.mean(0).argmax(1).cpu().squeeze()
        pred_full_mc = sliding_window_inference_1out(img, (96,96,96), 4, full_model,
                                                      overlap=0.7).argmax(1).cpu().squeeze()
        pos = ((pred_full_mc == lbl) & (pred_effi_mc != lbl)).float()
        neg = ((pred_full_mc != lbl) & (pred_effi_mc == lbl)).float()
        net_mc = pos - neg
        for b in range(20):
            q_lo = mc_var.quantile(b/20).item()
            q_hi = mc_var.quantile((b+1)/20).item()
            mask = (mc_var >= q_lo) & (mc_var < q_hi)
            if mask.sum() > 100:
                mc_bins_signal.append(mc_var[mask].mean().item())
                mc_bins_gain.append(net_mc[mask].mean().item())
lat_mc = (time.perf_counter() - t0) / len(val_loader) * 1000
effi_model.eval()
signal_results["MC Dropout"] = {
    "corr_btcv": float(pearsonr(mc_bins_signal, mc_bins_gain)[0]) if len(mc_bins_signal) > 2 else float('nan'),
    "latency_ms": round(lat_mc, 1),
}

# ---------- Print summary table ----------
print(f"\n{'Signal':15s} {'Corr (BTCV)':>12} {'Latency ms':>12}")
for sig, vals in signal_results.items():
    print(f"{sig:15s} {vals['corr_btcv']:12.3f} {vals['latency_ms']:12.1f}")

# Fill in the table below manually after running
save_obs("O11", signal_results)
```

| Signal | Corr (BTCV) | Corr (FeTA) | Latency (ms) | Memory (MB) |
|---|---|---|---|---|
| Entropy | | | ≈ 0 | ≈ 0 |
| Confidence | | | ≈ 0 | ≈ 0 |
| Feature Var | | | | |
| MC Dropout | | | | |
| Boundary | | | | |

**Expected winner**: entropy (best corr/overhead ratio). O11 informs the AdaDec3D routing signal choice (Paper B).

---

## Part 4: Go / No-Go Decision

### Minimum criteria for Paper A submission

| Obs | Criterion | Result | Pass? |
|-----|-----------|--------|-------|
| O3 | Entropy–Error Pearson r > 0.60 | | ☐ |
| O5 | Net benefit rises with entropy; positive > negative in high-entropy bins | | ☐ |
| O9 | Entropy outperforms matched random at 10–30% budgets (CI lower > 0) | | ☐ |
| O2 | Entropy distribution is skewed (most voxels low entropy) | | ☐ |

**All four must pass** → proceed to Paper A write-up.

### Additional criteria for Paper B

| Obs | Criterion | Result | Pass? |
|-----|-----------|--------|-------|
| O7 | FeTA replication: net-gain/entropy r > 0.40 | | ☐ |
| O8 | SwinUNETR backbone: net-gain/entropy r > 0.45 | | ☐ |
| O11 | Entropy is best or tied-best routing signal | | ☐ |

**All three must pass** → proceed to [Experiment-Design-AdaDec3D.md](Experiment-Design-AdaDec3D.md).

---

## Part 5: Deliverables

### Notebooks

| Notebook | Observations |
|---|---|
| `obs_error.ipynb` | O1 |
| `obs_entropy.ipynb` | O2, O10 |
| `obs_correlation.ipynb` | O3, O4 |
| `obs_decoder_gain.ipynb` | O5, O9 |
| `obs_evolution.ipynb` | O6 |
| `obs_crossdataset.ipynb` | O7, O8 |
| `obs_routing_signal.ipynb` | O11 |

### Figures (Paper A)

| ID | Content |
|---|---|
| Fig 1 | Error map: boundary vs interior (O1) |
| Fig 2 | Entropy heatmap overlay (O2) |
| Fig 3 | Entropy–error scatter by bin (O3) |
| Fig 4 | Organ-wise difficulty bar plot (O4) |
| Fig 5 | Net gain vs entropy curve (O5) |
| Fig 6 | Difficulty evolution over training (O6) |
| **Fig 7** | **Opportunity curve: entropy vs random (O9) — headline** |
| Fig 8 | Organ size vs difficulty scatter (O10) |

### Tables (Paper A)

| ID | Content |
|---|---|
| T1 | Organ-wise DICE, entropy, positive/negative transitions (O4, O5) |
| T2 | Cross-dataset replication (O7) |
| T3 | Backbone consistency (O8) |
| T4 | Routing signal comparison (O11) |

---

## Part 6: Timeline — Phase 1 (Paper A)

```
Week 1: Setup
  [ ] Environment install and verify
  [ ] BTCV dataset download and sanity check (18 train, 12 val)
  [ ] 100-iter sanity run, confirm no OOM

Week 2-3: Baseline training
  [ ] E0 full 3DUXNET — 20 000 iter, ~4 sessions
  [ ] E1 EffiDec3D   — 20 000 iter, ~2 sessions (matched budget)
  [ ] Verify: 51.47 GMac, 3.16M params, mean DICE 79.0–79.5%
  [ ] Save milestone checkpoints at epochs 5, 10, 20, 30, 50

Week 4: Observations O1–O5 + O9 (critical gate)
  [ ] O1: boundary >> interior error rate confirmed
  [ ] O2: entropy distribution plotted, fraction reported
  [ ] O3: r > 0.60 → entropy is a valid routing signal
  [ ] O4: organ-wise difficulty table
  [ ] O5: positive and negative transitions by entropy bin
  [ ] O9: entropy vs random opportunity curve with 95% CI
  [ ] --- PAPER A GO / NO-GO DECISION ---

Week 5-6: Extended observations
  [ ] O6: difficulty evolution (milestone checkpoints)
  [ ] O7: FeTA replication (E0_feta + E1_feta)
  [ ] O8: SwinUNETR consistency (E0_swin + E1_swin)
  [ ] O10: organ size vs difficulty
  [ ] O11: routing signal comparison table

Week 7: Paper A draft
  [ ] Write Paper A manuscript
  [ ] Target venue: MIDL / MLMI / ISBI (submission typically Aug–Oct)
```
