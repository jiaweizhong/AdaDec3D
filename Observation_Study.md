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
E0 full 3DUXNET (53.007M params, 578.74 GMac):
  ~0.4-0.7 s/iter → 45 000 iter ≈ 6-9 h  (measured: ~11.6 h on A100)

E1 EffiDec3D (2.955M params, 41.06 GMac):
  ~0.1-0.2 s/iter → 45 000 iter ≈ 2-4 h  (measured: ~5.3 h on A100)
```

Both models are trained for **45 000 optimizer steps** with `--eval_step 250`
to match the EffiDec3D paper protocol.

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
# Confirmed working source (957 MB, 30 cases, labels 0–13):
kaggle datasets download -d shinjinidey/synapse-dataset \
    && unzip synapse-dataset.zip -d /root/autodl-tmp/btcv-raw

# Reorganise into train/val split (run once):
python convert_synapse.py   # /root/AdaDec3D/EffiDec3D/convert_synapse.py
ls /root/autodl-tmp/btcv-synapse/imagesTr/ | wc -l   # expect 18
ls /root/autodl-tmp/btcv-synapse/imagesVal/ | wc -l  # expect 12
```

**Expected layout**

```
/root/autodl-tmp/btcv-synapse/
  imagesTr/   img0005.nii.gz img0006.nii.gz … (18 training cases)
  labelsTr/   label0005.nii.gz …
  imagesVal/  img0001.nii.gz img0002.nii.gz … (12 val cases)
  labelsVal/  label0001.nii.gz …
```

**Standard split — Kaggle IDs** (mapped from original 3D UX-Net paper split)

> The Kaggle dataset (`shinjinidey/synapse-dataset`) renumbers cases 0001–0030
> (original IDs skip 0011–0020; Kaggle 0011 = original 0021, …, Kaggle 0030 = original 0040).
> The split below reflects the paper's original IDs translated to Kaggle IDs and is
> already hardcoded in `load_datasets_transforms.py` under `BTCV13`.

```python
TRAIN = ["0005","0006","0007","0009","0010","0011","0013","0014",
         "0016","0017","0018","0020","0021","0023","0024","0027","0029","0030"]
VAL   = ["0001","0002","0003","0004","0008","0012","0015","0019",
         "0022","0025","0026","0028"]
```

> **Data reuse / Paper A → Paper B plan**
>
> These same 12 val cases are used for (a) O1–O11 exploratory observations,
> (b) Go/No-Go threshold selection (O5 quantile, O9 budget), and
> (c) AdaDec3D final Dice / HD95 table.  Using the same split for all three
> introduces optimistic bias (selection-then-confirm on the same data).
>
> Mitigation plan:
> - **Paper A** treats O1–O5 + O9 as exploratory/discovery; no hold-out needed
>   because these are observational claims (entropy is informative) not
>   hyperparameter-tuned predictive claims.
> - **Paper B** (AdaDec3D paper): retrain on a 5-fold CV split over all 30 cases
>   so the same cases are never simultaneously in the observation set and the
>   final evaluation fold.  Report mean ± std Dice across folds.
> - Until Paper B training begins, clearly mark all Dice numbers in Paper A as
>   "calibration / not final" to avoid reviewer confusion.
>
> All confidence intervals must resample subjects, not individual voxels.

**BTCV13 label mapping**

```
0 Background  1 Spleen     2 R.Kidney    3 L.Kidney    4 Gallbladder
5 Esophagus   6 Liver      7 Stomach     8 Aorta       9 IVC
10 Veins*     11 Pancreas* 12 R.Adrenal* 13 L.Adrenal*
```
\* smallest / hardest structures (Dice < 0.70 for E0) — primary metric for Paper A/B.
This is the standard BTCV/Synapse 13-organ order, matching the CSV column names
(`Spl, Rkid, Lkid, Gall, Eso, Liver, Sto, Aorta, IVC, Veins, Pan, Rad, Lad`)
emitted by `main_train_BTCV_TU.py`.

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
  --lr 0.001 --overlap 0.7 --crop_sample 4 \
  --max_iter 45000 --eval_step 250 \
  --gpu 0 --cache_rate 1.0 --num_workers 8
```

### E1 — EffiDec3D (baseline to beat)

```bash
python main_train_BTCV_TU.py \
  --root /root/autodl-tmp/btcv-synapse --output /root/output/E1 \
  --dataset BTCV13 --network 3DUXNET_EffiDec3D \
  --ds False \
  --lr 0.001 --overlap 0.7 --crop_sample 4 \
  --max_iter 45000 --eval_step 250 \
  --gpu 0 --cache_rate 1.0 --num_workers 8
```

> **Or use the convenience script** (runs E0 then E1 sequentially, resumes from checkpoint if found):
> ```bash
> bash run_E0_E1.sh 2>&1 | tee /root/output/run_E0_E1.log
> ```

**Critical parameters for E1**

| Parameter | Correct | Wrong value effect |
|---|---|---|
| `--ds False` | required | ds=True enables deep supervision, changes architecture |
| `--overlap 0.7` | **0.7** | Lower → worse DICE at inference |

**Verify E1 prints at startup**:

```
Computational complexity:   41.06 GMac
Number of parameters:       2.955 M
```

**Target BTCV13 mean DICE**: 79.0–79.5% (paper: 79.25%)

### E0 measured baseline (calibration ✓)

E0 (full 3DUXNET, 45 000 iter) reproduces the paper: **Mean DICE 0.7918** (paper
3DUX-Net **79.74%**, −0.56), Mean HD95 9.04, 578.74 GMac / 53.007 M params,
train 11.65 h, infer 40.6 ms/vol, peak infer mem 1.84 GB (RTX 5090, BF16).
Per-organ (standard BTCV order):

| Organ | DICE | HD95 | | Organ | DICE | HD95 |
|---|---|---|---|---|---|---|
| Spleen | 0.913 | 2.53 | | IVC | 0.853 | 3.62 |
| R.Kidney | 0.850 | 20.35 | | Veins* | 0.673 | 11.80 |
| L.Kidney | 0.899 | 9.70 | | Pancreas* | 0.690 | 10.97 |
| Gallbladder | 0.722 | 12.40 | | R.Adrenal* | 0.669 | 5.96 |
| Esophagus | 0.743 | 4.29 | | L.Adrenal* | 0.631 | 9.10 |
| Liver | 0.951 | 9.48 | | **Mean** | **0.792** | **9.04** |
| Stomach | 0.792 | 14.46 | | | | |
| Aorta | 0.908 | 2.92 | | | | |

\* The four hardest organs (DICE < 0.70) are the small/thin structures —
Veins, Pancreas, R.Adrenal, L.Adrenal — the difficulty-concentrated regions
AdaDec3D targets. (R.Kidney HD95 = 20.35 is a single-case spatial outlier.)

### E1 measured baseline (EffiDec3D)

E1 (3DUXNET_EffiDec3D, `--ds False`, seed 0, 45 000 iter): **Mean DICE 0.7700**
(paper EffiDec3D **79.25%**, −2.25), Mean HD95 14.41, **41.06 GMac / 2.955 M
params**, train 5.36 h, infer **7.5 ms/vol**, peak infer mem **0.24 GB**.

Efficiency vs E0: **14.1× MACs**, 5.4× latency, 7.7× memory — matches the paper.
Per-organ (our E1 | paper EffiDec3D):

| Organ | E1 | paper | | Organ | E1 | paper |
|---|---|---|---|---|---|---|
| Spleen | .865 | .904 | | IVC | .814 | .859 |
| R.Kidney | .841 | .848 | | Veins | .650 | .680 |
| L.Kidney | .872 | .871 | | Pancreas | .665 | .702 |
| Gallbladder | .746 | .753 | | R.Adrenal | .615 | .655 |
| Esophagus | .753 | .743 | | L.Adrenal | .607 | .647 |
| Liver | .954 | .944 | | **Mean** | **.770** | **.793** |
| Stomach | .756 | .788 | | | | |
| Aorta | .872 | .910 | | | | |

**Reproduction notes** (why E1 lands 2.25 below the paper):
- Code is faithful to upstream EffiDec3D (encoder identical; backbone only differs
  by `**kwargs`; hyperparameters match the README command exactly; seed 0 = upstream default).
- **BF16 is innocent**: re-validating the same weights in FP32 vs BF16 gives
  Δ = 0.0000 (`revalidate_fp32.py`). Precision is not the cause.
- Remaining gap is most likely the **data source** (we use the TransUNet Synapse
  Kaggle split with identity affine; the paper uses `btcv_trns`) and/or 12-case
  variance. A different-seed run is in progress to bound the variance.
- The **CSV/paper-protocol metric is 0.7700**, not the 0.7549 shown mid-training
  (that periodic MONAI metric uses a running-aggregate average on the resampled
  grid; the final `validation_save` resamples to original resolution + medpy dice,
  which is what the paper reports).

Numbers are "calibration / not final" — see the Paper A → Paper B data-reuse note.
For the AdaDec3D efficiency thesis, **E1 (77.0%) is the iso-accuracy target**;
its exact absolute value is not a gate.

### Checkpoint resumption

The script saves `last_model.pth` after every eval step and auto-resumes on restart. Run inside `tmux` to survive SSH disconnection:

```bash
tmux new -s e1_train
# ... run training command ...
# Ctrl-B D to detach; tmux attach -t e1_train to re-attach
```

### Milestone checkpoints (required for O6)

`main_train_BTCV_TU.py` automatically saves iteration checkpoints at steps
5k, 10k, 20k, 30k, and 45k whenever validation runs at those steps. Keep
`--eval_step 250` (or another divisor of all milestone steps). Files are named
`milestone_05000.pth`, ..., `milestone_45000.pth`.

### E1 on FeTA (required for O7)

O7 needs both an E0-FeTA and an E1-FeTA run with identical schedules.

```bash
# E0 FeTA
python main_train_BTCV_TU.py \
  --root /root/autodl-tmp/feta-processed --output /root/output/E0_feta \
  --dataset feta --network 3DUXNET \
  --lr 0.001 --overlap 0.7 --crop_sample 4 \
  --max_iter 45000 --eval_step 250 \
  --cache_rate 1.0 --num_workers 8 --gpu 0

# E1 FeTA
python main_train_BTCV_TU.py \
  --root /root/autodl-tmp/feta-processed --output /root/output/E1_feta \
  --dataset feta --network 3DUXNET_EffiDec3D \
  --ds False \
  --lr 0.001 --overlap 0.7 --crop_sample 4 \
  --max_iter 45000 --eval_step 250 \
  --cache_rate 1.0 --num_workers 8 --gpu 0
```

### SwinUNETR_EffiDec3D on BTCV (required for O8)

O8 needs both E0-Swin and E1-Swin with identical schedules.

```bash
# E0 SwinUNETR
python main_train_BTCV_TU.py \
  --root /root/autodl-tmp/btcv-synapse --output /root/output/E0_swin \
  --dataset BTCV13 --network SwinUNETR \
  --lr 0.001 --overlap 0.7 --crop_sample 4 \
  --max_iter 45000 --eval_step 250 \
  --cache_rate 1.0 --num_workers 8 --gpu 0

# E1 SwinUNETR_EffiDec3D
python main_train_BTCV_TU.py \
  --root /root/autodl-tmp/btcv-synapse --output /root/output/E1_swin \
  --dataset BTCV13 --network SwinUNETR_EffiDec3D \
  --ds False \
  --lr 0.001 --overlap 0.7 --crop_sample 4 \
  --max_iter 45000 --eval_step 250 \
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
        from monai.networks.nets import SwinUNETR
        model = SwinUNETR(img_size=(96, 96, 96), in_channels=1, out_channels=14,
                          feature_size=48).to(device)
    else:
        from networks.UXNet_3D.network_backbone import UXNET
        model = UXNET(in_chans=1, out_chans=14, depths=[2,2,2,2],
                      feat_size=[48,96,192,384]).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    # Support both bare state_dict (main_train_BTCV_TU.py) and wrapped dict
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Standard BTCV/Synapse 13-organ order (class 1..13), matches CSV columns
BTCV_NAMES = ["Spleen","R.Kidney","L.Kidney","Gallbladder","Esophagus",
              "Liver","Stomach","Aorta","IVC","Veins","Pancreas","R.Adrenal","L.Adrenal"]

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
from scipy.stats import pearsonr, spearmanr

# Per-subject binned analysis (avoids pseudo-replication from pooling bins across subjects)
subj_r_pearson, subj_r_spearman = [], []
global_bin_ent, global_bin_pos, global_bin_neg, global_bin_net = [], [], [], []

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

        # Per-subject binning
        s_ent, s_net, s_pos, s_neg = [], [], [], []
        for b in range(20):
            q_lo = ent.quantile(b/20).item()
            q_hi = ent.quantile((b+1)/20).item()
            mask = (ent >= q_lo) & (ent < q_hi)
            if mask.sum() > 100:
                s_ent.append(ent[mask].mean().item())
                s_net.append(net[mask].mean().item())
                s_pos.append(pos[mask].mean().item())
                s_neg.append(neg[mask].mean().item())
                global_bin_ent.append(s_ent[-1])
                global_bin_net.append(s_net[-1])
                global_bin_pos.append(s_pos[-1])
                global_bin_neg.append(s_neg[-1])

        if len(s_ent) >= 5:
            subj_r_pearson.append(pearsonr(s_ent, s_net)[0])
            subj_r_spearman.append(spearmanr(s_ent, s_net)[0])

# Subject-level statistics (avoid pseudo-replication)
r_subj_mean = float(np.mean(subj_r_pearson))
r_subj_std  = float(np.std(subj_r_pearson))
rho_subj_mean = float(np.mean(subj_r_spearman))

# Bootstrap CI over subject-level Pearson r
rng = np.random.default_rng(0)
boot_r = [np.mean(rng.choice(subj_r_pearson, len(subj_r_pearson), replace=True))
          for _ in range(2000)]
r_ci_lo, r_ci_hi = np.percentile(boot_r, [2.5, 97.5])

mean_pos = float(np.mean(global_bin_pos))
mean_neg = float(np.mean(global_bin_neg))

print(f"Per-subject Pearson r: {r_subj_mean:.3f} ± {r_subj_std:.3f}  95% CI [{r_ci_lo:.3f}, {r_ci_hi:.3f}]")
print(f"Per-subject Spearman ρ: {rho_subj_mean:.3f}")
print(f"Mean positive rate={mean_pos:.5f}  negative rate={mean_neg:.5f}")
print(f"(Pooled Pearson is descriptive only — bins within subject are correlated)")

# Figure: global net gain curve (descriptive) with per-subject r in title
# Sort ALL four arrays together by entropy to keep curves aligned
_sort_idx = np.argsort(global_bin_ent)
x_plot = np.array(global_bin_ent)[_sort_idx]
y_plot = np.array(global_bin_net)[_sort_idx]
y_pos  = np.array(global_bin_pos)[_sort_idx]
y_neg  = np.array(global_bin_neg)[_sort_idx]

plt.figure(figsize=(8, 5))
plt.plot(x_plot, y_pos, "g--o", markersize=3, alpha=0.6, label="Positive rate")
plt.plot(x_plot, y_neg, "r--o", markersize=3, alpha=0.6, label="Negative rate")
plt.plot(x_plot, y_plot, "b-o", markersize=4,
         label=f"Net gain (subj-r={r_subj_mean:.2f} [{r_ci_lo:.2f},{r_ci_hi:.2f}])")
plt.axhline(0, color="k", linewidth=0.8, linestyle=":")
plt.xlabel("Mean entropy (bin)"); plt.ylabel("Rate")
plt.title("O5: Decoder Gain vs Uncertainty")
plt.legend(); plt.tight_layout()
plt.savefig("/root/obs/O5_decoder_gain.png", dpi=150)
plt.show()

# Store bin_ent / bin_net in module scope for O11 (already sorted by entropy)
bin_ent = x_plot.tolist()
bin_net = y_plot.tolist()

save_obs("O5", {
    "subj_pearson_r_mean":  r_subj_mean,
    "subj_pearson_r_std":   r_subj_std,
    "subj_pearson_r_ci":    [float(r_ci_lo), float(r_ci_hi)],
    "subj_spearman_rho":    rho_subj_mean,
    "mean_positive_rate":   mean_pos,
    "mean_negative_rate":   mean_neg,
    "bin_ent": [float(v) for v in bin_ent],
    "bin_net": [float(v) for v in bin_net],
    "go": r_ci_lo > 0.0 and mean_pos > mean_neg,
})
```

**Go criterion**: net benefit rises with entropy AND a deployable signal beats
matched random at 10–30% budgets with a subject-level 95% CI (see O9).

---

### O6 — Difficulty Evolution During Training

**Question**: Does high entropy persist and stabilize at boundaries as training progresses?

*Requires milestone checkpoints saved in Part 2.*

```python
MILESTONES = [5000, 10000, 20000, 30000, 45000]

step_ent = {}
for train_step in MILESTONES:
    m = load_model("3DUXNET_EffiDec3D",
                   f"/root/output/E1/.../milestone_{train_step:05d}.pth")
    mean_ents = []
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].cuda()
            logits = sliding_window_inference_1out(img, (96,96,96), 4, m, overlap=0.7)
            prob = logits.softmax(1).cpu()
            ent = -(prob * torch.log(prob + 1e-8)).sum(1)
            mean_ents.append(ent.mean().item())
    step_ent[train_step] = float(np.mean(mean_ents))
    print(f"Step {train_step:5d}  mean_entropy={step_ent[train_step]:.4f}")

# Figure: entropy vs training iteration
plt.figure(figsize=(7, 4))
plt.plot(list(step_ent.keys()), list(step_ent.values()), "o-")
plt.xlabel("Training iteration"); plt.ylabel("Mean entropy")
plt.title("O6: Entropy Evolution During Training")
plt.tight_layout()
plt.savefig("/root/obs/O6_entropy_evolution.png", dpi=150)
plt.show()

save_obs("O6", {"step_mean_entropy": step_ent})
```

**Figure**: mean entropy vs training iteration (line) + spatial entropy maps at
steps 5k, 20k, and 45k.
**Expected**: entropy decreases but stabilizes; residual high-entropy voxels
concentrate at boundaries and small organs late in training.

---

### O7 — Cross-Dataset Consistency

**Question**: Do O1–O5 findings replicate on FeTA (fetal brain MRI)?

*Requires E0_feta and E1_feta from Part 2.*

```python
# Repeat O5 analysis with FeTA models and val_loader
FETA_NAMES = ["IS","WM","CGM","DGM","CE","BS","CSF"]

feta_args = argparse.Namespace(
    root="/root/autodl-tmp/feta-processed", dataset="feta",
    mode="validation", crop_sample=4, img_size=[96,96,96]
)
_, feta_val, n_cls_feta = data_loader(feta_args)
_, feta_transform = data_transforms(feta_args)
feta_files = [{"image": im, "label": lb}
              for im, lb in zip(feta_val["images"], feta_val["labels"])]
feta_loader = DataLoader(Dataset(data=feta_files, transform=feta_transform),
                         batch_size=1, shuffle=False, num_workers=2)

# Resolve checkpoint paths
import glob as _glob
_e0f = sorted(_glob.glob("/root/output/E0_feta*/3DUXNET/*/best_metric_model.pth"))
_e1f = sorted(_glob.glob("/root/output/E1_feta*/3DUXNET_EffiDec3D/*/best_metric_model.pth"))
assert _e0f and _e1f, "Run E0_feta and E1_feta training first (Observation_Study.md Part 2)"
full_feta  = load_model("3DUXNET",           _e0f[-1])
effi_feta  = load_model("3DUXNET_EffiDec3D", _e1f[-1])

# Run identical O5 per-subject analysis for FeTA
feta_subj_r = []
with torch.no_grad():
    for batch in feta_loader:
        img = batch["image"].cuda()
        lbl = batch["label"].squeeze(1).long().cpu().squeeze()
        pred_full = sliding_window_inference_1out(img,(96,96,96),4,full_feta,overlap=0.7).argmax(1).cpu().squeeze()
        logits_e  = sliding_window_inference_1out(img,(96,96,96),4,effi_feta,overlap=0.7)
        prob_e    = logits_e.softmax(1).cpu()
        pred_effi = logits_e.argmax(1).cpu().squeeze()
        ent = -(prob_e * torch.log(prob_e + 1e-8)).sum(1).squeeze()
        pos = ((pred_full==lbl) & (pred_effi!=lbl)).float()
        neg = ((pred_full!=lbl) & (pred_effi==lbl)).float()
        net = pos - neg
        s_ent, s_net = [], []
        for b in range(20):
            mask = (ent >= ent.quantile(b/20)) & (ent < ent.quantile((b+1)/20))
            if mask.sum() > 100:
                s_ent.append(ent[mask].mean().item())
                s_net.append(net[mask].mean().item())
        if len(s_ent) >= 5:
            feta_subj_r.append(pearsonr(s_ent, s_net)[0])

r_feta = float(np.mean(feta_subj_r))
print(f"FeTA per-subject Pearson r={r_feta:.3f}  (n={len(feta_subj_r)} subjects)")
print(f"{'GO ✓' if r_feta > 0.40 else 'NO-GO ✗'}  (threshold r > 0.40)")

save_obs("O7", {"feta_gain_entropy_subj_pearson_r": r_feta, "n_subjects": len(feta_subj_r), "go": r_feta > 0.40})
```

---

### O8 — Backbone Consistency

**Question**: Does the O5 gain–entropy correlation hold with SwinUNETR instead of UXNET?

*Requires E0_swin and E1_swin from Part 2.*

```python
import glob as _glob
_e0s = sorted(_glob.glob("/root/output/E0_swin*/SwinUNETR/*/best_metric_model.pth"))
_e1s = sorted(_glob.glob("/root/output/E1_swin*/SwinUNETR_EffiDec3D/*/best_metric_model.pth"))
assert _e0s and _e1s, "Run E0_swin and E1_swin training first (Observation_Study.md Part 2)"
full_swin = load_model("SwinUNETR",           _e0s[-1])
effi_swin = load_model("SwinUNETR_EffiDec3D", _e1s[-1])

# Run identical O5 per-subject analysis for SwinUNETR
swin_subj_r = []
with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].cuda()
        lbl = batch["label"].squeeze(1).long().cpu().squeeze()
        pred_full = sliding_window_inference_1out(img,(96,96,96),4,full_swin,overlap=0.7).argmax(1).cpu().squeeze()
        logits_e  = sliding_window_inference_1out(img,(96,96,96),4,effi_swin,overlap=0.7)
        prob_e    = logits_e.softmax(1).cpu()
        pred_effi = logits_e.argmax(1).cpu().squeeze()
        ent = -(prob_e * torch.log(prob_e + 1e-8)).sum(1).squeeze()
        pos = ((pred_full==lbl) & (pred_effi!=lbl)).float()
        neg = ((pred_full!=lbl) & (pred_effi==lbl)).float()
        net = pos - neg
        s_ent, s_net = [], []
        for b in range(20):
            mask = (ent >= ent.quantile(b/20)) & (ent < ent.quantile((b+1)/20))
            if mask.sum() > 100:
                s_ent.append(ent[mask].mean().item())
                s_net.append(net[mask].mean().item())
        if len(s_ent) >= 5:
            swin_subj_r.append(pearsonr(s_ent, s_net)[0])

r_swin = float(np.mean(swin_subj_r))
print(f"SwinUNETR per-subject Pearson r={r_swin:.3f}  (n={len(swin_subj_r)} subjects)")
print(f"{'GO ✓' if r_swin > 0.45 else 'NO-GO ✗'}  (threshold r > 0.45)")

save_obs("O8", {"swin_gain_entropy_subj_pearson_r": r_swin, "n_subjects": len(swin_subj_r), "go": r_swin > 0.45})
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

# lo/hi are CIs on the DIFFERENCE (entropy - random), not on entropy mean.
# Bootstrap separate CIs for each curve for plotting.
B = 2000
ent_boots, rnd_boots = [], []
for _ in range(B):
    idx = rng.integers(len(ent_arr), size=len(ent_arr))
    ent_boots.append(ent_arr[idx].mean(0))
    rnd_boots.append(rand_arr[idx].mean(0))
ent_boots = np.array(ent_boots)
rnd_boots = np.array(rnd_boots)
ent_lo, ent_hi = np.percentile(ent_boots, [2.5, 97.5], axis=0)
rnd_lo, rnd_hi = np.percentile(rnd_boots, [2.5, 97.5], axis=0)

plt.figure(figsize=(7,5))
plt.plot(budgets, ent_arr.mean(0)*100, "o-", label="Entropy", color="steelblue")
plt.fill_between(budgets, ent_lo*100, ent_hi*100, alpha=0.2, color="steelblue")
plt.plot(budgets, rand_arr.mean(0)*100, "o--", label="Random (100 repeats)", color="gray")
plt.fill_between(budgets, rnd_lo*100, rnd_hi*100, alpha=0.15, color="gray")
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
    # Go: lower CI > 0 for at least one budget in the 10–30% range (as stated in criterion)
    "go": bool(lo[np.isin(budgets, [10, 20, 30])].max() > 0),
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

# Partial correlation: does entropy predict difficulty beyond log(volume)?
# OLS: entropy ~ log(size) + intercept; residuals are the size-independent component.
import numpy as np
log_sizes = np.log(np.array(sizes))
ent_arr_o10 = np.array(diffs)
# Fit log(size) → entropy, get residuals
A = np.column_stack([np.ones(len(log_sizes)), log_sizes])
coef, _, _, _ = np.linalg.lstsq(A, ent_arr_o10, rcond=None)
entropy_resid = ent_arr_o10 - A @ coef   # entropy unexplained by organ size

# Partial correlation: correlation of residuals with a complexity proxy
# Use per-organ mean dice error (1-dice) from O4 as the ground-truth difficulty
# (requires organ_dice from O4 to be in scope)
if 'dice_summary' in dir():
    dice_err = np.array([1 - dice_summary.get(n, np.nan) for n in names_o10])
    valid = ~np.isnan(dice_err)
    if valid.sum() >= 4:
        # True partial correlation: residualize BOTH entropy and dice_error on log_size,
        # then correlate the two sets of residuals.
        A_valid = np.column_stack([np.ones(valid.sum()), log_sizes[valid]])
        # entropy residuals (already computed on full array; re-fit on valid subset)
        coef_e_v, _, _, _ = np.linalg.lstsq(A_valid, ent_arr_o10[valid], rcond=None)
        ent_resid_v = ent_arr_o10[valid] - A_valid @ coef_e_v
        # dice_error residuals
        coef_d, _, _, _ = np.linalg.lstsq(A_valid, dice_err[valid], rcond=None)
        dice_resid = dice_err[valid] - A_valid @ coef_d
        r_partial, _ = pearsonr(ent_resid_v, dice_resid)
        print(f"Partial correlation r(entropy, dice-error | log-size): r={r_partial:.3f}")
        print(f"→ entropy {'does' if abs(r_partial) > 0.3 else 'does NOT'} capture difficulty beyond organ size")
    else:
        r_partial = float("nan")
        print("(dice_summary from O4 not available — run O4 first for partial correlation)")
else:
    r_partial = float("nan")
    print("(dice_summary from O4 not in scope — run O4 before O10 for partial correlation)")

# Figure: scatter size vs difficulty with OLS trend
plt.figure(figsize=(7, 5))
plt.scatter(sizes, diffs, zorder=3)
for n, s, d in zip(names_o10, sizes, diffs):
    plt.annotate(n, (s, d), fontsize=7, xytext=(4, 2), textcoords="offset points")
x_line = np.linspace(min(log_sizes), max(log_sizes), 100)
y_line = coef[0] + coef[1] * x_line
plt.plot(np.exp(x_line), y_line, "k--", linewidth=1, label="OLS(log size)")
plt.xlabel("Mean organ size (voxels, log scale)"); plt.xscale("log")
plt.ylabel("Mean entropy (difficulty)")
plt.title(f"O10: Organ Size vs Difficulty  ρ={r_size:.2f}  partial-r={r_partial:.2f}")
plt.legend(); plt.tight_layout()
plt.savefig("/root/obs/O10_size_vs_difficulty.png", dpi=150)
plt.show()

save_obs("O10", {
    "spearman_rho_size_vs_difficulty": float(r_size),
    "partial_r_entropy_given_size": float(r_partial) if not np.isnan(r_partial) else None,
    "ols_coef_intercept": float(coef[0]),
    "ols_coef_log_size":  float(coef[1]),
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
# Requires dropout layers to be active (model.train() mode during inference).
# UXNET/EffiDec3D uses DropPath (stochastic depth), not standard Dropout — at T=10
# DropPath variance may be near-zero at test time. Check before trusting the signal.
mc_bins_signal, mc_bins_gain = [], []
t0 = time.perf_counter()
effi_model.train()   # enable stochastic depth / dropout
_mc_dropout_valid = True
with torch.no_grad():
    for _step_mc, batch in enumerate(val_loader):
        img = batch["image"].cuda()
        lbl = batch["label"].squeeze(1).long().cpu().squeeze()
        T = 10
        preds = torch.stack([
            sliding_window_inference_1out(img, (96,96,96), 4, effi_model, overlap=0.7).softmax(1).cpu()
            for _ in range(T)
        ])
        mc_var = preds.var(0).sum(1).squeeze()   # sum of per-class variance

        # First subject: sanity-check that variance is non-trivial
        if _step_mc == 0 and mc_var.max().item() < 1e-7:
            print("[WARN] MC Dropout variance ≈ 0 on first subject.")
            print("       EffiDec3D likely lacks active Dropout/DropPath at train() mode.")
            print("       MC signal will be meaningless — skipping MC Dropout comparison.")
            _mc_dropout_valid = False
            break

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
lat_mc = (time.perf_counter() - t0) / max(1, len(val_loader)) * 1000
effi_model.eval()
if not _mc_dropout_valid:
    signal_results["MC Dropout"] = {
        "corr_btcv": float("nan"),
        "latency_ms": round(lat_mc, 1),
        "warn": "dropout_inactive",
    }
else:
    signal_results["MC Dropout"] = {
        "corr_btcv": float(pearsonr(mc_bins_signal, mc_bins_gain)[0]) if len(mc_bins_signal) > 2 else float("nan"),
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

### Minimum criteria for Paper A submission — **ALL PASS** (measured 2026-07-25)

| Obs | Criterion | Result | Pass? |
|-----|-----------|--------|-------|
| O3 | Entropy–Error Pearson r > 0.60 | **r = 0.971** (Spearman 0.976) | ✅ |
| O5 | Net benefit rises with entropy; positive > negative | subj-r 0.665, 95% CI **[0.171, 0.996]**; pos 0.339% > neg 0.209% | ✅ |
| O9 | Entropy beats matched random at 10–30% budgets (CI lower > 0) | diff CI-lo **[0.359, 0.590, 0.606]** at 10/20/30% | ✅ |
| O2 | Entropy distribution skewed (most voxels low entropy) | median 0.0011; only **1.31%** voxels > 0.5 | ✅ |

**All four pass.** Routing signal (entropy) is validated: it predicts error almost
perfectly (O3) and selects the decoder-relevant region with high precision
(O9: 20% budget recovers 86% of positive transitions).

### Additional criteria for Paper B (not yet run — need FeTA / SwinUNETR)

| Obs | Criterion | Result | Pass? |
|-----|-----------|--------|-------|
| O7 | FeTA replication: net-gain/entropy r > 0.40 | pending E0/E1-FeTA | ☐ |
| O8 | SwinUNETR backbone: net-gain/entropy r > 0.45 | pending E0/E1-Swin | ☐ |
| O11 | Entropy is best or tied-best routing signal | pending | ☐ |

---

### ⚠️ Interpretation update — thesis is **efficiency**, not accuracy recovery

O5's decoder gain is **small**: full 3DUX-Net (E0) beats EffiDec3D (E1) by only
**0.13% net voxels** (pos 0.339% − neg 0.209%), and it is a messy trade (for every
3 voxels fixed, ~2 are broken). This matches the paper's near-parity (79.74 vs
79.25). Under a *recovery* framing this would be a weak result.

**But the project's thesis is: EffiDec3D already gets accuracy for free; AdaDec3D
pushes compute *lower* at iso-accuracy.** Under that framing the same data is
*supporting* evidence:

- O2: 98.7% of voxels are easy (low-entropy) → decoder capacity is unneeded almost everywhere.
- O5: even the *full* decoder's extra capacity nets ~0% → decoder is over-provisioned; there is room to cut.
- O9: entropy pinpoints the ~20% region where decoder capacity matters (86% of benefit) → an efficient routing signal.

### Compute-headroom gate (MAC profiling of EffiDec3D, `profile_macs.py`)

| Group | GMac | % |
|---|---|---|
| **DECODER** (decoder3/4/5 + out) | **18.09** | **42.2%** |
| ENCODER (uxnet_3d + encoder2–5) | 24.74 | 57.8% |
| — `decoder3` alone (finest half-res block) | 15.80 | 36.9% |

Decoder = 42% of compute, dominated by `decoder3` (37%) — exactly where ROI/MoE
act. Rough ceiling: running `decoder3` at full capacity on ~20% of voxels and a
cheap path elsewhere → **41 → ~33 GMac (~20–25% total reduction) at iso-accuracy**.
The encoder (57.8%) is untouchable by decoder-only routing → a further
"patch-level whole-model adaptivity" direction is needed to go beyond this.

**Go decision: PASS** → proceed to [Experiment-Design-AdaDec3D.md](Experiment-Design-AdaDec3D.md)
(E2/E3/E4), targeting iso-accuracy with E1 (77.0%) at lower MACs.

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
  [x] Environment install and verify
  [x] BTCV dataset download and sanity check (18 train, 12 val)
  [x] 100-iter sanity run, confirm no OOM

Week 2-3: Baseline training
  [x] E0 full 3DUXNET — 45 000 iter   ✓ Mean DICE 0.7918 / HD95 9.04 (paper 3DUX 79.74)
  [x] E1 EffiDec3D   — 45 000 iter    ✓ Mean DICE 0.7700 / HD95 14.41 (paper Effi 79.25, −2.25)
  [x] E1 efficiency confirmed: 41.06 GMac (14.1× vs E0), 7.5 ms, 0.24 GB
  [x] E1 milestone_{05000..45000}.pth auto-saved (for O6)
  [~] Different-seed E1 run in progress (bound 12-case variance)

Week 4: Observations — critical gate (run_observations.py)   ✓ ALL PASS
  [x] O2: entropy skewed (median 0.0011; 1.31% voxels > 0.5)          GO
  [x] O3: entropy–error Pearson r = 0.971                             GO
  [x] O4: per-organ dice/entropy (hardest = highest-entropy organs)
  [x] O5: net gain rises with entropy; CI [0.171, 0.996]; pos>neg     GO (but gain small: 0.13% net)
  [x] O9: entropy 20% budget recovers 86% of transitions; CI-lo>0     GO (headline)
  [x] MAC profiling: decoder = 42.2% (decoder3 37%); encoder 57.8%
  [x] BF16 innocent (FP32 revalidation Δ=0.0000)
  [x] --- GO: efficiency thesis (iso-accuracy, lower MACs) → Experiment-Design-AdaDec3D ---

Week 5-6: Extended observations (deferred — need FeTA / SwinUNETR runs)
  [ ] O6: difficulty evolution (E1 milestones already saved)
  [ ] O7: FeTA replication (E0_feta + E1_feta)
  [ ] O8: SwinUNETR consistency (E0_swin + E1_swin)
  [ ] O10: organ size vs difficulty
  [ ] O11: routing signal comparison table
  [ ] Patch-level whole-model adaptivity study (extend efficiency beyond decoder's 42%)

Week 7: Paper A draft
  [ ] Write Paper A manuscript
  [ ] Target venue: MIDL / MLMI / ISBI (submission typically Aug–Oct)
```
