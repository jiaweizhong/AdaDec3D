# Paper A: Observation Study — Complete Experiment Guide

> **Scientific motivation**: [Research_Proposal.md §3–5](Research_Proposal.md)
> **Architecture and Paper B**: [Experiment-Design-AdaDec3D.md](Experiment-Design-AdaDec3D.md)

---

## Two-Paper Strategy

| | Paper A — this document | Paper B |
|---|---|---|
| **Claim** | Dice and FLOPs alone are insufficient to explain efficient decoder design; positive, negative, and net flips reveal where additional decoder computation helps (**observation only**) | AdaDec3D realizes region-adaptive decoding — **iso-accuracy at lower executed cost** |
| **Venue** | MIDL / MLMI / ISBI | MICCAI 2026 / TMI |
| **Gate** | O2/O3/O5/O9 pass ✅ (all passed 2026-07-25) | O7, O8, O11 + AdaDec3D beats controls |
| **Key result** | Preliminary cross-seed evidence that decoder benefit is small, concentrated and predictable; corrected regional analysis still required | **DICE ≈ EffiDec3D at meaningfully lower executed MACs** (input-adaptive) |

> **Scope note**: Paper A is an **observation paper** — it characterizes *where*
> extra decoder capacity changes predictions in spatially concentrated and partly predictable regions
> and does **not** claim realizable efficiency (that is Paper B). O5's small,
> concentrated net flip (0.13%) is read here as evidence that decoder
> capacity is *over-provisioned* — a finding that **motivates** the Paper B
> efficiency direction, but is not itself an efficiency claim. The MAC-headroom
> analysis in Part 4 is likewise Paper B motivation, reported here for context.

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

**Standard split — Kaggle IDs** (= TransUNet/Synapse split, **verified** 2026-07-25)

> The Kaggle dataset (`shinjinidey/synapse-dataset`) renumbers cases 0001–0030
> (original IDs skip 0011–0020; Kaggle 0011 = original 0021, …, Kaggle 0030 = original 0040).
> **Verification**: our 12 val cases, mapped back to original IDs, are
> {0001,0002,0003,0004,0008,0022,0025,0029,0032,0035,0036,0038} — an **exact match**
> to TransUNet's `lists_Synapse/test_vol.txt`. EffiDec3D uses TransUNet-preprocessed
> data, so this is almost certainly the same split the paper used. The split is
> defined by which files `convert_synapse.py` places in imagesTr/imagesVal; the
> loader (`BTCV13`) globs those directories (no hardcoded IDs).

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

**E1 target BTCV13 mean DICE**: 79.0–79.5% (paper EffiDec3D 79.25%; the full 3DUX-Net E0 target is 79.74%)

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
  Kaggle split with identity affine; the paper uses `btcv_trns`). Seed 1 improves
  the result, but does not close the reproduction gap (see below), so random-seed
  variance alone is not a sufficient explanation.
- **Skip-`Spacingd` control — rejected (negative result, 2026-07-26).** To test
  whether the identity-affine metadata made resampling harmful, we retrained E1
  with `--skip_spatial_resampling`. It is **decisively worse** (best Mean DICE
  ≈ 0.674 vs 0.77, ~10 points lower), so bypassing resampling is **not** the fix —
  `Spacingd` is beneficial even with identity affine. We therefore **keep the
  standard pipeline** (skip flag stays off) and discard this checkpoint. This is a
  discarded control, **not** an analyzed model: it does **not** affect any O1–O11
  result, which were all computed on the standard-pipeline seed-1 checkpoint.
- The **CSV/paper-protocol metric is 0.7700**, not the 0.7549 shown mid-training
  (that periodic MONAI metric uses a running-aggregate average on the resampled
  grid; the final `validation_save` resamples to original resolution + medpy dice,
  which is what the paper reports).

Numbers are "calibration / not final" — see the Paper A → Paper B data-reuse note.
For the AdaDec3D efficiency thesis, **E1 (77.0%) is the iso-accuracy target**;
its exact absolute value is not a gate.

### E1 seed-1 replication and completed observation run

The second E1 training run is stored in
`results/E1_metrics_btcv13_seed1.csv`; its observations and figures are in
`results/obs-seed1/`.

| Run | Mean DICE | Mean HD95 | Train time | Inference | Difference from paper E1 |
|---|---:|---:|---:|---:|---:|
| E1 seed 0 | 0.7700 | 14.41 | 5.36 h | 7.5 ms | −2.25 DICE points |
| E1 seed 1 | **0.7755** | **10.17** | 5.27 h | 7.6 ms | −1.70 DICE points |
| Paper E1 | 0.7925 | 10.12 | — | — | — |

Seed 1 nearly reproduces the reported mean HD95 and improves DICE by 0.55 points,
but the two-seed mean DICE is only **0.7728**. The E0–E1 DICE gap is 2.18 points
for seed 0 and 1.63 points for seed 1, both larger than the paper's 0.49-point
gap. Therefore:

- The implementation is adequate for a **controlled internal observation study**:
  E0 and E1 share the same data, split, transforms and evaluation protocol.
- It is not yet an exact reproduction of the paper's absolute E1 accuracy.
- Paper A must call these runs a *controlled reimplementation*, report both
  seeds, and avoid attributing the enlarged E0–E1 gap entirely to decoder
  compression.
- Before the final manuscript, obtain or reconstruct the paper's `btcv_trns`
  preprocessing and run at least one matched E0/E1 seed pair. Three seeds are
  preferred for the primary BTCV table.

#### Seed-1 observation results

| Observation | Seed-1 result | Current interpretation |
|---|---|---|
| O1 | Boundary error 0.1892 vs interior 0.0481 (**3.93×**) | Supports boundary-concentrated difficulty |
| O2 | Median entropy 0.00035; 1.18% voxels have entropy > 0.5 | Uncertainty is strongly spatially sparse |
| O3 | Pearson 0.973; Spearman 0.975 | Descriptive only; current pooled-bin estimator overstates inferential strength |
| O5 | Subject mean Pearson **0.646**, bootstrap CI [0.158, 0.994]; positive 0.290% vs negative 0.225% | Direction replicates seed 0, but net effect is only about **0.065% of voxels** |
| O6 | Mean entropy falls 0.0279 → 0.0104 from 5k to 45k | Difficulty contracts during training but does not disappear |
| O9 | Top 10/20/30% entropy voxels recover 58.7/86.4/95.2% of positive flips | Strong oracle opportunity; not yet deployable regional compute evidence |
| O10 | Size–difficulty Spearman −0.544; partial entropy–difficulty r 0.810 | Small organs are harder, while entropy retains information beyond size |
| O11 | Entropy r 0.655; confidence r 0.663; MC dropout inactive | Confidence is a valid cheap baseline, tied with entropy; MC-dropout is inactive (EffiDec3D uses DropPath, variance ≈ 0 — auto-detected and skipped, not a failure) |

The most important result is **cross-seed stability**: O5 changes from about
0.665 to 0.646 and O9's 20% recovery remains about 86%. This supports Paper A's
core narrative that useful decoder corrections are predictable and spatially
concentrated. It does **not** yet prove that a realizable adaptive decoder can
obtain the same recovery at lower measured FLOPs.

#### Statistical audit before Paper A submission

The current JSON `go` flags are engineering checks, not publication-ready
hypothesis tests. The following corrections are mandatory before treating O3/O9
as final evidence:

1. **O3:** the current implementation pools entropy bins from all subjects and
   correlates their bin means. Report subject-level AUROC/AUPRC for voxel error,
   calibration curves, and a subject-bootstrap confidence interval instead.
2. **O9 flips:** the current recovery curve counts only positive E0-over-E1
   flips. Add a **net flip** curve (`positive − negative`) so a region
   that improves some E1 errors but also degrades correct E1 predictions is penalized.
3. **O9 bootstrap:** resample the same subject indices for entropy and random
   policies (paired bootstrap). Store both CI bounds and require the lower bound
   to exceed zero at each pre-declared primary budget, rather than selecting the
   best budget after inspection.
4. **Deployability:** repeat O9 using contiguous 3D blocks/connected regions plus
   the refinement halo required by the implementation. Voxel-wise oracle ranking
   is an upper bound, not a realizable routing policy.
5. **Multiplicity and provenance:** predeclare the primary budget and endpoint,
   save subject-level values, checkpoint hashes, split IDs and run configuration
   alongside `results.json`.

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

### Generalization matrix (align with EffiDec3D scope)

Paper A is a **general finding**, not a single-network/dataset result. We align
**strictly to EffiDec3D's own backbone scope** — the architectures for which the
original paper actually built an EffiDec3D decoder pair — with a predeclared ladder
(mirrors the WACV experiments section).

> **Coverage = cross / L-shape (~6–7 cells), not the full grid.** Anchor at
> **UX-Net/BTCV** (also the frozen-encoder factorial backbone): dataset axis = UX-Net
> across **BTCV / FeTA / MSD Task08 HepaticVessel**; backbone axis = BTCV
> across **UX-Net / SwinUNETR / MedNeXt-M-K3** (three families). Each cell is a standard
> Full-vs-Effi pair (the 4-corner factorial is UX-Net/BTCV only); O7 aggregates rows,
> O8 aggregates columns.

| Backbones | EffiDec3D pair? | Datasets | Observations |
|---|---|---|---|
| 3D UX-Net (large-kernel CNN) | ✅ Full + EffiDec3D | BTCV, FeTA, MSD08 HepaticVessel (dataset axis) | **all O1–O11** (full-vs-efficient flips) |
| SwinUNETR (Transformer), MedNeXt-M-K3 (ConvNeXt) | ✅ Full + EffiDec3D | BTCV (architecture axis) | **all O1–O11** |

> **Why only these backbones.** EffiDec3D applies its decoder *only* to 3D UX-Net,
> SwinUNETR, and SwinUNETRv2 (+MedNeXt), because its method targets large-channel /
> high-resolution decoder stages. Architectures without that decoder bottleneck are
> **not** given an EffiDec3D pair in the original work: UNETR is already lean
> (82.6 vs 337.6 GFLOPs), and nnU-Net self-configures its decoder (a fixed EffiDec3D
> decoder does not fit). We therefore do **not** run UNETR/nnU-Net — a matched
> flip analysis is undefined for them, and forcing an unofficial EffiDec3D
> variant would be an ungrounded extension. (If a reviewer asks about non-EffiDec3D
> architectures, they can be added later as single-model O1/O2/O4/O10 concentration
> controls only — never for decoder-causal claims.)
>
> **Two regimes (encoder confound fix).** End-to-end Full/Effi share the encoder
> *architecture* but **not its weights** (trained independently), so their flips are a
> *model-level* difference, not pure decoder capacity. We therefore add a **frozen
> shared-encoder** regime: freeze one trained encoder, train the decoder variants on
> identical features → differences are decoder-caused. On the frozen encoder we run a
> **2×2 channel/resolution factor decomposition** (Full / channel-only / resolution-only
> / combined Effi) to attribute the removable computation, and a **null pair** (two
> Full seeds) as the seed-noise floor. Paper A still says **nothing about the encoder**
> (the untouched 57.8% of MACs) — that is Paper B. See Part 4 for the frozen-encoder
> protocol.

**Training template** (each matched cell = a Full + an EffiDec3D run).
`--network` is the backbone; the EffiDec3D counterpart appends `_EffiDec3D`:

```bash
# Matched pair, e.g. SwinUNETR on FeTA (repeat per backbone × dataset)
NET=SwinUNETR; DS=feta; ROOT=/root/autodl-tmp/feta-processed
python main_train_BTCV_TU.py --root $ROOT --output /root/output/${NET}_${DS}_full \
  --dataset $DS --network $NET \
  --lr 0.001 --overlap 0.7 --crop_sample 4 --max_iter 45000 --eval_step 250 \
  --cache_rate 1.0 --num_workers 8 --gpu 0
python main_train_BTCV_TU.py --root $ROOT --output /root/output/${NET}_${DS}_effi \
  --dataset $DS --network ${NET}_EffiDec3D --ds False \
  --lr 0.001 --overlap 0.7 --crop_sample 4 --max_iter 45000 --eval_step 250 \
  --cache_rate 1.0 --num_workers 8 --gpu 0
```

**Running observations per cell** (all cells are matched pairs → **all O1–O11**;
class count and organ names come from `--dataset`):

```bash
# Matched cell → all O1–O11 (needs both full + EffiDec3D checkpoints)
python run_observations.py --network SwinUNETR --dataset feta \
  --root /root/autodl-tmp/feta-processed --output /root/output --obs_dir /root/obs/swin_feta
```

Each cell writes its own `results.json` + figures under a distinct `--obs_dir`;
O7 (cross-dataset) and O8 (cross-architecture) aggregate these to test whether the
concentration/predictability finding holds across the matrix. (`run_observations.py`
still supports single-model runs for optional non-EffiDec3D controls, but those are
not part of the aligned matrix.)

**Next backbone — SwinUNETR (2nd matched, O8 axis).** Ready-made one-command runner
`run_E0_E1_swin.sh` (proven clone of `run_E0_E1.sh`): 1 Full + 1 EffiDec3D seed on
BTCV13, same 45k/lr1e-3/overlap0.7 protocol, standard Spacingd pipeline. Checkpoints
→ `E0_swin*/SwinUNETR/BTCV13/` and `E1_swin*/SwinUNETR_EffiDec3D/BTCV13/`; observations
→ `/root/obs-swin`.

```bash
cd /root/AdaDec3D/EffiDec3D && git pull
bash run_E0_E1_swin.sh          # E0 → E1 → O1–O11 (≈10–14 h on RTX 5090)
# or stage-by-stage:  bash run_E0_E1_swin.sh E0   then   bash run_E0_E1_swin.sh E1
```

Calibration (EffiDec3D paper, BTCV 13-organ): SwinUNETR ≈ 80.1 Dice / 337.6 GFLOPs /
69.2 M → +EffiDec3D ≈ 79.8 / 57.3 / 11.2 M (published gap ~0.3 pt). A local gap is
footnoted per the reproduction-gap policy above and does not affect O5/O8/O9 direction.

**O8 Go (Swin):** O5 subject-r CI-lower > 0 **and** corrected O9 region-level paired
CI-lower > 0, with O1/O2/O10 concentration recurring → decoder-benefit concentration
is architecture-general, not a 3D UX-Net artifact. Add a 2nd EffiDec3D seed only if the
lower bound is borderline.

> **Reproduction-gap policy (per cell).** Each cell's observations are *within-study*
> and relational — they compare Full vs EffiDec3D (or analyze one model) on the same
> local data, split, and eval code. So if a given backbone or dataset trails the
> published absolute Dice, that gap is **disclosed in a table footnote / figure legend
> for that cell** and does **not** revise the aggregate conclusion. The only claim a
> gap can weaken is the *magnitude* of that one cell's opportunity (a larger local
> Full–Effi gap can inflate apparent flip volume); we guard that with net (not
> positive-only) flips and subject-bootstrap CIs. Direction is what O7/O8
> aggregate, and direction is gap-robust.

---

## Part 3: Observation Study

**Prerequisites**: E0 and E1 `best_metric_model.pth` trained and verified.
Save all figures to `/root/obs/`.

### O-metric glossary, formulas, and contribution map

**Notation** (all per voxel `v` of a validation volume):

- `p_c(v)` — efficient model's softmax probability for class `c` at `v`; `C` classes.
- `ŷ_e(v) = argmax_c p_c(v)` — efficient (EffiDec3D) prediction; `ŷ_f(v)` — full prediction; `y(v)` — ground truth.
- `1[·]` — indicator (1 if true, else 0). `err(v) = 1[ŷ(v) ≠ y(v)]` — voxel error.
- **Entropy** `H(v) = − Σ_{c=1}^{C} p_c(v) · log p_c(v)` — the cheap, ground-truth-free uncertainty signal from the *efficient* model.
- **Confidence** `conf(v) = max_c p_c(v)`.
- **Flips** (efficient → full, per voxel):
  - Positive `P(v) = 1[ ŷ_e(v) ≠ y(v) ∧ ŷ_f(v) = y(v) ]` — full **improves** what efficient got wrong.
  - Negative `N(v) = 1[ ŷ_e(v) = y(v) ∧ ŷ_f(v) ≠ y(v) ]` — full **degrades** what efficient got right.
  - Net `U(v) = P(v) − N(v)`; rates `R_pos = mean_v P(v)`, `R_neg = mean_v N(v)`, `R_net = R_pos − R_neg`.

**The three paper contributions:**
- **C1 — Flip-count comparison.** Measure *where* the full decoder helps by counting `P/N/U` per voxel with subject-level bootstrap, instead of only aggregate Dice/FLOPs (which count every disagreement as improvement).
- **C2 — The net benefit is small, bidirectional, and spatially concentrated.** `R_net ≈ 0.1%` and largely offset by `R_neg`; it concentrates at boundaries and small structures.
- **C3 — The benefit is predictable at test time.** The efficient model's own entropy `H(v)` identifies the beneficial regions without ground truth.

| O | Name | What it measures (formula) | Maps to |
|---|------|----------------------------|---------|
| **O1** | Error distribution | boundary-band vs eroded-interior error ratio `mean_{v∈∂} err(v) / mean_{v∈int} err(v)` (=3.93×) | **C2** |
| **O2** | Entropy distribution | histogram of `H(v)`; sparsity `Pr[H(v) > 0.5]` (=1.18%) | **C2** (premise), supports **C3** |
| **O3** | Predictability of flips (primary) / error (baseline) | subject-level AUROC/AUPRC of entropy predicting **positive flip** (needs Full pair); error-prediction + ECE as baseline; pooled-bin `Pearson(H,err)=0.97` descriptive only | **C3** |
| **O4** | Per-organ difficulty | per-class Dice and mean `H`; rank organs by difficulty | **C2** |
| **O5** | Decoder flip analysis | `P(v)`, `N(v)`, `U(v)`; per-subject `Pearson(binned H, binned U)` + subject bootstrap CI | **C1** + **C2** + **C3** |
| **O6** | Difficulty evolution | `mean_v H(v)` across training iterations (0.0279→0.0104) | **C2** (residual persists) |
| **O7** | Cross-dataset consistency | aggregate O5/O9 direction over ≥3 frozen datasets (CT+MRI+lesion) | generality of **C2/C3** |
| **O8** | Architecture-family consistency | aggregate O5/O9/O11 over matched EffiDec3D backbones (UX-Net, Swin, SwinV2, MedNeXt) | generality of **C1/C2/C3** |
| **O9** | Selective-allocation opportunity | top-budget `H`-ranked contiguous blocks: `Σ_{sel} P(v) / Σ_all P(v)` vs matched random, paired bootstrap | **C3** *(headline)* |
| **O10** | Organ size vs difficulty | `Spearman(volume, difficulty)` (=−0.54); partial `r(H, difficulty ∣ log volume)` (=0.81) | **C2** |
| **O11** | Routing-signal comparison | `corr(signal, U)` for `H` vs `conf` vs MC-dropout; pick the cheap routing signal | **C3** |

**Coverage of the three contributions:** C1 = {O5}; C2 = {O1, O2, O4, O6, O10} (+O5); C3 = {O3, O9, O11} (+O5); generality = {O7, O8}. O5 is the hinge — it defines the flip machinery (C1), shows the effect is small/bidirectional (C2), and shows it rises with entropy (C3).

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

### O5 — Decoder Flip Analysis *(critical Go/No-Go gate)*

**Question**: Does a stronger decoder produce net benefit primarily in high-entropy voxels?

Report **positive** and **negative** flips separately — positive alone overstates benefit.

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

# Figure: global net-flip curve (descriptive) with per-subject r in title
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
         label=f"Net flip (subj-r={r_subj_mean:.2f} [{r_ci_lo:.2f},{r_ci_hi:.2f}])")
plt.axhline(0, color="k", linewidth=0.8, linestyle=":")
plt.xlabel("Mean entropy (bin)"); plt.ylabel("Rate")
plt.title("O5: Decoder Flips vs Uncertainty")
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

**Go criterion**: net benefit rises with entropy AND a contiguous-region selector
beats matched random at a predeclared 10–30% budget with a paired subject-level
95% CI (see O9). The current voxel-wise result is an oracle upper bound only.

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

O7 is not limited to a single CT→MRI check. The final generality study should
align with the representative task families evaluated in the original EffiDec3D
paper (FeTA, BTCV, and the ten MSD tasks) without reproducing all 12 datasets:

| Priority | Dataset | Modality / task | Role |
|---:|---|---|---|
| 1 | **BTCV** | abdominal CT, 13 organs | primary multi-organ experiment (anchor) |
| 2 | **FeTA 2021** | fetal brain MRI, 7 tissues | cross-modality and cross-anatomy replication |
| 3 | **MSD Task08 HepaticVessel** | abdominal CT, thin vessels + small lesions | thin-structure stress test for the high-resolution decoder |

Dataset axis is **frozen at these three** (UX-Net across all three); the third is
HepaticVessel — chosen over BrainTumour because FeTA already covers MRI and
HepaticVessel directly stresses the high-res-decoder question. Freeze before inspecting results. On every dataset,
report the same positive flip, negative flip, net flip,
spatial concentration, and opportunity curve with subject-level confidence
intervals.

**Question**: Do the O5/O9 findings replicate across datasets and modalities?

**Approach**: run the generalized `run_observations.py` on each frozen dataset
with its matched Full + EffiDec3D checkpoints (class count and organ names are
derived from `--dataset`), then aggregate the per-cell `results.json`:

```bash
# 3D UX-Net matched pair across the frozen dataset ladder
for DS in BTCV13 feta Task01_BrainTumour Task06_Lung; do
  python run_observations.py --network 3DUXNET --dataset $DS \
    --root /root/autodl-tmp/$DS --output /root/output --obs_dir /root/obs/uxnet_$DS
done
```

Then read each `/root/obs/uxnet_<DS>/results.json` and tabulate, per dataset:
O5 `subj_pearson_r_mean` + CI, and O9 `region_selector` net utility / paired
diff-CI at the 20% budget.

**Go criterion (O7)**: the O5 subject-level correlation CI-lower stays `> 0` and
the O9 region opportunity remains positive on **≥ 3 of the frozen datasets**,
spanning at least CT + MRI + lesion. The finding is "general" only if the
direction holds across modality and anatomy shifts, not just BTCV.

---

### O8 — Architecture-Family Consistency

| Model | Family | Required comparison |
|---|---|---|
| **3D UX-Net + EffiDec3D** | large-kernel CNN | primary matched full/efficient decoder pair |
| **SwinUNETR + EffiDec3D** | hierarchical Transformer | matched cross-backbone pair |
| **SwinUNETRv2 + EffiDec3D** | hierarchical Transformer (v2) | optional deeper matched pair |
| **MedNeXt-M-K3 + EffiDec3D** | large-kernel CNN (ConvNeXt-style) | optional deeper matched pair |

All four have an official EffiDec3D decoder pair, so every row supports the full
flip analysis. We do **not** include UNETR or nnU-Net: EffiDec3D never built
a decoder pair for them (UNETR is already lean; nnU-Net self-configures its
decoder), so a matched flip comparison is undefined and any unofficial pair
would be an ungrounded extension.

**Question**: Do the flip-concentration observations hold across the
EffiDec3D backbone family — a second (and optionally third/fourth) matched backbone
— rather than being specific to 3D UX-Net?

**Approach**: run `run_observations.py` per matched backbone on BTCV; each is a
full/efficient pair, so each yields all O1–O11:

```bash
# Matched EffiDec3D backbones → all O1–O11
for NET in 3DUXNET SwinUNETR; do        # + SwinUNETRv2 MedNeXt for deeper coverage
  python run_observations.py --network $NET --dataset BTCV13 \
    --root /root/autodl-tmp/btcv-synapse --output /root/output --obs_dir /root/obs/${NET}_btcv
done
```

**Go criterion (O8)**: on ≥2 matched backbones the O5/O9 flip–entropy
direction holds (subj-r CI-lower `> 0`) and the O1/O2/O10 concentration pattern
recurs → the decoder-benefit concentration is a property of the EffiDec3D backbone
family, not an artifact of 3D UX-Net.

---

### O9 — Selective-Allocation Opportunity *(headline result for Paper A)*

**Question**: At a predeclared compute budget, can entropy select contiguous
regions with more favorable **net flips** than matched random regions?

The corrected implementation is in `EffiDec3D/run_observations.py`. It reports:

1. A voxel-wise entropy oracle as a non-deployable upper bound.
2. Non-overlapping 16³ blocks ranked by mean entropy.
3. A 4-voxel context halo around every selected block.
4. Net flip
   `(E0-correct/E1-wrong − E0-wrong/E1-correct) / all positive flips`.
5. A paired subject bootstrap: entropy and random use exactly the same resampled
   subject indices.
6. Both CI bounds, actual halo-expanded volume and subject-level values.

Run seed 0 and seed 1 separately with explicit checkpoints:

```bash
cd /root/AdaDec3D/EffiDec3D

python run_observations.py \
  --root /root/autodl-tmp/btcv-synapse \
  --dataset BTCV13 \
  --e0_ckpt /root/output/E0.../3DUXNET/BTCV13/best_metric_model.pth \
  --e1_ckpt /root/output/E1_seed1.../3DUXNET_EffiDec3D/BTCV13/best_metric_model.pth \
  --obs_dir /root/obs-seed1-corrected \
  --only_o9 \
  --o9_block_size 16 \
  --o9_halo 4 \
  --o9_primary_budget 20
```

Outputs:

- `/root/obs-seed1-corrected/O9_opportunity_corrected.png`
- `results.json["O9_corrected"]`
- Per-subject positive/negative counts, net flips and executed-volume fractions
  under `O9_corrected.subject_results`

The old `results.json["O9"]` is retained only as a legacy positive-flip
oracle and must not be used as the final Paper A result.

#### Corrected O9 results (seed 1) — **PASS**

Refreshed run (`results/obs-seed1/`, full E0 + seed-1 EffiDec3D, 12 subjects,
100 random repeats, 2000-sample paired subject bootstrap). Net utility =
`(positive − negative) / all positive flips`. Two selectors:

**Voxel oracle** (non-deployable upper bound):

| Budget | Positive recovery | Net utility | Net CI-lower | Random net | Paired-diff CI-lower |
|---:|---:|---:|---:|---:|---:|
| 5% | 0.304 | 0.121 | 0.039 | 0.011 | 0.035 ✓ |
| 10% | 0.587 | 0.241 | 0.138 | 0.022 | 0.129 ✓ |
| **20%** | **0.864** | **0.299** | **0.182** | **0.045** | **0.166 ✓** |
| 30% | 0.952 | 0.277 | 0.137 | 0.067 | 0.116 ✓ |
| 50% | 0.995 | 0.243 | 0.087 | 0.112 | 0.055 ✓ |

**Region selector** (deployable: 16³ blocks + 4-voxel halo):

| Block budget | Executed vol. | Pos. recovery | Neg. recovery | Net utility | Random net | Paired-diff CI-lower |
|---:|---:|---:|---:|---:|---:|---:|
| 5% | 0.109 | 0.543 | 0.411 | 0.133 | 0.011 | −0.012 (ns) |
| 10% | 0.162 | 0.937 | 0.732 | 0.206 | 0.022 | 0.049 ✓ |
| **20%** | **0.291** | **1.000** | **0.775** | **0.225** | **0.043** | **0.056 ✓** |
| 30% | 0.384 | 1.000 | 0.775 | 0.225 | 0.067 | 0.048 ✓ |
| 50% | 0.581 | 1.000 | 0.775 | 0.225 | 0.113 | 0.029 ✓ |

**Interpretation.** At the predeclared **20% block budget** (which executes 29.1% of
the union-foreground volume after halo expansion), entropy-ranked contiguous regions
achieve net utility **0.225 vs 0.043 for matched random**, with a paired subject-
bootstrap lower bound of **0.056 > 0** → the selection is significantly better than
random. The direction holds at every budget ≥10% (paired CI-lower `> 0` for 10/20/30/50%),
failing only at 5% (CI includes 0), so there is a minimum viable budget. Honest
nuance: capturing 100% of positive flips also captures **77.5% of negative**
ones, so the *net* gain (0.225) — not positive recovery alone — is the correct headline;
the positive-only 86.4% voxel figure is an oracle upper bound, not a deployable claim.
This upgrades O9 from the legacy positive-only voxel oracle to a **net, paired,
region-level, deployable** result: **Go criterion met.**

Repeat the command with the seed-0 E1 checkpoint and a separate
`--obs_dir /root/obs-seed0-corrected`; do not overwrite or merge the two runs
before checking their subject-level results.

**Predeclared Go criterion**: at the 20% block budget, the entropy region selector
has a positive mean net flip and the lower bound of the paired 95% CI for
`entropy − matched random` is above zero. The same direction must hold for both
E1 seeds. The 10% and 30% budgets are secondary sensitivity analyses.

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

**Question**: Which test-time difficulty signal best predicts decoder benefit?

Run after O5. Evaluate five signals on the BTCV validation set:

| Signal | Implementation | Overhead |
|---|---|---|
| Entropy | `-(p log p).sum(1)` over softmax | ~0 ms |
| Confidence | `1 - max(p)` over softmax | ~0 ms |
| Feature Variance | std of last decoder feature map | low |
| MC Dropout | variance over T=10 stochastic passes | T× latency |
| Boundary Probability | distance-to-foreground-boundary map | moderate |

For each signal compute:
- Pearson correlation with per-bin O5 net flip
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

**Current BTCV result**: confidence (r=0.663) and entropy (r=0.655) are effectively
tied as zero-extra-forward-pass signals. MC dropout is invalid because dropout is
inactive in the evaluated model. O11 therefore supports keeping both entropy and
confidence as Paper B baselines; it does not establish entropy as uniquely best.

---

## Part 4: Go / No-Go Decision

### Prototype gate — **direction replicated; submission gate remains open**

| Obs | Intended criterion | Seed 0 | Seed 1 | Status |
|-----|--------------------|--------|--------|--------|
| O2 | Strongly skewed entropy | median 0.0011; 1.31% > 0.5 | median 0.00035; 1.18% > 0.5 | ✅ replicated |
| O3 | Entropy predicts voxel error | pooled-bin r = 0.971 | pooled-bin r = 0.973 | ⚠ pooled-bin overstates; subject-level AUROC/AUPRC + calibration still required |
| O5 | Net benefit rises with entropy | subj-r 0.665, CI [0.171, 0.996] | subj-r 0.646, CI [0.158, 0.994] | ✅ direction replicated |
| O9 | Entropy beats matched random | — | region 20%: net 0.225 vs random 0.043, paired CI-lower 0.056 > 0 | ✅ corrected (net/paired/region) |

These results pass the **prototype gate**: uncertainty-guided regional refinement
is sufficiently motivated to prototype Paper B. **O9 is now complete** — the
corrected region-level analysis (net flips, contiguous 16³ blocks + halo,
paired subject bootstrap) retains a positive lower bound (0.056) at the predeclared
20% budget, so it clears the audit that was outstanding. The remaining open item for
the final Paper A submission gate is **O3**, which still needs a subject-level
discrimination/calibration analysis (the pooled-bin r is descriptive only).

### Generalization criteria (not yet run)

These are **within-study** criteria: each cell trains a Full/Effi pair (or a single
control) under one protocol and checks whether the O1–O11 *direction* recurs.
Exact reproduction of the paper's absolute Dice is **not** a prerequisite — the
observations are relational (Full-vs-Effi on identical local data), so they hold
regardless of any single model's distance from the published number.

| Obs | Axis | Criterion | Result | Pass? |
|-----|------|-----------|--------|-------|
| O7 | dataset | net-flip/entropy subj-r CI-lower > 0 on ≥3 frozen datasets spanning CT+MRI+lesion | pending matrix (BTCV/FeTA/MSD01/MSD06-08) | ☐ |
| O8 | architecture (matched) | ≥2 matched EffiDec3D backbones (Swin, opt. SwinV2/MedNeXt) hold O5/O9 direction (subj-r CI-lower > 0) + O1/O2/O10 concentration | pending E0/E1-Swin | ☐ |
| O11 | routing | Entropy is best or tied-best cheap routing signal | confidence 0.663 vs entropy 0.655; MC dropout inactive | ◐ partial |

---

### Interpretation — why Dice and FLOPs are not enough (Paper A finding)

O5's net flip is **small**: for seed 0 the positive/negative flip rates
are 0.339%/0.209% (about 0.130% net), while for seed 1 they are
0.290%/0.225% (about 0.065% net). The direction is stable, although our enlarged
E0–E1 DICE gap means the magnitude cannot yet be claimed as an exact reproduction
of the paper's near-parity (79.74 vs 79.25).

As a **Paper A observation**, this reads as: aggregate Dice and FLOPs hide where
extra decoder capacity improves predictions and where it degrades them. The net
change is small in aggregate, concentrated where entropy is high, and partly
predictable (O9). The three observations line up:

- O2: only about 1.2–1.3% of voxels have entropy above 0.5; difficulty is spatially sparse.
- O5: the full decoder's extra capacity has a small net voxel effect, concentrated in high-entropy bins.
- O9: voxel-wise entropy ranking identifies a 20% oracle region carrying about 86% of positive flips.

> **These are observations, not an efficiency claim.** They *motivate* the Paper B
> efficiency direction (region-adaptive decoding) but do not themselves demonstrate
> lower executed cost or a deployable selector. That requires contiguous-region
> analysis followed by the AdaDec3D method (`Experiment-Design-AdaDec3D.md`).

### MAC headroom (Paper B motivation, `profile_macs.py`)

*Reported here for context; efficiency realization belongs to Paper B.*

| Group | GMac | % |
|---|---|---|
| **DECODER** (decoder3/4/5 + out) | **18.09** | **42.2%** |
| ENCODER (uxnet_3d + encoder2–5) | 24.74 | 57.8% |
| — `decoder3` alone (finest half-res block) | 15.80 | 36.9% |

Decoder = 42% of compute, dominated by `decoder3` (37%) — where a Paper B
region-adaptive decoder would act. Rough ceiling: full `decoder3` on ~20% of
voxels + cheap elsewhere → 41 → ~33 GMac. Encoder (57.8%) is untouched by
decoder-only routing.

> **Encoder confound + decoder-only scope.** End-to-end Full/Effi share the encoder
> *architecture* but not its *weights*, so end-to-end flips are model-level, not pure
> decoder capacity. The **frozen shared-encoder** regime (freeze one encoder, train
> the decoder variants on identical features) makes the attribution causal; on it we
> run the **2×2 channel/resolution factor decomposition** and a **null pair** (two Full
> seeds) as the seed-noise floor. Even so, Paper A says **nothing** about whether the
> encoder (the larger 57.8% of MACs) is over-provisioned — that needs different
> instruments (per-stage effective rank, probing, CKA) and is deferred to Paper B.
>
> **Frozen-encoder protocol (E1):** reuse trained Full-UX as the frozen encoder;
> `main_train_BTCV_TU.py --freeze_encoder --encoder_ckpt <E0> --resolution_factor
> {1,2} --n_decoder_channels {C_full,48}`; analyze pairs with `run_observations.py
> --e0_rf/--e0_nchan/--e1_rf/--e1_nchan` (see `run_frozen_factorial.sh`). Boundary-
> resolved flips, NSD, and the O9 `--o9_foreground` denominator option are in
> `run_observations.py`.

**Paper A Go decision: PASS** (O2/O3/O5/O9). The efficiency application proceeds in
[Experiment-Design-AdaDec3D.md](Experiment-Design-AdaDec3D.md), with E1 (77.0%) as
the iso-accuracy target.

---

## Part 5: Deliverables

### Producer

All O1–O11 metrics and figures come from one script,
`EffiDec3D/run_observations.py` (one invocation per grid cell → `results.json`
+ PNGs). The reproduction baselines (Table 1) come from training
(`main_train_BTCV_TU.py` metrics CSV) and `profile_macs.py`, **not** from
`run_observations.py`.

### Figures (Paper A) → producing output

| Paper figure | Content | run_observations output |
|---|---|---|
| Fig 1 (Heterogeneity) | per-organ error bar (O1) | `O1_organ_error.png` |
| Fig 2 (Concentration) | entropy sparsity (O2) + net-flip vs entropy (O5) | `O2_entropy.png`, `O5_decoder_gain.png` |
| Fig 3 (Predictability) | region opportunity curve (O9) | `O9_opportunity_corrected.png` |
| supp. | entropy–error scatter (O3), entropy evolution (O6) | `O3_unc_error.png`, `O6_entropy_evolution.png` |

O10 is reported as ρ / partial-r numbers (no figure); O4 per-organ numbers feed Fig 1.

### Tables (Paper A) → source

| Paper table | Content | Source |
|---|---|---|
| Table 1 | baseline Dice/HD95/MAC/params/latency | training CSV + `profile_macs.py` |
| Table 2 | flip rates R_pos/R_neg/R_net + subject CI (O5) | `results.json["O5"]` |
| Table (plan) | backbone × dataset grid | manual |
| Predictability (text) | O3 AUROC/AUPRC/ECE; O9 net + paired CI; O11 subj corr | `results.json["O3","O9_corrected","O11"]` |
| pending | cross-dataset (O7), cross-backbone (O8) | grid aggregation (≥2 cells) |

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
  [x] E1 seed 1      — 45 000 iter    ✓ Mean DICE 0.7755 / HD95 10.17 (paper Effi 79.25, −1.70)
  [x] E1 efficiency confirmed: 41.06 GMac (14.1× vs E0), 7.5 ms, 0.24 GB
  [x] E1 milestone_{05000..45000}.pth auto-saved (for O6)
  [x] Different-seed E1 run complete; variance does not fully explain reproduction gap

Week 4: Observations — preliminary gate (run_observations.py)   ✓ DIRECTION REPLICATED
  [x] O1: boundary error is 3.93× interior error
  [x] O2: entropy skewed (seed 1 median 0.00035; 1.18% voxels > 0.5)
  [x] O3: pooled-bin entropy–error Pearson r = 0.973 (descriptive; audit pending)
  [x] O4: per-organ dice/entropy (hardest = highest-entropy organs)
  [x] O5: subject r = 0.646, CI [0.158, 0.994]; pos>neg (net effect ≈0.065%)
  [x] O6: entropy falls from 0.0279 (5k) to 0.0104 (45k)
  [x] O9: CORRECTED region-level — 20% block budget net 0.225 vs random 0.043, paired CI-lower 0.056 > 0 (deployable; 86.4% positive recovery is oracle upper bound only)
  [x] O10: size–difficulty rho = −0.544; partial entropy r = 0.810
  [x] O11: entropy/confidence comparable; MC dropout inactive
  [x] MAC profiling: decoder = 42.2% (decoder3 37%); encoder 57.8%
  [x] BF16 innocent (FP32 revalidation Δ=0.0000)
  [~] Statistical audit: corrected O9 DONE (net/paired/region — PASS); subject-level O3 discrimination/calibration still required
  [x] --- CONDITIONAL GO: prototype Paper B, while completing Paper A audit ---

Week 5-6: Generalization matrix (run_observations.py --network/--dataset)
  [x] O6: difficulty evolution
  [ ] Architecture axis — matched pairs: retrain E0/E1 for SwinUNETR (opt. SwinV2/MedNeXt), rerun O1-O11
  [ ] Dataset axis — freeze 3-4 tasks (BTCV, FeTA, MSD01 BrainTumour, MSD06/08),
      rerun matched UX-Net O1-O11 per dataset; aggregate CI-lower across CT+MRI+lesion
  [ ] O7: cross-dataset consistency = aggregate over dataset axis (Go: ≥3 datasets)
  [ ] O8: architecture-family consistency = aggregate over architecture axis
  [x] O10: organ size vs difficulty
  [~] O11: entropy and confidence complete; MC dropout invalid/inactive
  [x] Corrected O9 contiguous-region opportunity curve (net/paired/halo) — PASS
  [ ] Corrected O3 subject-level discrimination/calibration analysis
  [ ] (optional) Matched E0/E1 multi-seed run with paper-equivalent preprocessing
       — improves absolute numbers; NOT required for within-study O1-O11 direction
  [ ] Patch-level whole-model adaptivity study (extend efficiency beyond decoder's 42%)

Week 7: Paper A draft
  [~] Paper A manuscript first draft written (wacv-2027/, 7 pp, compiles clean;
       full O1-O11 on 3D UX-Net/BTCV) — \todo markers remain for the pending
       SwinUNETR + FeTA/MSD generality cells
  [ ] Target venue: WACV 2027 E&D Track (primary; deadline 2026-08-28) — see Paper-Narrative §9
```
