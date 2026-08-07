# Modifications to the EffiDec3D codebase

This document records every change we made to the EffiDec3D code we built on, so the
observation study is reproducible and so the paper can transparently state what was
altered. Each entry is tagged:

- **[BUG FIX]** — a latent defect in the inherited code (not introduced by us) that
  produced wrong results or crashes once exercised by our protocol.
- **[COMPAT]** — an adaptation for our runtime (PyTorch 2.6 / MONAI 1.5, Blackwell RTX 5090).
- **[FEATURE]** — a new capability we added for the study (not a fix).
- **[CONFIG]** — a deliberate configuration choice for the matched-pair protocol.

**Attribution caveat.** These describe the code *as we received it*. Exact upstream
attribution would require a line-by-line diff against a pristine EffiDec3D release; items
marked **[BUG FIX]** were present in the inherited code and were **not** caused by our own
additions. Line numbers are approximate and may drift as the code evolves.

---

## Bug fixes (affect correctness of results)

### BF1 — `--ds False` was parsed as a truthy string  **[BUG FIX]**
- **File:** `main_train_BTCV_TU.py` (~line 147)
- **Before:**
  ```python
  if args.ds == 'True':
      args.ds = True
  ```
- **Defect:** `--ds` has no argparse `type`/`action`, so `--ds False` arrives as the
  **string** `'False'`. The conversion only handled the `'True'` case, leaving `'False'`
  as a non-empty string — which is **truthy** in Python. Any code doing
  `deep_supervision=args.ds` therefore turned deep supervision **ON** when the caller
  asked for it OFF.
- **Fix:**
  ```python
  if isinstance(args.ds, str):
      args.ds = (args.ds == 'True')
  ```
- **Impact:** Corrupted the **MedNeXt-M-K3 + EffiDec3D** cell (see BF2 for the downstream
  mechanism). Did **not** affect 3D UX-Net or SwinUNETR Effi models — their network
  classes take no `deep_supervision` argument, so the truthy string never reached them.

### BF2 — MedNeXt-Effi trained one output head but deployed another  **[BUG FIX]**
- **File:** `networks/MedNeXt/mednextv1/MedNeXtV1_EffiDec3D.py` (`forward`, both the
  checkpointed and non-checkpointed branches, ~lines 386–456)
- **Defect:** The efficient decoder has two heads — `out_0` (full resolution, the
  expensive high-res stage EffiDec3D is meant to *remove*) and `out_1` (half resolution,
  the efficient head). At **inference** (`mode='test'`) the model returns `out_1` and
  skips `out_0` (this is the efficiency the GMac profile measures). At **training** it
  instead ran through to `out_0`:
  - With deep supervision **on** (the state BF1 wrongly produced): the loss used `out_0`,
    so `out_1` — the head validation/inference actually reads — **never received a
    gradient**. Result: train loss looked healthy (~0.5) while validation Dice collapsed
    to **~0.02**.
  - With deep supervision **off** (correct, after BF1): the non-DS return path was
    `return [x_ds_1]`, but `x_ds_1` was never computed on that path →
    `UnboundLocalError` crash.
- **Fix:** Always compute and return `out_1` for **both** `test` and non-deep-supervision
  `train`, and skip the removed `out_0` stage:
  ```python
  x_ds_1 = self.out_1(x)
  if mode == 'test' or not self.do_ds:
      return [x_ds_1]
  ```
- **Impact:** Restored the MedNeXt-Effi cell (first-validation Dice jumped from ~0.02 to
  ~0.19 and climbs toward ~0.80; the full MedNeXt is 0.837). Also removed wasted `out_0`
  compute during training (~1.4 → ~2.5 it/s). No other backbone was affected: full
  MedNeXt deploys `out_0` (train head == test head), and UX-Net / Swin Effi have a single
  consistent output head.

**Why BF1+BF2 only hit MedNeXt-Effi:** it is the one cell that both (a) takes a
`deep_supervision` argument (exposed to BF1) and (b) has the dual-head train/test split
(BF2). The two defects stacked only there.

---

## Compatibility adaptations (runtime, not results)

### C1 — `mode` keyword only passed to networks that accept it  **[COMPAT]**
- **File:** `main_train_BTCV_TU.py` (~lines 453–457, used at the train call)
- The repo's own networks (UXNET, the EffiDec3D variants, MedNeXt) accept a `mode` kwarg;
  MONAI nets (SwinUNETR, UNETR, …) do not. We detect once via `inspect.signature` and call
  `model(x, mode='train')` only when supported, else `model(x)`. Without this, adding
  SwinUNETR to the architecture axis raised a `forward() got an unexpected keyword
  argument 'mode'`.
- Mixed-precision uses the PyTorch 2.x API (`from torch.amp import GradScaler`,
  `torch.autocast("cuda", dtype=...)`), BF16 on Blackwell.

### C2 — dataset loader accepts `.nii` as well as `.nii.gz`  **[COMPAT]**
- **File:** `load_datasets_transforms.py` (`data_loader` globs for imagesTr/labelsTr/
  imagesVal/labelsVal/imagesTs)
- Upstream globbed only `'*.nii.gz'`. MSD **Task01 BrainTumour** ships **uncompressed
  `.nii`** files, so the loader matched zero files → an empty validation set → `Dice = nan`
  and `Best Avg. Dice` stuck at `0.0`. Changed the five globs to `'*.nii*'` (matches both
  `.nii` and `.nii.gz`; backward-compatible for the `.nii.gz` datasets).

---

## Features added for the observation study

### F1 — Checkpoint resume  **[FEATURE]**
- **File:** `main_train_BTCV_TU.py` (~lines 497–525, 650–660)
- Every eval step atomically writes `last_model.pth` (model/optimizer/scaler/global_step)
  and, on startup, resumes from it (skipping the loop if already at `max_iter`). Corrupt
  or empty checkpoints are detected and removed rather than crashing. Enables overnight
  runs to survive interruption.
- **Note:** a resume only succeeds when the run directory (its full arg-suffix) and the
  model architecture match the saved checkpoint. A resume from a checkpoint saved under a
  *different* config silently falls through to "train from scratch."

### F2 — Frozen shared-encoder training  **[FEATURE]**
- **File:** `main_train_BTCV_TU.py` (`--freeze_encoder`, `--encoder_ckpt`, ~lines 436–451)
- Loads encoder weights from a checkpoint (prefixes `uxnet_3d`/`encoder`/`swinViT`), sets
  `requires_grad=False`, and passes only trainable params to AdamW. Backs the frozen-encoder
  factor-decomposition control (`run_frozen_factorial.sh`).

### F3 — Determinism seed  **[FEATURE]**
- `--seed` (default 0) for the null-pair seed-noise control (Full-s0 vs Full-s1).

### F4 — Identity-affine control  **[FEATURE]**
- `--skip_spatial_resampling` (BTCV13 only) drops `Orientationd`+`Spacingd` to test the
  identity-affine data confound.

### F5 — Milestone checkpoints  **[FEATURE]**
- Saves `milestone_{05000..45000}.pth` for the O6 difficulty-evolution analysis.

### F6 — Skip-aggregation exposed on the CLI  **[FEATURE]**
- `--skip_aggregation {addition, concatenation}` surfaces a previously internal knob and
  threads it to the Effi builds. (MedNeXt-Effi has no such knob and ignores it.)

### F7 — Configurable input channels (multi-modal MRI)  **[FEATURE]**
- **File:** `run_observations.py` (`build_model(..., in_channels=1)`; `main()` derives
  `in_ch = 4 if dataset == "Task01_BrainTumour" else 1`)
- Upstream `build_model` hard-coded `ic = 1` (single-channel CT). Task01 BrainTumour stacks
  **4 MRI modalities** (FLAIR/T1/T1ce/T2), so the obs rebuild needs `in_channels=4` to load
  those checkpoints. Training already supported it via `--n_channels 4`
  (`in_chans=args.n_channels`); this threads the same choice through the observation path.

### F8 — Depthwise-separable decoder (different-logic efficient decoder)  **[FEATURE]**
- **File:** `networks/UXNet_3D/network_backbone.py` (`SeparableConv3d`, `SepUnetBasicBlock`,
  `SeparableUnetrUpBlock`, `UXNET_SepDec`); `main_train_BTCV_TU.py` (`--network 3DUXNET_SepDec`
  branch); `run_observations.py` (`3DUXNET_SEP` in `MATCHED_BACKBONES`/`EFFI_NETWORK`/
  `FULL_NETWORK`, and a `build_model` branch).
- **Why:** reviewer noted the decoder intervention was single-form (only EffiDec3D's
  capacity *removal* — channel reduction + high-resolution stage omission). `UXNET_SepDec`
  is a **different-logic** efficient decoder: it keeps the encoder, all high-resolution
  decoder stages, and every channel width identical to the full UXNET (E0), and factorizes
  each dense decoder convolution into depthwise + pointwise — a *capacity-preserving,
  compute-reducing* mechanism. A dense 192→192 3×3×3 decoder conv drops from 995K to 42K
  parameters (23.7× fewer).
- **How it fits the pipeline:** `UXNET_SepDec` subclasses `UXNET` and only swaps the decoder
  modules, so `forward()`/encoder are untouched and a full-UXNET (E0) checkpoint's encoder
  loads with `strict=False`. The `3DUXNET_SEP` obs key reuses the **existing full 3DUXNET
  checkpoint** as the E0 member (`FULL_NETWORK["3DUXNET_SEP"]="3DUXNET"`), paired against
  the separable E1' (`3DUXNET_SepDec`). Trained end-to-end like the other E1 pairs; single
  output (no deep supervision), mirroring the `3DUXNET` branch.
- **Scope:** anchor cell only (UX-Net / BTCV) — tests whether the characterization findings
  (net-neutral, boundary-localized, direction-unpredictable) hold under a structurally
  different decoder-lightweighting logic, not just EffiDec3D pruning.

---

## Configuration choices for the matched-pair protocol

### CFG1 — MedNeXt kernel size 5 → 3  **[CONFIG]**
- **File:** `main_train_BTCV_TU.py` — `MedNeXt_M` (~line 384) and `MedNeXt_M_EffiDec3D`
  (~line 201) both use `kernel_size=3`.
- Makes the full and efficient models a clean matched pair differing **only** in the
  decoder, matching the paper's MedNeXt-M-K3.

### CFG2 — Skip-aggregation: concatenation is the canonical config  **[CONFIG]**
- `build_model` in `run_observations.py` defaults `skip_aggregation="concatenation"`,
  matching the EffiDec3D network class default. **Concatenation reproduces the paper's
  parameter counts** (UX-Net Effi 3.16M, Swin Effi 11.21M — exact), so it is our canonical
  matched-pair config for UX-Net and Swin; `addition` is retained only as an ablation. The
  rebuilt model must use the same skip as the checkpoint (`--skip_aggregation`). **MedNeXt**
  has no such knob — its decoder is natively additive — so MedNeXt uses `addition`.

### CFG3 — Task01 BrainTumour: single-label 4-class softmax, NOT BraTS multi-label  **[CONFIG]**
- **File:** `load_datasets_transforms.py` (`Task01_BrainTumour` branch) + `run_observations.py`
  (`BRAIN_NAMES`)
- **Upstream (= EffiDec3D paper's setup):** `out_classes = 3 # for sigmoid` +
  `ConvertToMultiChannelBasedOnBratsClassesd` → three **overlapping** BraTS regions
  (WT ⊃ TC ⊃ ET), multi-label **sigmoid**. This is the standard brain-tumour evaluation the
  paper reports (Table 3, ~78% average, where the large easy WT region dominates).
- **Change:** `out_classes` 3 → 4; **drop** `ConvertToMultiChannelBasedOnBratsClassesd`
  (train + val); add the label to `EnsureChannelFirstd(keys=["image","label"])` → **single-label
  softmax** over the raw atomic sub-regions (bg + edema=1 + enhancing=2 + NCR/necrotic=3).
  Registered class names `["Edema","Enh","NCR"]`. Train with `--n_channels 4`.
- **Why:** our entire analysis is per-voxel prediction **flips** (`P/N/U`, H1/H2/C1/P1/P2/R),
  which require a single **`argmax`** class per voxel. BraTS's overlapping regions have **no
  `argmax`** (a voxel can be WT *and* TC *and* ET), so the flip framework cannot be computed
  on the multi-label setup. The atomic sub-regions are single-label and argmax-compatible,
  and consistent with our other cells (BTCV organs, Task08 vessel/tumour are mutually
  exclusive).
- **Consequence:** our Task01 Dice (~0.66, per-subregion) is **not comparable** to the paper's
  WT/TC/ET average (~0.786) — a *different quantity*, not worse performance. Footnoted in the
  paper. The flip analysis is relational (Full vs Effi on identical data), so absolute Dice is
  not a gate. (A WT/TC/ET Dice can be derived post-hoc from the `argmax` for a paper-comparable
  baseline row if desired, without changing the flip pipeline.)

### CFG4 — Task08 HepaticVessel: `Spacingd` resampling enabled  **[CONFIG]**
- **File:** `load_datasets_transforms.py` (`Task08_HepaticVessel` branch, train/val/test)
- **Upstream:** `Spacingd` was **commented out** for Task08 in the inherited code (base commit
  `d5a71e6`), i.e. no spacing normalization — the config behind EffiDec3D's own Task08 result
  (~0.557). (Note: BTCV's `Spacingd` is active but a *nominal* no-op on our identity-affine
  data; Task01's is active upstream.)
- **Change:** uncommented `Spacingd(pixdim=(1.0,1.0,1.0))` for train/val/test.
- **Why:** HepaticVessel CT has highly variable slice thickness (~0.8–8 mm); without spacing
  normalization thin vessels sit on inconsistent voxel scales and are ~unlearnable (Dice
  ~0.54). Resampling is standard (nnU-Net) and aligns Task08 with the real-affine MSD data.
  Lifts Dice to ~0.575/0.586 (paper-comparable to their 0.557).
- **Consequence:** deviates from EffiDec3D's released Task08 preprocessing, so our absolute
  Task08 Dice is not a strict reproduction of their Table 3. Footnoted; the relational flip
  analysis is unaffected.

### CFG5 — Dataset axis: BTCV + Task08 + Task01 (FeTA dropped)  **[CONFIG]**
- The generality dataset axis is BTCV (CT, main) / MSD Task08 HepaticVessel (CT, thin
  structures) / MSD Task01 BrainTumour (MRI, heterogeneous lesions). **FeTA 2021 was dropped**
  (could not be obtained) and Task01 replaces it for the MRI + lesion coverage.

---

## New analysis code (not modifications to training)

These are new files, listed for completeness; they do not alter EffiDec3D training/inference:

- `run_observations.py` — the **six converged metrics** along Heterogeneity → Concentration
  → Predictability → Recoverability: H1 global flips (`O5`), H2 boundary/anatomy net
  (`H2_boundary`, `O_anatomy`), C1/C2 oracle concentration + P2 signal-vs-random (`O_pareto`),
  P1 any/positive/direction AUROC+AUPRC (`O3`), R hybrid selective-decoding Dice recovery
  (`R_recovery`), plus Activity=P+N (in `O5`) and an opt-in TTA-uncertainty diagnostic
  (`--tta`, `O_tta`). The exploratory O1/O2/O4/O9 survive behind `--appendix`; O6/O10/O11,
  O_dice_aware and O_errortype were deleted. Arbitrary-pair (frozen-encoder factorial) support.
- `make_figure1.py` — Figure 1 teaser (`--skip_aggregation`, `--zoom_half`).
- `aggregate_generality.py` — Figure 4 (cross-dataset / cross-architecture aggregation).
- `run_E0_E1.sh`, `run_E0_E1_swin.sh`, `run_E0_E1_mednext.sh`, `run_frozen_factorial.sh`
  — training + observation runners.

---

## Known non-issues

- **MC-Dropout was removed, not fixed.** These backbones use dropout = 0, so MC-Dropout
  predictive variance is degenerate; the old O11 routing-signal comparison (and MC-dropout)
  were **deleted**. The richer-uncertainty question is instead answered by the opt-in
  **TTA mutual-information** diagnostic (`O_tta`), which needs no dropout. Only entropy and
  confidence are used as the default deployable signals.
- **`R_recovery.recovered_fraction_at_20pct = null`** when Full Dice ≤ Effi Dice (e.g. Task08):
  expected. The fraction `(hybrid−Effi)/(Full−Effi)` is undefined when the full decoder does
  not improve Dice; the code emits `null` + a note and the raw `dice_mean` curve is the source
  of truth.
- **Non-tuple indexing `UserWarning`s** from `monai_utils/inferers/utils.py` on PyTorch 2.6
  — harmless deprecation notices, not errors.
