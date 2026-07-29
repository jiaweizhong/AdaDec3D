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

---

## Configuration choices for the matched-pair protocol

### CFG1 — MedNeXt kernel size 5 → 3  **[CONFIG]**
- **File:** `main_train_BTCV_TU.py` — `MedNeXt_M` (~line 384) and `MedNeXt_M_EffiDec3D`
  (~line 201) both use `kernel_size=3`.
- Makes the full and efficient models a clean matched pair differing **only** in the
  decoder, matching the paper's MedNeXt-M-K3.

### CFG2 — Skip-aggregation default = concatenation  **[CONFIG]**
- `build_model` in `run_observations.py` defaults `skip_aggregation="concatenation"`,
  matching the EffiDec3D network class default (the training CLI historically defaulted to
  `addition`). Our currently-analyzed UX-Net/Swin Effi checkpoints were trained with
  `addition`; observations on them must pass `--skip_aggregation addition` so the rebuilt
  model matches the checkpoint.

---

## New analysis code (not modifications to training)

These are new files, listed for completeness; they do not alter EffiDec3D training/inference:

- `run_observations.py` — O1–O11 + O_pareto / O_anatomy / O_boundary_flips / O_surface,
  boundary-resolved flips, surface metrics (NSD), O3 flip-prediction, O9 foreground
  denominator, arbitrary-pair (factorial) support.
- `make_figure1.py` — Figure 1 teaser (now with `--skip_aggregation`).
- `aggregate_generality.py` — Figure 4 (cross-dataset / cross-architecture aggregation).
- `run_E0_E1.sh`, `run_E0_E1_swin.sh`, `run_E0_E1_mednext.sh`, `run_frozen_factorial.sh`
  — training + observation runners.

---

## Known non-issues

- **`O11.MC Dropout.subject_corr_mean = NaN`** in `results.json` — expected. MC Dropout needs
  active dropout layers at inference; these backbones use dropout = 0, so predictive
  variance is degenerate. The code flags `"warn": "dropout_inactive"`. Only the Entropy and
  Confidence signals are used. (The literal `NaN` token is non-strict-JSON; Python reads it,
  strict parsers may reject it.)
- **Non-tuple indexing `UserWarning`s** from `monai_utils/inferers/utils.py` on PyTorch 2.6
  — harmless deprecation notices, not errors.
