#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper A observation gate: O1-O11, generalized across backbones and datasets.

The ALIGNED matrix uses only the backbones EffiDec3D actually pairs a decoder with
(3DUXNET, SwinUNETR, SwinUNETRv2, MedNeXt): each is a full-vs-efficient pair and
runs all O1-O11. UNETR/nnUNet are NOT part of the aligned matrix -- EffiDec3D never
built a decoder pair for them, so a matched transition comparison is undefined.

Two modes, keyed by --network:
  * MATCHED backbones -> full-vs-efficient transition analysis; runs all O1-O11.
  * CONTROL backbones (UNETR, nnUNet) -> OPTIONAL full-only single-model run
    (O1/O2/O3/O4/O6/O10 concentration only, no O5/O9/O11). Retained for ad-hoc
    "does concentration recur on a non-EffiDec3D architecture" checks; never used
    for decoder-causal claims and not required for the paper.

Class count and per-organ names are derived from the dataset (BTCV13, feta, or any
dataset in load_datasets_transforms), so the same analysis runs on BTCV/FeTA/MSD.
Checkpoints are resolved by network subfolder; results/figures go to --obs_dir.

Usage (from /root/AdaDec3D/EffiDec3D):
  # matched, BTCV, 3D UX-Net (default)
  python run_observations.py --root /root/autodl-tmp/btcv-synapse --output /root/output
  # matched, FeTA, Swin UNETR
  python run_observations.py --root /root/autodl-tmp/feta-processed --dataset feta \
      --network SwinUNETR --obs_dir /root/obs/swin_feta
  # optional non-EffiDec3D control (full only; not part of the aligned matrix)
  python run_observations.py --network UNETR --dataset BTCV13 --obs_dir /root/obs/unetr_btcv
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.stats import pearsonr, spearmanr

from monai.data import DataLoader, Dataset
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.utils import set_determinism

from load_datasets_transforms import data_loader, data_transforms
from monai_utils.inferers.utils import sliding_window_inference_1out

# --- Class-name maps (foreground classes 1..N; length must equal out_classes-1) --
BTCV13_NAMES = ["Spleen", "R.Kidney", "L.Kidney", "Gallbladder", "Esophagus",
                "Liver", "Stomach", "Aorta", "IVC", "Veins",
                "Pancreas", "R.Adrenal", "L.Adrenal"]
FETA_NAMES = ["ECF", "GM", "WM", "Ventricles", "Cerebellum", "DeepGM", "Brainstem"]
HEPATIC_NAMES = ["Vessel", "Tumour"]   # MSD Task08: 2 foreground classes (out_classes=3)
DATASET_NAMES = {"BTCV13": BTCV13_NAMES, "feta": FETA_NAMES,
                 "Task08_HepaticVessel": HEPATIC_NAMES}

# Module global; main() sets it from the dataset. O1/O4/O10 reference CLASS_NAMES,
# so per-organ observations stay generic across datasets (else class_1..N).
CLASS_NAMES = list(BTCV13_NAMES)

# Aligned matrix = backbones EffiDec3D actually pairs a decoder with (matched
# full-vs-efficient comparison, all O1-O11). CONTROL_BACKBONES are OPTIONAL, off-
# matrix, full-only (single-model observations, no O5/O9/O11) -- kept for ad-hoc
# checks, never for decoder-causal claims. See module docstring.
MATCHED_BACKBONES = {"3DUXNET", "SwinUNETR", "SwinUNETRv2", "MedNeXt"}
CONTROL_BACKBONES = {"UNETR", "nnUNet"}  # optional / off-matrix
# main_train_BTCV_TU.py --network string per role (== output subfolder name).
EFFI_NETWORK = {"3DUXNET": "3DUXNET_EffiDec3D", "SwinUNETR": "SwinUNETR_EffiDec3D",
                "SwinUNETRv2": "SwinUNETRv2_EffiDec3D", "MedNeXt": "MedNeXt_M_EffiDec3D"}
FULL_NETWORK = {"3DUXNET": "3DUXNET", "SwinUNETR": "SwinUNETR",
                "SwinUNETRv2": "SwinUNETRv2", "MedNeXt": "MedNeXt_M",
                "UNETR": "UNETR", "nnUNet": "nnUNet"}
DEFAULT_FEATURE_SIZE = {"SwinUNETR": 48, "SwinUNETRv2": 48, "UNETR": 16, "MedNeXt": 48}

ROI = (96, 96, 96)
SW_BATCH = 4
OVERLAP = 0.7
OBS_DIR = "/root/obs"
RESULTS_FILE = os.path.join(OBS_DIR, "results.json")


def save_obs(tag, metrics):
    os.makedirs(OBS_DIR, exist_ok=True)
    data = {}
    if os.path.exists(RESULTS_FILE):
        try:
            data = json.load(open(RESULTS_FILE))
        except json.JSONDecodeError:
            pass
    data[tag] = metrics
    json.dump(data, open(RESULTS_FILE, "w"), indent=2)
    print(f"[{tag}] saved -> {RESULTS_FILE}")


def build_model(network, role, out_classes, device, img_size=(96, 96, 96),
                feature_size=48, resolution_factor=2, n_decoder_channels=48,
                skip_aggregation="concatenation"):
    """Build Full ('full') or EffiDec3D ('effi') model for `network`.

    Mirrors main_train_BTCV_TU.py instantiation. MedNeXt uses kernel size 3 to
    match the paper's ``MedNeXt-M-K3`` comparison. Only MATCHED_BACKBONES have an
    'effi' role. `resolution_factor`/`n_decoder_channels` parameterize the EffiDec3D
    decoder for the frozen-encoder factor decomposition (E1): the four corners are
    all role='effi' with (rf, nchan) in {(1,C_full),(1,48),(2,C_full),(2,48)}.
    """
    ic = 1
    if network == "3DUXNET":
        from networks.UXNet_3D.network_backbone import UXNET, UXNET_EffiDec3D
        if role == "effi":
            m = UXNET_EffiDec3D(in_chans=ic, out_chans=out_classes, depths=[2, 2, 2, 2],
                feat_size=[48, 96, 192, 384], n_decoder_channels=n_decoder_channels,
                drop_path_rate=0, layer_scale_init_value=1e-6, spatial_dims=3,
                skip_aggregation=skip_aggregation, resolution_factor=resolution_factor)
        else:
            m = UXNET(in_chans=ic, out_chans=out_classes, depths=[2, 2, 2, 2],
                feat_size=[48, 96, 192, 384], drop_path_rate=0,
                layer_scale_init_value=1e-6, spatial_dims=3)
    elif network in ("SwinUNETR", "SwinUNETRv2"):
        use_v2 = network == "SwinUNETRv2"
        if role == "effi":
            from networks.swin_unetr_effidec3d import SwinUNETR_EffiDec3D
            m = SwinUNETR_EffiDec3D(img_size=img_size, in_channels=ic,
                out_channels=out_classes, feature_size=feature_size,
                n_decoder_channels=n_decoder_channels, resolution_factor=resolution_factor,
                use_checkpoint=False, skip_aggregation=skip_aggregation, use_v2=use_v2)
        elif use_v2:
            from networks.swin_unetr_effidec3d import SwinUNETR as SwinFull
            m = SwinFull(img_size=img_size, in_channels=ic, out_channels=out_classes,
                feature_size=feature_size, use_checkpoint=False, use_v2=True)
        else:
            from monai.networks.nets import SwinUNETR
            m = SwinUNETR(img_size=img_size, in_channels=ic, out_channels=out_classes,
                feature_size=feature_size, use_checkpoint=False)
    elif network == "MedNeXt":
        if role == "effi":
            from networks.MedNeXt.mednextv1.create_mednextv1_effidec3d import create_mednextv1_effidec3d
            m = create_mednextv1_effidec3d(ic, out_classes, 'M', n_channels=feature_size,
                kernel_size=3, deep_supervision=False)
        else:
            # Paper's matched comparison is MedNeXt-M-K3, so the full pair member
            # must also be K3. NOTE: main_train_BTCV_TU.py's `MedNeXt_M` uses
            # kernel_size=5 -> a checkpoint trained under that name will NOT load
            # here. Train the matched full MedNeXt at K3 (or set K5 to match a K5
            # checkpoint, accepting kernel size as a confound).
            from networks.MedNeXt.mednextv1.create_mednext_v1 import create_mednext_v1
            m = create_mednext_v1(ic, out_classes, 'M', kernel_size=3,
                n_channels=feature_size, deep_supervision=False)
    elif network == "UNETR":                       # ecological control (full only)
        from monai.networks.nets import UNETR
        m = UNETR(in_channels=ic, out_channels=out_classes, img_size=img_size,
            feature_size=feature_size, hidden_size=768, mlp_dim=3072, num_heads=12,
            pos_embed="perceptron", norm_name="instance", res_block=True, dropout_rate=0.0)
    elif network == "nnUNet":                       # ecological control (full only)
        import torch.nn as nn
        from networks.nnunet.network_architecture.generic_UNet import Generic_UNet
        m = Generic_UNet(input_channels=ic, base_num_features=48, num_classes=out_classes,
            num_pool=4, num_conv_per_stage=2, conv_op=nn.Conv3d, norm_op=nn.BatchNorm3d,
            dropout_op=nn.Dropout3d, max_num_features=512, deep_supervision=False)
    else:
        raise NotImplementedError(f"network '{network}' is not wired in build_model")
    return m.to(device)


def load_ckpt(model, path, device):
    state = torch.load(path, map_location=device)
    state = state.get("model_state_dict", state) if isinstance(state, dict) else state
    # A full model trained WITH deep supervision (e.g. MedNeXt-M, --ds True) carries
    # auxiliary heads (out_1..out_4) that the inference model (deep_supervision=False)
    # does not have. These are training-only; keep only keys the model expects, then
    # require every remaining key to be present so genuine mismatches still surface.
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state.items() if k in model_keys}
    dropped = [k for k in state if k not in model_keys]
    missing, _ = model.load_state_dict(filtered, strict=False)
    if missing:
        raise RuntimeError(
            f"Checkpoint {path} is missing required weights for "
            f"{type(model).__name__}: {list(missing)[:8]}")
    if dropped:
        print(f"[load_ckpt] dropped {len(dropped)} training-only keys "
              f"(e.g. deep-supervision heads): {dropped[:4]}")
    model.eval()
    return model


def infer(model, img):
    return sliding_window_inference_1out(img, ROI, SW_BATCH, model, overlap=OVERLAP)


# --------------------------------------------------------------------------- #
def O1_error_distribution(val_loader, effi):
    from scipy.ndimage import binary_erosion
    boundary_err, interior_err, subj_ratio = [], [], []
    organ_err = {n: [] for n in CLASS_NAMES}
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].cuda()
            lbl = batch["label"].squeeze(1).long().cpu()
            pred = infer(effi, img).argmax(1).cpu()
            error = (pred != lbl).float().squeeze()
            lbl_sq = lbl.squeeze()
            for c, name in enumerate(CLASS_NAMES, start=1):
                mask = (lbl_sq == c)
                if mask.sum() > 0:
                    organ_err[name].append(error[mask].mean().item())
            fg = (lbl_sq > 0).numpy()
            interior = torch.from_numpy(binary_erosion(fg, iterations=3).astype(np.float32))
            boundary = torch.from_numpy(fg.astype(np.float32)) - interior
            b_s = (((error * boundary).sum() / boundary.sum()).item() if boundary.sum() > 0 else None)
            i_s = (((error * interior).sum() / interior.sum()).item() if interior.sum() > 0 else None)
            if b_s is not None:
                boundary_err.append(b_s)
            if i_s is not None:
                interior_err.append(i_s)
            if b_s is not None and i_s is not None and i_s > 0:
                subj_ratio.append(b_s / i_s)
    b_err = float(np.mean(boundary_err)); i_err = float(np.mean(interior_err))
    ratio = b_err / i_err if i_err > 0 else float("nan")
    _rng = np.random.default_rng(0)
    if subj_ratio:
        _rb = [np.mean(_rng.choice(subj_ratio, len(subj_ratio), replace=True)) for _ in range(2000)]
        ratio_ci = [float(np.percentile(_rb, 2.5)), float(np.percentile(_rb, 97.5))]
        ratio_subj_mean = float(np.mean(subj_ratio))
    else:
        ratio_ci = [float("nan"), float("nan")]; ratio_subj_mean = float("nan")
    print(f"[O1] boundary err={b_err:.3f}  interior err={i_err:.3f}  ratio={ratio:.1f}x  "
          f"subj-mean {ratio_subj_mean:.2f} CI[{ratio_ci[0]:.2f},{ratio_ci[1]:.2f}]")
    names_o1 = [n for n, v in organ_err.items() if v]
    vals_o1 = [float(np.mean(organ_err[n])) for n in names_o1]
    plt.figure(figsize=(11, 4))
    plt.bar(range(len(names_o1)), vals_o1)
    plt.xticks(range(len(names_o1)), names_o1, rotation=45, ha="right")
    plt.ylabel("Mean pixel error rate"); plt.title("O1: Per-Organ Error Rate")
    plt.tight_layout(); plt.savefig(f"{OBS_DIR}/O1_organ_error.png", dpi=150); plt.close()
    save_obs("O1", {"boundary_error": b_err, "interior_error": i_err,
                    "boundary_interior_ratio": round(ratio, 2),
                    "boundary_interior_ratio_subject_mean": round(ratio_subj_mean, 3),
                    "boundary_interior_ratio_ci": [round(ratio_ci[0], 3), round(ratio_ci[1], 3)],
                    "organ_error": {n: round(float(np.mean(v)), 4) for n, v in organ_err.items() if v}})


# --------------------------------------------------------------------------- #
def O2_entropy_distribution(val_loader, effi):
    all_entropy, high_unc_frac = [], []
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].cuda()
            logits = infer(effi, img)
            prob = logits.softmax(1).cpu()
            ent = -(prob * torch.log(prob + 1e-8)).sum(1).squeeze()
            all_entropy.append(ent.flatten().numpy()[::10])
            high_unc_frac.append((ent > 0.5).float().mean().item())
    all_ent = np.concatenate(all_entropy)
    pcts = {p: float(np.percentile(all_ent, p)) for p in [50, 75, 90, 95, 99]}
    frac_high = float(np.mean(high_unc_frac))
    print(f"[O2] entropy pcts={ {k: round(v,4) for k,v in pcts.items()} }  frac>0.5={frac_high:.2%}")
    plt.figure(figsize=(8, 4)); plt.hist(all_ent, bins=50, log=True)
    plt.xlabel("Entropy"); plt.ylabel("Voxel count (log)"); plt.title("O2: Entropy Distribution")
    plt.tight_layout(); plt.savefig(f"{OBS_DIR}/O2_entropy.png", dpi=150); plt.close()
    skewed = pcts[50] < pcts[95]
    save_obs("O2", {"percentiles": pcts, "fraction_above_0.5": frac_high, "go_skewed": bool(skewed)})


def O3_unc_error_corr(val_loader, effi, full=None):
    """Entropy-based predictability (on-thesis: predicting FLIPS, not just error).

    Primary (needs `full`): per-subject AUROC/AUPRC of the efficient model's entropy
    predicting where the full decoder produces a **positive flip** (efficient wrong,
    full right) --- i.e. can a test-time signal locate where extra decoder computation
    helps. This is the novel, on-narrative metric. Baselines retained: per-subject
    AUROC/AUPRC/ECE of entropy predicting the efficient model's own **error** (a
    well-studied misclassification-detection quantity), and the descriptive pooled-bin
    entropy--error correlation (pseudo-replicated, diagnostic only). Go: flip AUROC
    (or error AUROC when no full pair) CI-lower > 0.5.
    """
    try:                                   # prefer sklearn; fall back to numpy
        from sklearn.metrics import roc_auc_score, average_precision_score
        def _auroc(s, y): return float(roc_auc_score(y, s))
        def _auprc(s, y): return float(average_precision_score(y, s))
    except Exception:
        from scipy.stats import rankdata
        def _auroc(s, y):
            n_pos = int(y.sum()); n_neg = int((~y).sum())
            if n_pos == 0 or n_neg == 0:
                return float("nan")
            r = rankdata(s)               # average ranks handle entropy ties
            return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
        def _auprc(s, y):
            order = np.argsort(s)[::-1]; ys = y[order].astype(np.float64)
            tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
            prec = tp / np.maximum(tp + fp, 1e-9)
            total = ys.sum()
            if total == 0:
                return float("nan")
            rec = tp / total
            return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))

    def _ece(conf, correct, n_bins=15):
        edges = np.linspace(0, 1, n_bins + 1); ece = 0.0
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            m = (conf >= lo) & (conf <= hi) if i == n_bins - 1 else (conf >= lo) & (conf < hi)
            if m.any():
                ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
        return float(ece)

    rng = np.random.default_rng(0)
    x_ent, y_err = [], []                                  # pooled-bin (descriptive)
    subj_auroc, subj_auprc, subj_ece, subj_prev = [], [], [], []
    subj_fauroc, subj_fauprc, subj_fprev = [], [], []      # positive-flip prediction
    subj_nauroc, subj_aauroc, subj_pvn = [], [], []        # neg-flip, any-flip, pos-vs-neg AUROC
    subj_aauprc, subj_pvnprc = [], []                      # any-flip, direction AUPRC (minority-class)
    predict_flip = full is not None and full is not effi
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].cuda()
            lbl = batch["label"].squeeze(1).long().cpu().squeeze()
            logits = infer(effi, img)
            prob = logits.softmax(1).cpu()
            pred = logits.argmax(1).cpu().squeeze()
            ent = -(prob * torch.log(prob + 1e-8)).sum(1).squeeze()
            conf = prob.max(1).values.squeeze()
            ent_np = ent.numpy().reshape(-1)
            err_np = (pred != lbl).numpy().reshape(-1)
            conf_np = conf.numpy().reshape(-1)
            # (1) pooled-bin correlation over the whole volume (descriptive).
            emax = float(ent_np.max())
            for b in range(20):
                lo, hi = b / 20 * emax, (b + 1) / 20 * emax
                mask = (ent_np >= lo) & (ent_np < hi)
                if mask.sum() > 100:
                    x_ent.append(float(ent_np[mask].mean())); y_err.append(float(err_np[mask].mean()))
            # (2) subject-level discrimination/calibration on the foreground union.
            body = ((lbl > 0) | (pred > 0)).numpy().reshape(-1)
            idx = np.flatnonzero(body)
            if idx.size == 0:
                continue
            if idx.size > 300000:                          # cap for speed; seeded
                idx = rng.choice(idx, 300000, replace=False)
            e = ent_np[idx]; yb = err_np[idx].astype(bool); cb = conf_np[idx]
            if yb.sum() > 0 and (~yb).sum() > 0:
                subj_auroc.append(_auroc(e, yb))
                subj_auprc.append(_auprc(e, yb))
                subj_prev.append(float(yb.mean()))
            subj_ece.append(_ece(cb, (~yb).astype(np.float64)))
            if predict_flip:                               # entropy -> flip discrimination
                pred_f = infer(full, img).argmax(1).cpu().squeeze()
                posflip = ((pred_f == lbl) & (pred != lbl)).numpy().reshape(-1)[idx]
                negflip = ((pred_f != lbl) & (pred == lbl)).numpy().reshape(-1)[idx]
                anyflip = posflip | negflip
                yf = posflip.astype(bool)                  # (a) positive flip: full fixes effi
                if yf.sum() > 0 and (~yf).sum() > 0:
                    subj_fauroc.append(_auroc(e, yf))
                    subj_fauprc.append(_auprc(e, yf))
                    subj_fprev.append(float(yf.mean()))
                yn = negflip.astype(bool)                  # (b) negative flip: full breaks effi
                if yn.sum() > 0 and (~yn).sum() > 0:
                    subj_nauroc.append(_auroc(e, yn))
                ya = anyflip.astype(bool)                  # (c) any flip: the prediction changes
                if ya.sum() > 0 and (~ya).sum() > 0:
                    subj_aauroc.append(_auroc(e, ya))
                    subj_aauprc.append(_auprc(e, ya))      # flips are the minority class
                # (d) KEY: within flip voxels, can entropy tell better-from-worse?
                if ya.sum() > 10 and yf[ya].sum() > 0 and (~yf[ya]).sum() > 0:
                    subj_pvn.append(_auroc(e[ya], yf[ya]))
                    subj_pvnprc.append(_auprc(e[ya], yf[ya]))
    r_p = float(pearsonr(x_ent, y_err)[0]); r_s = float(spearmanr(x_ent, y_err)[0])

    def _boot_ci(vals):
        vals = np.asarray([v for v in vals if not np.isnan(v)], dtype=np.float64)
        if vals.size == 0:
            return float("nan"), [float("nan"), float("nan")]
        boot = [np.mean(rng.choice(vals, vals.size, replace=True)) for _ in range(2000)]
        return float(vals.mean()), [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    auroc_m, auroc_ci = _boot_ci(subj_auroc)        # error-prediction (baseline)
    auprc_m, auprc_ci = _boot_ci(subj_auprc)
    ece_m, ece_ci = _boot_ci(subj_ece)
    prev_m = float(np.mean(subj_prev)) if subj_prev else float("nan")
    fauroc_m, fauroc_ci = _boot_ci(subj_fauroc)     # flip-prediction (primary)
    fauprc_m, fauprc_ci = _boot_ci(subj_fauprc)
    fprev_m = float(np.mean(subj_fprev)) if subj_fprev else float("nan")
    nauroc_m, nauroc_ci = _boot_ci(subj_nauroc)     # negative-flip prediction
    aauroc_m, aauroc_ci = _boot_ci(subj_aauroc)     # any-flip prediction
    aauprc_m, aauprc_ci = _boot_ci(subj_aauprc)
    pvn_m, pvn_ci = _boot_ci(subj_pvn)              # KEY: pos-vs-neg discrimination
    pvnprc_m, pvnprc_ci = _boot_ci(subj_pvnprc)
    err_go = bool(not np.isnan(auroc_ci[0]) and auroc_ci[0] > 0.5)
    flip_go = bool(not np.isnan(fauroc_ci[0]) and fauroc_ci[0] > 0.5)
    go = flip_go if predict_flip else err_go        # headline gate = flip when available
    print(f"[O3] pooled-bin r={r_p:.3f} rho={r_s:.3f} (descriptive only)")
    if predict_flip:
        print(f"     FLIP  AUROC={fauroc_m:.3f} CI[{fauroc_ci[0]:.3f},{fauroc_ci[1]:.3f}]  "
              f"AUPRC={fauprc_m:.3f} (pos-flip prev={fprev_m:.4f})  "
              f"{'GO' if flip_go else 'NO-GO'} (AUROC CI-lower > 0.5)")
        print(f"     any-flip AUROC={aauroc_m:.3f}  neg-flip AUROC={nauroc_m:.3f}  "
              f"POS-vs-NEG AUROC={pvn_m:.3f} CI[{pvn_ci[0]:.3f},{pvn_ci[1]:.3f}]  "
              f"(~0.5 => entropy cannot separate improvement from degradation)")
    print(f"     error AUROC={auroc_m:.3f} CI[{auroc_ci[0]:.3f},{auroc_ci[1]:.3f}]  "
          f"AUPRC={auprc_m:.3f} (err prev={prev_m:.3f})  ECE={ece_m:.3f}  (baseline)")
    plt.figure(figsize=(6, 5)); plt.scatter(x_ent, y_err, alpha=0.7)
    plt.xlabel("Mean entropy (bin)"); plt.ylabel("Error rate (bin)")
    plt.title(f"O3: pooled-bin r={r_p:.3f} (descriptive); flip AUROC={fauroc_m:.3f}")
    plt.tight_layout(); plt.savefig(f"{OBS_DIR}/O3_unc_error.png", dpi=150); plt.close()
    save_obs("O3", {
        # primary: entropy predicts positive flips (needs full pair)
        "flip_auroc_mean": fauroc_m, "flip_auroc_ci": fauroc_ci,
        "flip_auprc_mean": fauprc_m, "flip_auprc_ci": fauprc_ci,
        "positive_flip_prevalence_mean": fprev_m, "flip_go": flip_go,
        "n_subjects_flip": len(subj_fauroc),
        # flip discrimination: any / negative / and KEY positive-vs-negative (AUROC + AUPRC)
        "any_flip_auroc_mean": aauroc_m, "any_flip_auroc_ci": aauroc_ci,
        "any_flip_auprc_mean": aauprc_m, "any_flip_auprc_ci": aauprc_ci,
        "neg_flip_auroc_mean": nauroc_m, "neg_flip_auroc_ci": nauroc_ci,
        "pos_vs_neg_auroc_mean": pvn_m, "pos_vs_neg_auroc_ci": pvn_ci,
        "pos_vs_neg_auprc_mean": pvnprc_m, "pos_vs_neg_auprc_ci": pvnprc_ci,
        "n_subjects_pvn": len(subj_pvn),
        # baseline: entropy predicts the efficient model's own error
        "subject_auroc_mean": auroc_m, "subject_auroc_ci": auroc_ci,
        "subject_auprc_mean": auprc_m, "subject_auprc_ci": auprc_ci,
        "error_prevalence_mean": prev_m,
        "subject_ece_mean": ece_m, "subject_ece_ci": ece_ci,
        "n_subjects": len(subj_auroc),
        "pooled_bin_pearson_r": r_p, "pooled_bin_spearman_rho": r_s,
        "pearson_r": r_p, "spearman_rho": r_s,   # backward-compat keys
        "go": go,
    })


def O4_per_organ(val_loader, effi, post_pred, post_lbl):
    organ_dice = {n: [] for n in CLASS_NAMES}
    organ_ent = {n: [] for n in CLASS_NAMES}
    dm = DiceMetric(include_background=False, reduction="none")
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].cuda()
            lbl = batch["label"].cpu()
            logits = infer(effi, img).cpu()   # cpu so DiceMetric y_pred/y share device
            prob = logits.softmax(1)
            ent = -(prob * torch.log(prob + 1e-8)).sum(1).squeeze()
            dvals = dm(post_pred(logits.squeeze(0)).unsqueeze(0),
                       post_lbl(lbl.squeeze(0)).unsqueeze(0))[0]
            for c, name in enumerate(CLASS_NAMES):
                organ_dice[name].append(dvals[c].item())
                mask = (lbl.squeeze() == c + 1)
                if mask.sum() > 0:
                    organ_ent[name].append(ent[mask].mean().item())
    dice_summary = {n: float(np.nanmean(organ_dice[n])) for n in CLASS_NAMES}
    ent_summary = {n: (float(np.nanmean(organ_ent[n])) if organ_ent[n] else float("nan"))
                   for n in CLASS_NAMES}
    print("[O4] per-organ dice/entropy:")
    for n in CLASS_NAMES:
        print(f"      {n:12s} dice={dice_summary[n]:.3f}  ent={ent_summary[n]:.4f}")
    save_obs("O4", {"dice": {k: round(v, 4) for k, v in dice_summary.items()},
                    "entropy": {k: round(v, 4) for k, v in ent_summary.items()}})
    return dice_summary, organ_ent


def O5_decoder_gain(val_loader, effi, full):
    """H1: global positive / negative / net flip rates with a subject bootstrap CI.

    For each subject R_pos = mean_v P(v), R_neg = mean_v N(v), and two summaries:
      - GAIN     R_net = R_pos - R_neg  = U(v)=P(v)-N(v)  (where the decoder *helps*)
      - ACTIVITY R_act = R_pos + R_neg  = A(v)=P(v)+N(v)  (where the decoder *acts*, either way)
    We bootstrap the twelve subject-level values (not the correlated voxels). Answers whether
    the full decoder improves predictions uniformly or simultaneously corrects and degrades
    them: R_net ~ 0 with a much larger R_act > 0 is a bidirectional cancellation (lots of
    activity, ~no net gain) that a Dice gap alone cannot reveal."""
    subj_pos_rate, subj_neg_rate = [], []
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].cuda()
            lbl = batch["label"].squeeze(1).long().cpu().squeeze()
            pred_full = infer(full, img).argmax(1).cpu().squeeze()
            pred_effi = infer(effi, img).argmax(1).cpu().squeeze()
            pos = ((pred_full == lbl) & (pred_effi != lbl)).float()
            neg = ((pred_full != lbl) & (pred_effi == lbl)).float()
            subj_pos_rate.append(float(pos.mean())); subj_neg_rate.append(float(neg.mean()))
    if not subj_pos_rate:
        print("[H1] no subjects; skipping"); return
    mean_pos, mean_neg = float(np.mean(subj_pos_rate)), float(np.mean(subj_neg_rate))
    net_rate = np.array(subj_pos_rate) - np.array(subj_neg_rate)     # GAIN  = P - N
    act_rate = np.array(subj_pos_rate) + np.array(subj_neg_rate)     # ACTIVITY = P + N
    rng = np.random.default_rng(0)
    nr_boot = [np.mean(rng.choice(net_rate, len(net_rate), replace=True)) for _ in range(2000)]
    nr_ci = [float(np.percentile(nr_boot, 2.5)), float(np.percentile(nr_boot, 97.5))]
    ar_boot = [np.mean(rng.choice(act_rate, len(act_rate), replace=True)) for _ in range(2000)]
    ar_ci = [float(np.percentile(ar_boot, 2.5)), float(np.percentile(ar_boot, 97.5))]
    net_neutral = bool(nr_ci[0] <= 0 <= nr_ci[1])
    print(f"[H1] R_pos={mean_pos:.5f}  R_neg={mean_neg:.5f}  "
          f"GAIN(R_net)={float(net_rate.mean()):.5f} CI[{nr_ci[0]:.5f},{nr_ci[1]:.5f}] "
          f"{'net-neutral' if net_neutral else 'directional'}  "
          f"ACTIVITY(R_act)={float(act_rate.mean()):.5f} CI[{ar_ci[0]:.5f},{ar_ci[1]:.5f}]")
    plt.figure(figsize=(5, 4))
    plt.bar([0, 1, 2], [mean_pos, mean_neg, float(net_rate.mean())],
            color=["seagreen", "firebrick", "steelblue"], alpha=.8)
    plt.errorbar([2], [float(net_rate.mean())],
                 yerr=[[float(net_rate.mean()) - nr_ci[0]], [nr_ci[1] - float(net_rate.mean())]],
                 fmt="none", ecolor="k", capsize=5)
    plt.axhline(0, color="k", lw=.6, ls=":")
    plt.xticks([0, 1, 2], ["$R_{pos}$", "$R_{neg}$", "$R_{net}$"])
    plt.ylabel("flip rate"); plt.title("H1: global decoder flip rates")
    plt.tight_layout(); plt.savefig(f"{OBS_DIR}/H1_global_flips.png", dpi=150); plt.close()
    save_obs("O5", {"mean_positive_rate": mean_pos, "mean_negative_rate": mean_neg,
                    "subject_pos_rate_mean": mean_pos, "subject_neg_rate_mean": mean_neg,
                    "subject_net_rate_mean": float(net_rate.mean()),      # GAIN = P - N
                    "subject_net_rate_ci": nr_ci,
                    "subject_activity_rate_mean": float(act_rate.mean()),  # ACTIVITY = P + N
                    "subject_activity_rate_ci": ar_ci,
                    "net_neutral": net_neutral})


def O9_opportunity(val_loader, effi, full, block_size=16, halo=4, primary_budget=20,
                   provenance=None, foreground="union"):
    """Estimate selective-decoding opportunity without overstating the result.

    Two selectors are reported:
      1. voxel oracle: an intentionally non-deployable upper bound;
      2. region selector: non-overlapping 3-D blocks ranked by mean entropy and
         expanded by ``halo`` voxels to approximate convolutional context.

    Utility is (positive transitions - negative transitions), normalized by all
    positive E0-over-E1 transitions in that subject. Random and entropy policies
    are compared with a paired subject bootstrap.
    """
    if block_size <= 0 or halo < 0:
        raise ValueError("block_size must be > 0 and halo must be >= 0")
    selection_rng = np.random.default_rng(0)
    bootstrap_rng = np.random.default_rng(1)
    budgets = np.array([5, 10, 20, 30, 50])
    n_random = 100
    n_bootstrap = 2000

    voxel_entropy, voxel_random = [], []
    region_entropy, region_random, region_executed = [], [], []
    voxel_positive, voxel_negative = [], []
    region_positive, region_negative = [], []
    subject_results = []

    def paired_summary(selected, random):
        selected = np.asarray(selected, dtype=np.float64)
        random = np.asarray(random, dtype=np.float64)
        n_subjects = len(selected)
        boot_selected, boot_random, boot_diff = [], [], []
        for _ in range(n_bootstrap):
            indices = bootstrap_rng.integers(n_subjects, size=n_subjects)
            selected_mean = selected[indices].mean(0)
            random_mean = random[indices].mean(0)
            boot_selected.append(selected_mean)
            boot_random.append(random_mean)
            boot_diff.append(selected_mean - random_mean)
        selected_ci = np.percentile(boot_selected, [2.5, 97.5], axis=0)
        random_ci = np.percentile(boot_random, [2.5, 97.5], axis=0)
        diff_ci = np.percentile(boot_diff, [2.5, 97.5], axis=0)
        return {
            "selected_mean": selected.mean(0),
            "random_mean": random.mean(0),
            "selected_ci": selected_ci,
            "random_ci": random_ci,
            "diff_ci": diff_ci,
        }

    def block_slices(shape):
        blocks = []
        for z in range(0, shape[0], block_size):
            for y in range(0, shape[1], block_size):
                for x in range(0, shape[2], block_size):
                    core = (slice(z, min(z + block_size, shape[0])),
                            slice(y, min(y + block_size, shape[1])),
                            slice(x, min(x + block_size, shape[2])))
                    expanded = (slice(max(0, z - halo), min(z + block_size + halo, shape[0])),
                                slice(max(0, y - halo), min(y + block_size + halo, shape[1])),
                                slice(max(0, x - halo), min(x + block_size + halo, shape[2])))
                    blocks.append((core, expanded))
        return blocks

    with torch.no_grad():
        for subject_idx, batch in enumerate(val_loader):
            meta = batch.get("image_meta_dict", {})
            source = meta.get("filename_or_obj", f"subject_{subject_idx:02d}")
            if isinstance(source, (list, tuple)):
                source = source[0]
            subject_id = os.path.basename(str(source))
            img = batch["image"].cuda()
            lbl = batch["label"].squeeze(1).long().cpu().squeeze()
            pred_full = infer(full, img).argmax(1).cpu().squeeze()
            logits_e = infer(effi, img)
            pred_effi = logits_e.argmax(1).cpu().squeeze()
            prob_e = logits_e.softmax(1).cpu()
            ent = -(prob_e * torch.log(prob_e + 1e-8)).sum(1).squeeze()

            pos = ((pred_full == lbl) & (pred_effi != lbl)).numpy().astype(np.float32)
            neg = ((pred_full != lbl) & (pred_effi == lbl)).numpy().astype(np.float32)
            net = pos - neg
            if foreground == "all":
                body = np.ones(lbl.shape, dtype=bool)
            elif foreground == "gt":
                body = (lbl > 0).numpy()
            else:   # union(GT, Full, Effi) — predeclared main result
                body = ((lbl > 0) | (pred_full > 0) | (pred_effi > 0)).numpy()
            entropy = ent.numpy()
            total_positive = float(pos[body].sum())
            if total_positive == 0:
                continue

            # Voxel-wise oracle upper bound.
            body_indices = np.flatnonzero(body)
            body_entropy = entropy.flat[body_indices]
            body_pos = pos.flat[body_indices]
            body_neg = neg.flat[body_indices]
            body_net = net.flat[body_indices]
            voxel_order = np.argsort(body_entropy)[::-1]
            subject_voxel_entropy, subject_voxel_random = [], []
            subject_voxel_positive, subject_voxel_negative = [], []
            for budget in budgets:
                k = max(1, int(np.ceil(len(body_indices) * budget / 100)))
                selected_voxels = voxel_order[:k]
                subject_voxel_positive.append(
                    float(body_pos[selected_voxels].sum() / total_positive))
                subject_voxel_negative.append(
                    float(body_neg[selected_voxels].sum() / total_positive))
                subject_voxel_entropy.append(
                    subject_voxel_positive[-1] - subject_voxel_negative[-1])
                random_values = [
                    float(body_net[selection_rng.choice(len(body_net), k, replace=False)].sum()
                          / total_positive)
                    for _ in range(n_random)
                ]
                subject_voxel_random.append(float(np.mean(random_values)))

            # Contiguous block selector. Block count is the compute-budget proxy;
            # halo-expanded volume is saved as the actual executed-volume proxy.
            blocks = block_slices(entropy.shape)
            scores = np.array([
                float(entropy[core][body[core]].mean()) if body[core].any() else -np.inf
                for core, _ in blocks
            ])
            block_utilities = np.array([
                float(net[core][body[core]].sum() / total_positive)
                for core, _ in blocks
            ])
            block_positive = np.array([
                float(pos[core][body[core]].sum() / total_positive)
                for core, _ in blocks
            ])
            block_negative = np.array([
                float(neg[core][body[core]].sum() / total_positive)
                for core, _ in blocks
            ])
            ranked_blocks = np.argsort(scores)[::-1]
            subject_region_entropy, subject_region_random = [], []
            subject_region_positive, subject_region_negative = [], []
            subject_executed = []
            for budget in budgets:
                n_selected = max(1, int(np.ceil(len(blocks) * budget / 100)))

                def executed_fraction(block_ids):
                    selected_mask = np.zeros(entropy.shape, dtype=bool)
                    for block_id in block_ids:
                        selected_mask[blocks[int(block_id)][1]] = True
                    return float(selected_mask.mean())

                selected_ids = ranked_blocks[:n_selected]
                positive_utility = float(block_positive[selected_ids].sum())
                negative_utility = float(block_negative[selected_ids].sum())
                utility = positive_utility - negative_utility
                executed = executed_fraction(selected_ids)
                subject_region_positive.append(positive_utility)
                subject_region_negative.append(negative_utility)
                subject_region_entropy.append(utility)
                subject_executed.append(executed)
                random_values = [
                    float(block_utilities[selection_rng.choice(
                        len(blocks), n_selected, replace=False)].sum())
                    for _ in range(n_random)
                ]
                subject_region_random.append(float(np.mean(random_values)))

            voxel_entropy.append(subject_voxel_entropy)
            voxel_random.append(subject_voxel_random)
            voxel_positive.append(subject_voxel_positive)
            voxel_negative.append(subject_voxel_negative)
            region_entropy.append(subject_region_entropy)
            region_random.append(subject_region_random)
            region_positive.append(subject_region_positive)
            region_negative.append(subject_region_negative)
            region_executed.append(subject_executed)
            subject_results.append({
                "subject_index": subject_idx,
                "subject_id": subject_id,
                "positive_transitions": int(pos[body].sum()),
                "negative_transitions": int(neg[body].sum()),
                "voxel_entropy_positive_utility": subject_voxel_positive,
                "voxel_entropy_negative_utility": subject_voxel_negative,
                "voxel_entropy_net_utility": subject_voxel_entropy,
                "voxel_random_net_utility": subject_voxel_random,
                "region_entropy_positive_utility": subject_region_positive,
                "region_entropy_negative_utility": subject_region_negative,
                "region_entropy_net_utility": subject_region_entropy,
                "region_random_net_utility": subject_region_random,
                "region_executed_volume_fraction": subject_executed,
            })

    if not subject_results:
        print("[O9] no positive decoder transitions found; skipping")
        return

    voxel_stats = paired_summary(voxel_entropy, voxel_random)
    region_stats = paired_summary(region_entropy, region_random)
    primary_idx = int(np.where(budgets == primary_budget)[0][0])
    go = bool(
        region_stats["diff_ci"][0, primary_idx] > 0
        and region_stats["selected_mean"][primary_idx] > 0
    )

    print(f"[O9] budgets={budgets.tolist()}  block={block_size}  halo={halo}")
    print(f"     voxel oracle net ={voxel_stats['selected_mean'].round(3).tolist()}")
    print(f"     region net       ={region_stats['selected_mean'].round(3).tolist()}")
    print(f"     region random    ={region_stats['random_mean'].round(3).tolist()}")
    print(f"     paired diff CI-lo={region_stats['diff_ci'][0].round(3).tolist()}")
    print(f"     executed volume  ={np.mean(region_executed, axis=0).round(3).tolist()}")
    print(f"     {'GO' if go else 'NO-GO'} at predeclared {primary_budget}% block budget")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, title, stats in [
        (axes[0], "Voxel oracle (upper bound)", voxel_stats),
        (axes[1], f"{block_size}³ blocks + {halo}-voxel halo", region_stats),
    ]:
        ax.plot(budgets, stats["selected_mean"] * 100, "o-", color="steelblue",
                label="Entropy")
        ax.fill_between(budgets, stats["selected_ci"][0] * 100,
                        stats["selected_ci"][1] * 100, alpha=.2, color="steelblue")
        ax.plot(budgets, stats["random_mean"] * 100, "o--", color="gray",
                label="Matched random")
        ax.fill_between(budgets, stats["random_ci"][0] * 100,
                        stats["random_ci"][1] * 100, alpha=.15, color="gray")
        ax.axhline(0, color="black", lw=.8, ls=":")
        ax.set_xlabel("Selection budget (%)")
        ax.set_title(title)
        ax.legend()
    axes[0].set_ylabel("Net utility / all positive transitions (%)")
    fig.suptitle("O9: Net Selective-Allocation Opportunity")
    fig.tight_layout()
    fig.savefig(f"{OBS_DIR}/O9_opportunity_corrected.png", dpi=150)
    plt.close(fig)

    def serialise_stats(stats):
        return {
            "entropy_net_utility_mean": stats["selected_mean"].round(6).tolist(),
            "entropy_net_utility_ci_lower": stats["selected_ci"][0].round(6).tolist(),
            "entropy_net_utility_ci_upper": stats["selected_ci"][1].round(6).tolist(),
            "random_net_utility_mean": stats["random_mean"].round(6).tolist(),
            "random_net_utility_ci_lower": stats["random_ci"][0].round(6).tolist(),
            "random_net_utility_ci_upper": stats["random_ci"][1].round(6).tolist(),
            "paired_diff_ci_lower": stats["diff_ci"][0].round(6).tolist(),
            "paired_diff_ci_upper": stats["diff_ci"][1].round(6).tolist(),
        }

    save_obs("O9_corrected", {
        "budgets_pct": budgets.tolist(),
        "primary_budget_pct": int(primary_budget),
        "utility_definition": "(positive-negative)/all_positive_flips",
        "foreground": foreground,
        "n_subjects": len(subject_results),
        "n_random_repeats": n_random,
        "n_bootstrap": n_bootstrap,
        "provenance": provenance or {},
        "voxel_oracle": {
            "entropy_positive_utility_mean":
                np.mean(voxel_positive, axis=0).round(6).tolist(),
            "entropy_negative_utility_mean":
                np.mean(voxel_negative, axis=0).round(6).tolist(),
            **serialise_stats(voxel_stats),
        },
        "region_selector": {
            "block_size": int(block_size),
            "halo": int(halo),
            "executed_volume_fraction_mean":
                np.mean(region_executed, axis=0).round(6).tolist(),
            "entropy_positive_utility_mean":
                np.mean(region_positive, axis=0).round(6).tolist(),
            "entropy_negative_utility_mean":
                np.mean(region_negative, axis=0).round(6).tolist(),
            **serialise_stats(region_stats),
        },
        "subject_results": subject_results,
        "go": go,
    })


def O_surface_metrics(val_loader, analyzed, out_classes):
    """E4: per-organ Normalized Surface Dice (NSD) at 1- and 2-voxel tolerance (MONAI).
    A surface-level metric for the 'high-res decoder serves boundaries' claim, rather
    than only voxel label flips.
    """
    try:
        from monai.metrics import compute_surface_dice
    except Exception:
        print("[O_surface] monai compute_surface_dice unavailable; skipping"); return
    taus = [1.0, 2.0]
    onehot = AsDiscrete(to_onehot=out_classes)
    onehot_p = AsDiscrete(argmax=True, to_onehot=out_classes)
    per_tau = {t: [] for t in taus}
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].cuda(); lbl = batch["label"].cpu()
            logits = infer(analyzed, img).cpu()
            yp = onehot_p(logits.squeeze(0)).unsqueeze(0)
            yt = onehot(lbl.squeeze(0)).unsqueeze(0)
            for t in taus:
                nsd = compute_surface_dice(yp, yt, class_thresholds=[t] * (out_classes - 1),
                                           include_background=False)
                per_tau[t].append(nsd.numpy())
    res = {}
    for t in taus:
        arr = np.concatenate(per_tau[t], axis=0)   # (N, C-1)
        res[f"nsd_tau{int(t)}_mean"] = float(np.nanmean(arr))
        res[f"nsd_tau{int(t)}_per_organ"] = {
            CLASS_NAMES[c]: round(float(np.nanmean(arr[:, c])), 4)
            for c in range(arr.shape[1]) if c < len(CLASS_NAMES)}
    print(f"[O_surface] NSD tau1={res.get('nsd_tau1_mean', float('nan')):.4f}  "
          f"tau2={res.get('nsd_tau2_mean', float('nan')):.4f}")
    save_obs("O_surface", res)


def O_boundary_flips(val_loader, effi, full):
    """H2-A: net decoder benefit resolved by distance to the GT organ boundary.

    For each voxel v let d(v) be the Euclidean distance (voxels) to the nearest
    foreground-boundary voxel; bin into {0-1, 1-2, 2-4, 4-8, >8}. Per bin report the
    subject-mean positive / negative / net flip rate, with a subject bootstrap CI on
    the net rate:

        R_pos(D_k) = mean_v in D_k  P(v)   [full right, effi wrong]
        R_neg(D_k) = mean_v in D_k  N(v)   [effi right, full wrong]
        R_net(D_k) = R_pos(D_k) - R_neg(D_k)

    Upgrades the error-based observation 'error concentrates at the boundary' to the
    benefit-based claim 'the decoder's *net* benefit concentrates at the boundary'.
    Rates are computed on the union(GT, Full, Effi) foreground so false positives are
    not dropped."""
    from scipy.ndimage import distance_transform_edt, binary_erosion
    edges = [0, 1, 2, 4, np.inf]                 # boundary / near / transition / interior
    labels = ["0-1", "1-2", "2-4", ">4"]
    nb = len(labels)
    s_pos = [[] for _ in range(nb)]
    s_neg = [[] for _ in range(nb)]
    s_net = [[] for _ in range(nb)]
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].cuda()
            lbl = batch["label"].squeeze(1).long().cpu().squeeze().numpy()
            pf = infer(full, img).argmax(1).cpu().squeeze().numpy()
            pe = infer(effi, img).argmax(1).cpu().squeeze().numpy()
            fg = lbl > 0
            if not fg.any():
                continue
            pos = (pf == lbl) & (pe != lbl)
            neg = (pf != lbl) & (pe == lbl)
            bnd = fg & ~binary_erosion(fg, iterations=1)      # inner boundary shell
            dist = distance_transform_edt(~bnd)               # 0 on the shell
            body = fg | (pf > 0) | (pe > 0)                   # union foreground
            for bi in range(nb):
                lo, hi = edges[bi], edges[bi + 1]
                m = body & (dist >= lo) & (dist < hi)
                cnt = int(m.sum())
                if cnt > 0:
                    p = float(pos[m].sum()); q = float(neg[m].sum())
                    s_pos[bi].append(p / cnt)
                    s_neg[bi].append(q / cnt)
                    s_net[bi].append((p - q) / cnt)

    def _m(x):
        return [float(np.mean(v)) if v else float("nan") for v in x]

    pos_m, neg_m, net_m = _m(s_pos), _m(s_neg), _m(s_net)
    rng = np.random.default_rng(0)
    net_ci = []
    for bi in range(nb):
        v = np.asarray(s_net[bi], dtype=np.float64)
        if v.size == 0:
            net_ci.append([float("nan"), float("nan")]); continue
        boot = [np.mean(rng.choice(v, v.size, replace=True)) for _ in range(2000)]
        net_ci.append([float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))])
    print("[H2-boundary] distance-to-boundary net flip (subject mean):")
    for bi in range(nb):
        print(f"      d={labels[bi]:>4}  pos={pos_m[bi]:.5f}  neg={neg_m[bi]:.5f}  "
              f"net={net_m[bi]:.5f}  CI[{net_ci[bi][0]:.5f},{net_ci[bi][1]:.5f}]")
    plt.figure(figsize=(7, 5)); xs = np.arange(nb)
    plt.plot(xs, pos_m, "g--o", ms=4, alpha=.7, label="positive")
    plt.plot(xs, neg_m, "r--o", ms=4, alpha=.7, label="negative")
    plt.plot(xs, net_m, "b-o", ms=5, label="net")
    plt.axhline(0, color="k", lw=.6, ls=":")
    plt.xticks(xs, labels); plt.xlabel("distance to GT boundary (voxels)")
    plt.ylabel("flip rate"); plt.title("H2: boundary-resolved decoder benefit")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{OBS_DIR}/H2_boundary.png", dpi=150); plt.close()
    save_obs("H2_boundary", {
        "distance_bins": labels,
        "positive_rate": pos_m, "negative_rate": neg_m, "net_rate": net_m,
        "net_rate_ci": net_ci,
        "note": "voxels binned by distance-to-GT-boundary; subject-mean rates over "
                "union(GT,Full,Effi) foreground; net = positive - negative",
    })
    return net_m


def O_pareto(val_loader, effi, full, block_size=16, halo=4):
    """Fig 2 (headline): multi-signal region opportunity curve. Rank contiguous
    ``block_size`` blocks by several signals and plot the net-flip utility recovered
    vs. selection budget --- true-flip ORACLE (upper bound), entropy, confidence,
    a boundary proxy, and matched random (with 95% CI). Utility per block is the net
    flip within the block, normalized by the subject's positive net mass
    sum_r max(B(r),0) so coverage is a fraction of the recoverable net benefit (C1)."""
    from scipy.ndimage import distance_transform_edt, binary_erosion
    budgets = np.array([5, 10, 20, 30, 50]); n_random = 100; n_boot = 2000
    signals = ["oracle", "entropy", "confidence", "boundary"]
    rec = {s: [] for s in signals + ["random"]}
    abs_net, abs_pos_mass = [], []          # per-subject total net and positive-net mass (voxels)
    srng = np.random.default_rng(0); brng = np.random.default_rng(1)
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].cuda()
            lbl = batch["label"].squeeze(1).long().cpu().squeeze().numpy()
            pred_f = infer(full, img).argmax(1).cpu().squeeze().numpy()
            le = infer(effi, img); prob_e = le.softmax(1).cpu()
            pred_e = le.argmax(1).cpu().squeeze().numpy()
            ent = (-(prob_e * torch.log(prob_e + 1e-8)).sum(1)).squeeze().numpy()
            conf = (1 - prob_e.max(1).values.squeeze()).numpy()   # 1-maxprob: high=uncertain
            pos = ((pred_f == lbl) & (pred_e != lbl)).astype(np.float32)
            neg = ((pred_f != lbl) & (pred_e == lbl)).astype(np.float32)
            net = pos - neg
            body = (lbl > 0) | (pred_f > 0) | (pred_e > 0)
            if not body.any():
                continue
            fg = lbl > 0
            bnd = fg & ~binary_erosion(fg, iterations=1) if fg.any() else fg
            dist = distance_transform_edt(~bnd) if bnd.any() else np.zeros_like(ent)
            shp = ent.shape
            cores = [(slice(z, min(z + block_size, shp[0])),
                      slice(y, min(y + block_size, shp[1])),
                      slice(x, min(x + block_size, shp[2])))
                     for z in range(0, shp[0], block_size)
                     for y in range(0, shp[1], block_size)
                     for x in range(0, shp[2], block_size)]
            util_raw, s_ent, s_conf, s_bnd = [], [], [], []
            for c in cores:
                bm = body[c]
                if bm.any():
                    util_raw.append(float(net[c][bm].sum()))      # raw block net benefit B(r)
                    s_ent.append(float(ent[c][bm].mean()))
                    s_conf.append(float(conf[c][bm].mean()))
                    s_bnd.append(float(-dist[c][bm].mean()))      # nearer boundary = higher
                else:
                    util_raw.append(0.0); s_ent.append(-np.inf); s_conf.append(-np.inf); s_bnd.append(-np.inf)
            util_raw = np.array(util_raw)
            pos_mass = float(util_raw[util_raw > 0].sum())        # C1 denom: sum_r max(B(r),0)
            if pos_mass == 0:
                continue
            util = util_raw / pos_mass                            # fraction of positive net mass
            abs_net.append(float(util_raw.sum())); abs_pos_mass.append(pos_mass)
            score = {"entropy": np.array(s_ent), "confidence": np.array(s_conf),
                     "boundary": np.array(s_bnd), "oracle": util}
            nb = len(cores)
            subj = {s: [] for s in rec}
            for bud in budgets:
                k = max(1, int(np.ceil(nb * bud / 100)))
                for s in signals:
                    top = np.argsort(score[s])[::-1][:k]
                    subj[s].append(float(util[top].sum()))
                rr = [float(util[srng.choice(nb, k, replace=False)].sum()) for _ in range(n_random)]
                subj["random"].append(float(np.mean(rr)))
            for s in rec:
                rec[s].append(subj[s])
    if not rec["oracle"]:
        print("[O_pareto] no positive flips; skipping"); return
    out = {"budgets_pct": budgets.tolist(), "block_size": int(block_size)}
    colors = {"oracle": "black", "entropy": "steelblue", "confidence": "darkorange",
              "boundary": "seagreen", "random": "gray"}
    plt.figure(figsize=(7, 5))
    for s in ["oracle", "entropy", "confidence", "boundary", "random"]:
        arr = np.asarray(rec[s], dtype=np.float64)        # (subjects, budgets)
        m = arr.mean(0)
        boot = np.array([arr[brng.integers(arr.shape[0], size=arr.shape[0])].mean(0)
                         for _ in range(n_boot)])
        lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)
        ls = "--" if s == "random" else "-"
        plt.plot(budgets, m * 100, ls, marker="o", color=colors[s], label=s)
        if s in ("random", "entropy"):
            plt.fill_between(budgets, lo * 100, hi * 100, color=colors[s], alpha=.15)
        out[s] = {"net_recovered_mean": m.round(6).tolist(),
                  "ci_lower": lo.round(6).tolist(), "ci_upper": hi.round(6).tolist()}
    plt.xlabel("Fraction of regions given the full decoder (%)")
    plt.ylabel("Net flip utility recovered (%)")
    plt.title("Selective-allocation opportunity: signals vs.\\ random")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{OBS_DIR}/O_pareto.png", dpi=150); plt.close()

    # ---- C1/C2: oracle concentration (how concentrated is the useful net benefit) ----
    # out["oracle"]["net_recovered_mean"][k] = C_oracle(k), the fraction of the subject's
    # positive net mass sum_r max(B(r),0) captured by the top-k% oracle blocks.
    orc = np.asarray(out["oracle"]["net_recovered_mean"], dtype=np.float64)

    def _cov(bud):
        return float(orc[int(np.where(budgets == bud)[0][0])])

    k80 = next((int(b) for b, v in zip(budgets, orc) if v >= 0.8), None)
    out["concentration"] = {                              # C1 + C2
        "oracle_coverage_at_10pct": _cov(10),
        "oracle_coverage_at_20pct": _cov(20),
        "budget_for_80pct_net_pct": k80,
        "abs_net_voxels_mean": float(np.mean(abs_net)),        # signed total net (voxels)
        "abs_pos_mass_voxels_mean": float(np.mean(abs_pos_mass)),  # positive net mass (voxels)
        "note": "C1/C2: coverage = fraction of the subject's positive net mass "
                "sum_r max(B(r),0) recovered by the top-k% oracle blocks; K80 = smallest "
                "block budget reaching 0.8. abs_net is the signed total net benefit in "
                "voxels: report it WITH the coverage so a steep curve over a ~0 absolute "
                "total is not mistaken for a large opportunity.",
    }
    # ---- P2: paired (signal - random) net recovery at the primary 20% budget ----
    pidx = int(np.where(budgets == 20)[0][0])
    rnd = np.asarray(rec["random"], dtype=np.float64)[:, pidx]
    sel_gap = {}
    for s in ["oracle", "entropy", "confidence", "boundary"]:
        diff = np.asarray(rec[s], dtype=np.float64)[:, pidx] - rnd
        boot = [float(diff[brng.integers(diff.shape[0], size=diff.shape[0])].mean())
                for _ in range(n_boot)]
        sel_gap[s] = {"mean_minus_random": float(diff.mean()),
                      "ci": [float(np.percentile(boot, 2.5)),
                             float(np.percentile(boot, 97.5))]}
    out["selection_gap_at_20pct"] = {                     # P2
        **sel_gap,
        "note": "P2: paired subject-bootstrap of (signal - matched random) net recovered "
                "at 20% block budget. A deployable signal beats random iff its CI excludes 0; "
                "the oracle row is the non-deployable upper bound.",
    }
    save_obs("O_pareto", out)
    print(f"[O_pareto] C1 oracle@20%={_cov(20):.3f} K80={k80}  "
          f"P2 entropy-random@20%={sel_gap['entropy']['mean_minus_random']:.4f} "
          f"CI[{sel_gap['entropy']['ci'][0]:.4f},{sel_gap['entropy']['ci'][1]:.4f}]  "
          f"({len(rec['oracle'])} subjects)")


def O_anatomy(val_loader, effi, full):
    """H2-B: per-organ decoder benefit on the union(GT, Full, Effi) foreground.

    For organ c let Omega_c = {v : y(v)=c OR y_full(v)=c OR y_effi(v)=c} (union, so false
    positives are not dropped). Reports the subject-mean positive/negative/net flip rate
        R_pos,c = mean_{v in Omega_c} P(v),  R_neg,c = mean_{v in Omega_c} N(v),
        R_net,c = R_pos,c - R_neg,c,
    the per-organ Dice change dDice_c = Dice_full,c - Dice_effi,c (on GT), organ size, and
    the size-vs-net-flip Spearman rho. Answers whether the decoder's value is
    anatomy-dependent, and how a global net ~0 (H1) coexists with a macro Dice gap."""
    from scipy.stats import spearmanr
    o_pos = {n: [] for n in CLASS_NAMES}; o_neg = {n: [] for n in CLASS_NAMES}
    o_size = {n: [] for n in CLASS_NAMES}
    dde = {n: [] for n in CLASS_NAMES}; ddf = {n: [] for n in CLASS_NAMES}

    def _dice(a, b):
        s = int(a.sum()) + int(b.sum())
        return (2.0 * int((a & b).sum()) / s) if s > 0 else float("nan")

    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].cuda()
            lbl = batch["label"].squeeze(1).long().cpu().squeeze().numpy()
            pred_f = infer(full, img).argmax(1).cpu().squeeze().numpy()
            pred_e = infer(effi, img).argmax(1).cpu().squeeze().numpy()
            pos = (pred_f == lbl) & (pred_e != lbl)
            neg = (pred_f != lbl) & (pred_e == lbl)
            for c, name in enumerate(CLASS_NAMES, start=1):
                gt = (lbl == c)
                dom = gt | (pred_f == c) | (pred_e == c)     # union foreground for organ c
                if dom.sum() > 0:
                    o_pos[name].append(float(pos[dom].mean()))
                    o_neg[name].append(float(neg[dom].mean()))
                if gt.sum() > 0:
                    o_size[name].append(float(gt.sum()))
                    dde[name].append(_dice(pred_e == c, gt))
                    ddf[name].append(_dice(pred_f == c, gt))
    names = [n for n in CLASS_NAMES if o_pos[n]]
    pos_m = [float(np.mean(o_pos[n])) for n in names]
    neg_m = [float(np.mean(o_neg[n])) for n in names]
    net_m = [p - q for p, q in zip(pos_m, neg_m)]
    size = [float(np.mean(o_size[n])) if o_size[n] else float("nan") for n in names]
    ddice = [float(np.nanmean(ddf[n]) - np.nanmean(dde[n])) if dde[n] else float("nan")
             for n in names]
    rho = float(spearmanr(size, net_m)[0]) if len(names) > 2 else float("nan")
    print(f"[H2-anatomy] per-organ net flip (union fg) + dDice; size~net Spearman={rho:.3f}")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    xs = np.arange(len(names))
    axs[0].bar(xs - 0.2, pos_m, 0.4, color="seagreen", alpha=.7, label="positive")
    axs[0].bar(xs + 0.2, neg_m, 0.4, color="firebrick", alpha=.7, label="negative")
    axs[0].plot(xs, net_m, "b-o", ms=3, label="net")
    axs[0].axhline(0, color="k", lw=.6, ls=":")
    axs[0].set_xticks(xs); axs[0].set_xticklabels(names, rotation=45, ha="right")
    axs[0].set_ylabel("flip rate (union fg)")
    axs[0].set_title("(a) Per-organ decoder benefit"); axs[0].legend()
    axs[1].scatter(size, net_m, color="steelblue")
    for i, n in enumerate(names):
        axs[1].annotate(n, (size[i], net_m[i]), fontsize=7)
    axs[1].set_xscale("log"); axs[1].axhline(0, color="k", lw=.6, ls=":")
    axs[1].set_xlabel("organ size (voxels, log)"); axs[1].set_ylabel("net flip rate")
    axs[1].set_title(f"(b) Size vs.\\ benefit (Spearman ${rho:.2f}$)")
    fig.tight_layout(); fig.savefig(f"{OBS_DIR}/O_anatomy.png", dpi=150); plt.close(fig)
    save_obs("O_anatomy", {
        "organ": names, "positive_rate": pos_m, "negative_rate": neg_m, "net_rate": net_m,
        "organ_size": size, "dice_delta_full_minus_effi": ddice, "size_net_spearman": rho,
        "foreground": "union(GT,Full,Effi) per organ",
    })


def O_recovery(val_loader, effi, full, out_classes, block_size=16, halo=4):
    """Recoverability (R): hybrid selective-decoding performance-recovery curve.

    For a spatial budget k%, build a HYBRID prediction --- the top-k% of contiguous 16^3
    blocks (ranked by a routing signal) take the FULL decoder's argmax, the rest keep the
    EFFICIENT decoder's argmax --- and report the resulting mean foreground Dice vs budget
    for four routers: true-net ORACLE (upper bound), entropy, confidence, and matched
    random; plus the Efficient-only (0%) and Full (100%) anchors, per-router NSD at the
    20% budget, and the halo-expanded executed-volume fraction (the real compute proxy).
    This turns the abstract net-flip recovery (P2) into *actual segmentation performance*
    recovered: how much of Full's Dice can a k%-budget selective decoder recover, and can a
    deployable signal find those regions? Offline analysis --- not a deployed system."""
    try:
        from monai.metrics import compute_surface_dice
        have_nsd = True
    except Exception:
        have_nsd = False
    budgets = np.array([5, 10, 20, 30, 50]); n_random = 10; n_boot = 2000
    nsd_budget = 20; fgcls = list(range(1, out_classes))
    signals = ["oracle", "entropy", "confidence", "random"]

    def _dice(pred, gt):
        ds = []
        for c in fgcls:
            b = (gt == c)
            if int(b.sum()) == 0:
                continue
            a = (pred == c); s = int(a.sum()) + int(b.sum())
            ds.append(2.0 * int((a & b).sum()) / s if s > 0 else 0.0)
        return float(np.mean(ds)) if ds else float("nan")

    def _nsd(pred, gt, tau=1.0):
        if not have_nsd:
            return float("nan")
        def oh(a):
            return AsDiscrete(to_onehot=out_classes)(torch.as_tensor(a)[None].long()).unsqueeze(0)
        v = compute_surface_dice(oh(pred), oh(gt),
                                 class_thresholds=[tau] * (out_classes - 1),
                                 include_background=False)
        return float(np.nanmean(v.numpy()))

    rec_dice = {s: [] for s in signals}     # per subject -> [len(budgets)]
    rec_exec = {s: [] for s in signals}
    rec_nsd20 = {s: [] for s in signals}    # per subject NSD at the 20% budget
    effi_dice, full_dice, effi_nsd, full_nsd = [], [], [], []
    srng = np.random.default_rng(0); brng = np.random.default_rng(1)
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].cuda()
            gt = batch["label"].squeeze(1).long().cpu().squeeze().numpy()
            pf = infer(full, img).argmax(1).cpu().squeeze().numpy()
            le = infer(effi, img); prob_e = le.softmax(1).cpu()
            pe = le.argmax(1).cpu().squeeze().numpy()
            ent = (-(prob_e * torch.log(prob_e + 1e-8)).sum(1)).squeeze().numpy()
            conf = (1 - prob_e.max(1).values.squeeze()).numpy()   # high = uncertain
            body = (gt > 0) | (pf > 0) | (pe > 0)
            net = ((pf == gt) & (pe != gt)).astype(np.float32) - ((pf != gt) & (pe == gt)).astype(np.float32)
            shp = ent.shape
            cores = [(slice(z, min(z + block_size, shp[0])),
                      slice(y, min(y + block_size, shp[1])),
                      slice(x, min(x + block_size, shp[2])))
                     for z in range(0, shp[0], block_size)
                     for y in range(0, shp[1], block_size)
                     for x in range(0, shp[2], block_size)]
            nb = len(cores)
            s_net = np.array([float(net[c][body[c]].sum()) if body[c].any() else 0.0 for c in cores])
            s_ent = np.array([float(ent[c][body[c]].mean()) if body[c].any() else -np.inf for c in cores])
            s_conf = np.array([float(conf[c][body[c]].mean()) if body[c].any() else -np.inf for c in cores])
            score = {"oracle": s_net, "entropy": s_ent, "confidence": s_conf}

            def hybrid(ids):
                out = pe.copy(); m = np.zeros(shp, dtype=bool)
                for b in ids:
                    m[cores[int(b)]] = True
                out[m] = pf[m]
                return out

            def exec_frac(ids):
                m = np.zeros(shp, dtype=bool)
                for b in ids:
                    cz, cy, cx = cores[int(b)]
                    m[max(0, cz.start - halo):min(shp[0], cz.stop + halo),
                      max(0, cy.start - halo):min(shp[1], cy.stop + halo),
                      max(0, cx.start - halo):min(shp[2], cx.stop + halo)] = True
                return float(m.mean())

            effi_dice.append(_dice(pe, gt)); full_dice.append(_dice(pf, gt))
            effi_nsd.append(_nsd(pe, gt)); full_nsd.append(_nsd(pf, gt))
            for s in signals:
                sd, sx, snsd20 = [], [], float("nan")
                for bud in budgets:
                    k = max(1, int(np.ceil(nb * bud / 100)))
                    if s == "random":
                        ds, xs = [], []
                        for _ in range(n_random):
                            ids = srng.choice(nb, k, replace=False)
                            ds.append(_dice(hybrid(ids), gt)); xs.append(exec_frac(ids))
                        sd.append(float(np.mean(ds))); sx.append(float(np.mean(xs)))
                        if bud == nsd_budget:
                            snsd20 = _nsd(hybrid(srng.choice(nb, k, replace=False)), gt)
                    else:
                        ids = np.argsort(score[s])[::-1][:k]
                        hy = hybrid(ids)
                        sd.append(_dice(hy, gt)); sx.append(exec_frac(ids))
                        if bud == nsd_budget:
                            snsd20 = _nsd(hy, gt)
                rec_dice[s].append(sd); rec_exec[s].append(sx); rec_nsd20[s].append(snsd20)

    if not effi_dice:
        print("[R-recovery] no subjects; skipping"); return
    ed, fd = float(np.mean(effi_dice)), float(np.mean(full_dice))
    en, fn = float(np.nanmean(effi_nsd)), float(np.nanmean(full_nsd))
    out = {"budgets_pct": budgets.tolist(), "block_size": int(block_size), "halo": int(halo),
           "effi_dice": ed, "full_dice": fd, "full_minus_effi_dice": fd - ed,
           "effi_nsd_tau1": en, "full_nsd_tau1": fn, "nsd_budget_pct": nsd_budget}
    colors = {"oracle": "black", "entropy": "steelblue", "confidence": "darkorange", "random": "gray"}
    plt.figure(figsize=(7, 5))
    plt.axhline(ed, color="seagreen", ls=":", lw=1, label=f"Efficient only ({ed:.3f})")
    plt.axhline(fd, color="crimson", ls=":", lw=1, label=f"Full ({fd:.3f})")
    for s in signals:
        arr = np.asarray(rec_dice[s], dtype=np.float64)      # (subjects, budgets)
        m = arr.mean(0)
        boot = np.array([arr[brng.integers(arr.shape[0], size=arr.shape[0])].mean(0) for _ in range(n_boot)])
        lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)
        ls = "--" if s == "random" else "-"
        plt.plot(budgets, m, ls, marker="o", color=colors[s], label=s)
        if s in ("oracle", "entropy", "random"):
            plt.fill_between(budgets, lo, hi, color=colors[s], alpha=.12)
        out[s] = {"dice_mean": m.round(5).tolist(),
                  "dice_ci_lower": lo.round(5).tolist(), "dice_ci_upper": hi.round(5).tolist(),
                  "executed_volume_fraction_mean": np.mean(rec_exec[s], axis=0).round(4).tolist(),
                  f"nsd_tau1_at_{nsd_budget}pct_mean": float(np.nanmean(rec_nsd20[s]))}
    plt.xlabel("Spatial budget: blocks routed to Full decoder (%)")
    plt.ylabel("Mean foreground Dice")
    plt.title("Recoverability: hybrid selective-decoding Dice")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{OBS_DIR}/R_recovery.png", dpi=150); plt.close()
    # Dice-space recovered fraction at 20% for the deployable entropy router vs oracle.
    i20 = int(np.where(budgets == 20)[0][0]); gap = max(fd - ed, 1e-9)
    ent20 = out["entropy"]["dice_mean"][i20]; orc20 = out["oracle"]["dice_mean"][i20]; rnd20 = out["random"]["dice_mean"][i20]
    out["recovered_fraction_at_20pct"] = {
        "oracle": (orc20 - ed) / gap, "entropy": (ent20 - ed) / gap, "random": (rnd20 - ed) / gap,
        "note": "(hybrid Dice - Efficient Dice) / (Full Dice - Efficient Dice) at 20% block budget",
    }
    save_obs("R_recovery", out)
    print(f"[R-recovery] Effi={ed:.3f} Full={fd:.3f} (gap {fd-ed:+.3f})  @20%: "
          f"oracle={orc20:.3f} entropy={ent20:.3f} random={rnd20:.3f}  "
          f"entropy recovers {(ent20-ed)/gap*100:.0f}% of the gap")


def main():
    global OBS_DIR, RESULTS_FILE, CLASS_NAMES
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/root/autodl-tmp/btcv-synapse")
    p.add_argument("--output", default="/root/output")
    p.add_argument("--dataset", default="BTCV13")
    p.add_argument("--network", default="3DUXNET",
                   choices=sorted(MATCHED_BACKBONES | CONTROL_BACKBONES),
                   help="Backbone. MATCHED_BACKBONES run full-vs-EffiDec3D (all O1-O11); "
                        "CONTROL_BACKBONES (UNETR, nnUNet) are full-only ecological "
                        "baselines (single-model O1/O2/O3/O4/O6/O10 only).")
    p.add_argument("--feature_size", type=int, default=None,
                   help="Backbone feature size (default: 48 Swin/MedNeXt, 16 UNETR)")
    p.add_argument("--obs_dir", default="/root/obs",
                   help="Directory for figures and results.json")
    p.add_argument("--e0_ckpt", default=None, help="Explicit full-model checkpoint")
    p.add_argument("--e1_ckpt", default=None, help="Explicit EffiDec3D checkpoint")
    # Frozen-encoder factor decomposition (E1): build each side as an EffiDec3D-family
    # decoder with explicit knobs instead of the default full/effi roles. When any of
    # these is set for a side, that side is built as role='effi' with (rf, nchan).
    p.add_argument("--e0_rf", type=int, default=None, help="E0 EffiDec3D resolution_factor (factorial)")
    p.add_argument("--e0_nchan", type=int, default=None, help="E0 EffiDec3D n_decoder_channels (factorial)")
    p.add_argument("--e1_rf", type=int, default=None, help="E1 EffiDec3D resolution_factor (factorial)")
    p.add_argument("--e1_nchan", type=int, default=None, help="E1 EffiDec3D n_decoder_channels (factorial)")
    p.add_argument("--o9_foreground", default="union", choices=["all", "gt", "union"],
                   help="O9 denominator/evaluation domain: all voxels, GT foreground, "
                        "or union(GT,Full,Effi) [default; predeclared main result]")
    p.add_argument("--only_o9", action="store_true",
                   help="Run only the corrected O9 analysis (matched backbones only)")
    p.add_argument("--o9_block_size", type=int, default=16,
                   help="O9 contiguous selector core block edge length in voxels")
    p.add_argument("--o9_halo", type=int, default=4,
                   help="O9 context halo added around each selected block")
    p.add_argument("--o9_primary_budget", type=int, default=20,
                   choices=[5, 10, 20, 30, 50],
                   help="Predeclared O9 block budget used for the Go/No-Go test")
    p.add_argument("--skip_aggregation", default="concatenation",
                   choices=["addition", "concatenation"],
                   help="EffiDec3D decoder skip aggregation; must match how the effi "
                        "checkpoint was TRAINED. Default 'concatenation' = the EffiDec3D "
                        "network default / paper config; use 'addition' for legacy checkpoints.")
    p.add_argument("--appendix", action="store_true",
                   help="Also emit the demoted appendix diagnostics (O1 boundary-error "
                        "ratio, O2 entropy histogram, O4 per-organ difficulty, and the O9 "
                        "executed-volume/halo sensitivity). Default off: the main pipeline "
                        "reports only the six converged metrics H1,H2,C1/C2,P1,P2.")
    args_ns = p.parse_args()
    OBS_DIR = args_ns.obs_dir
    RESULTS_FILE = os.path.join(OBS_DIR, "results.json")
    set_determinism(seed=0)
    os.makedirs(OBS_DIR, exist_ok=True)
    device = torch.device("cuda:0")

    network = args_ns.network
    matched = network in MATCHED_BACKBONES
    fsize = args_ns.feature_size or DEFAULT_FEATURE_SIZE.get(network, 48)
    full_folder = FULL_NETWORK[network]

    # Data + class count/names derived from the dataset (generalizes beyond BTCV13).
    ld_args = argparse.Namespace(root=args_ns.root, dataset=args_ns.dataset,
                                 mode="validation", crop_sample=4, img_size=[96, 96, 96])
    _, val_samples, out_classes = data_loader(ld_args)
    _, val_transform = data_transforms(ld_args)
    CLASS_NAMES = list(DATASET_NAMES.get(args_ns.dataset,
                       [f"class_{i}" for i in range(1, out_classes)]))
    val_files = [{"image": im, "label": lb}
                 for im, lb in zip(val_samples["images"], val_samples["labels"])]
    val_loader = DataLoader(Dataset(data=val_files, transform=val_transform),
                            batch_size=1, shuffle=False, num_workers=2)
    post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)
    post_lbl = AsDiscrete(to_onehot=out_classes)
    print(f"Network: {network} ({'matched' if matched else 'control'})  "
          f"dataset: {args_ns.dataset}  classes: {out_classes}  val cases: {len(val_files)}\n")

    # Factorial override: if a side's rf/nchan is given, build it as an EffiDec3D-family
    # decoder with those knobs (frozen-encoder E1 corners) instead of full/effi roles.
    e0_factorial = args_ns.e0_rf is not None or args_ns.e0_nchan is not None
    e1_factorial = args_ns.e1_rf is not None or args_ns.e1_nchan is not None

    # Full model (always present).
    e0 = ([args_ns.e0_ckpt] if args_ns.e0_ckpt else
          sorted(glob.glob(f"{args_ns.output}/*/{full_folder}/{args_ns.dataset}/best_metric_model.pth")))
    assert e0, f"No full checkpoint under {args_ns.output}/*/{full_folder}/{args_ns.dataset}/"
    if e0_factorial:
        full = load_ckpt(build_model(network, "effi", out_classes, device, feature_size=fsize,
                                     resolution_factor=(args_ns.e0_rf or 2),
                                     n_decoder_channels=(args_ns.e0_nchan or 48),
                                     skip_aggregation=args_ns.skip_aggregation), e0[-1], device)
        print(f"Full (factorial rf={args_ns.e0_rf},nchan={args_ns.e0_nchan}): {e0[-1]}")
    else:
        full = load_ckpt(build_model(network, "full", out_classes, device, feature_size=fsize),
                         e0[-1], device)
        print(f"Full: {e0[-1]}")

    # EffiDec3D counterpart (matched backbones only, or explicit factorial side).
    effi, e1 = None, None
    if matched or e1_factorial:
        e1 = ([args_ns.e1_ckpt] if args_ns.e1_ckpt else
              sorted(glob.glob(f"{args_ns.output}/*/{EFFI_NETWORK[network]}/{args_ns.dataset}/best_metric_model.pth")))
        if e1:
            effi = load_ckpt(build_model(network, "effi", out_classes, device, feature_size=fsize,
                                         resolution_factor=(args_ns.e1_rf or 2),
                                         n_decoder_channels=(args_ns.e1_nchan or 48),
                                         skip_aggregation=args_ns.skip_aggregation), e1[-1], device)
            print(f"Effi (rf={args_ns.e1_rf or 2},nchan={args_ns.e1_nchan or 48}): {e1[-1]}")
        else:
            print(f"[warn] no EffiDec3D checkpoint for {network}; falling back to single-model mode")
    analyzed = effi if effi is not None else full   # model whose error/entropy we characterize
    role = "effi" if effi is not None else "full"

    if args_ns.only_o9:
        if effi is None:
            print("[O9] needs a matched full+EffiDec3D pair; nothing to do"); return
        O9_opportunity(val_loader, effi, full, block_size=args_ns.o9_block_size,
                       halo=args_ns.o9_halo, primary_budget=args_ns.o9_primary_budget,
                       foreground=args_ns.o9_foreground,
                       provenance={"network": network, "dataset": args_ns.dataset,
                                   "full_checkpoint": e0[-1], "effi_checkpoint": e1[-1]})
        print(f"\nDone. Corrected O9 figure + results.json in {OBS_DIR}")
        return

    # Converged reporting framework: every main observation maps to one of the six
    # metrics along Heterogeneity -> Concentration -> Predictability. The deprecated
    # O1/O2/O4/O9 diagnostics survive only behind --appendix.
    save_obs("framework_map", {
        "H1_global_flips": "O5 (subject-mean positive/negative/net flip + bootstrap CI)",
        "H2_boundary": "H2_boundary (distance-to-boundary net flip)",
        "H2_anatomy": "O_anatomy (per-organ union-fg net flip + per-organ Dice delta + size)",
        "H2_support": "O_surface (per-organ NSD, optional boundary-quality column)",
        "C1_C2_concentration": "O_pareto['oracle'] curve + O_pareto['concentration'] top-k/K80/abs_net",
        "P1_predictability": "O3 (any / positive / direction=pos-vs-neg AUROC)",
        "P2_selection": "O_pareto signals vs oracle vs random + "
                        "O_pareto['selection_gap_at_20pct'] paired diff CI",
        "R_recovery": "R_recovery (hybrid selective-decoding Dice/NSD vs budget; "
                      "oracle/entropy/confidence/random + Effi/Full anchors)",
        "activity_gain": "O5 activity=P+N (where the decoder acts) vs gain=P-N (where it helps)",
        "appendix": "O1, O2, O4, O9 (behind --appendix)",
    })

    # ---- P1 predictability (single-model signal; flip-prediction needs the full pair) ----
    O3_unc_error_corr(val_loader, analyzed, full=(full if effi is not None else None))
    O_surface_metrics(val_loader, analyzed, out_classes)   # H2 support: per-organ NSD

    # ---- H1/H2/C1/C2/P2: need the full-vs-EffiDec3D transition pair ----
    if effi is not None:
        O5_decoder_gain(val_loader, effi, full)            # H1: global P/N/net + subject CI
        O_boundary_flips(val_loader, effi, full)           # H2-A: boundary-resolved net
        O_anatomy(val_loader, effi, full)                  # H2-B: per-organ union-fg net + Dice delta
        O_pareto(val_loader, effi, full, block_size=args_ns.o9_block_size,
                 halo=args_ns.o9_halo)                     # C1+C2 (oracle) + P2 (signal vs random)
        O_recovery(val_loader, effi, full, out_classes,    # R: hybrid Dice recovery curve
                   block_size=args_ns.o9_block_size, halo=args_ns.o9_halo)
    else:
        print("[H1/H2/C/P2] skipped — ecological control has no EffiDec3D pair")

    # ---- Appendix diagnostics (demoted; error-based or superseded) ----
    if args_ns.appendix:
        O1_error_distribution(val_loader, analyzed)
        O2_entropy_distribution(val_loader, analyzed)
        O4_per_organ(val_loader, analyzed, post_pred, post_lbl)
        if effi is not None:
            O9_opportunity(val_loader, effi, full, block_size=args_ns.o9_block_size,
                           halo=args_ns.o9_halo, primary_budget=args_ns.o9_primary_budget,
                           foreground=args_ns.o9_foreground,
                           provenance={"network": network, "dataset": args_ns.dataset,
                                       "full_checkpoint": e0[-1], "effi_checkpoint": e1[-1]})
    print(f"\nDone. Figures + results.json in {OBS_DIR}")


if __name__ == "__main__":
    main()
