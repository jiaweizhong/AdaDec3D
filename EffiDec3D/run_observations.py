#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper A observation gate: O2, O3, O4, O5, O9 on the existing E0/E1 checkpoints.

Runnable version of the code blocks in Observation_Study.md (command-line, no
notebook). Resolves E0/E1 best_metric_model.pth via glob, runs the decisive
decoder-gain analyses, saves figures to /root/obs and metrics to
/root/obs/results.json. O5 is the critical Go/No-Go gate; O9 is the headline.

Usage (from /root/AdaDec3D/EffiDec3D):
  python run_observations.py --root /root/autodl-tmp/btcv-synapse --output /root/output
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
from networks.UXNet_3D.network_backbone import UXNET, UXNET_EffiDec3D

# Standard BTCV/Synapse 13-organ order (class 1..13), matches CSV columns
BTCV_NAMES = ["Spleen", "R.Kidney", "L.Kidney", "Gallbladder", "Esophagus",
              "Liver", "Stomach", "Aorta", "IVC", "Veins",
              "Pancreas", "R.Adrenal", "L.Adrenal"]

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


def build_model(kind, device):
    if kind == "effi":
        return UXNET_EffiDec3D(
            in_chans=1, out_chans=14, depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384], n_decoder_channels=48,
            drop_path_rate=0, layer_scale_init_value=1e-6, spatial_dims=3,
            skip_aggregation="addition", resolution_factor=2).to(device)
    return UXNET(in_chans=1, out_chans=14, depths=[2, 2, 2, 2],
                 feat_size=[48, 96, 192, 384]).to(device)


def load_ckpt(model, path, device):
    state = torch.load(path, map_location=device)
    state = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state)
    model.eval()
    return model


def infer(model, img):
    return sliding_window_inference_1out(img, ROI, SW_BATCH, model, overlap=OVERLAP)


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


def O3_unc_error_corr(val_loader, effi):
    x_ent, y_err = [], []
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].cuda()
            lbl = batch["label"].squeeze(1).long().cpu()
            logits = infer(effi, img)
            prob = logits.softmax(1).cpu()
            ent = -(prob * torch.log(prob + 1e-8)).sum(1).squeeze()
            err = (logits.argmax(1).cpu().squeeze() != lbl.squeeze()).float()
            emax = ent.max().item()
            for b in range(20):
                lo, hi = b / 20 * emax, (b + 1) / 20 * emax
                mask = (ent >= lo) & (ent < hi)
                if mask.sum() > 100:
                    x_ent.append(ent[mask].mean().item())
                    y_err.append(err[mask].mean().item())
    r_p = float(pearsonr(x_ent, y_err)[0]); r_s = float(spearmanr(x_ent, y_err)[0])
    print(f"[O3] Pearson r={r_p:.3f}  Spearman rho={r_s:.3f}  {'GO' if r_p>0.60 else 'NO-GO'} (>0.60)")
    plt.figure(figsize=(6, 5)); plt.scatter(x_ent, y_err, alpha=0.7)
    plt.xlabel("Mean entropy (bin)"); plt.ylabel("Error rate (bin)")
    plt.title(f"O3: Uncertainty-Error r={r_p:.3f}")
    plt.tight_layout(); plt.savefig(f"{OBS_DIR}/O3_unc_error.png", dpi=150); plt.close()
    save_obs("O3", {"pearson_r": r_p, "spearman_rho": r_s, "go": r_p > 0.60})


def O4_per_organ(val_loader, effi, post_pred, post_lbl):
    organ_dice = {n: [] for n in BTCV_NAMES}
    organ_ent = {n: [] for n in BTCV_NAMES}
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
            for c, name in enumerate(BTCV_NAMES):
                organ_dice[name].append(dvals[c].item())
                mask = (lbl.squeeze() == c + 1)
                if mask.sum() > 0:
                    organ_ent[name].append(ent[mask].mean().item())
    dice_summary = {n: float(np.nanmean(organ_dice[n])) for n in BTCV_NAMES}
    ent_summary = {n: (float(np.nanmean(organ_ent[n])) if organ_ent[n] else float("nan"))
                   for n in BTCV_NAMES}
    print("[O4] per-organ dice/entropy:")
    for n in BTCV_NAMES:
        print(f"      {n:12s} dice={dice_summary[n]:.3f}  ent={ent_summary[n]:.4f}")
    save_obs("O4", {"dice": {k: round(v, 4) for k, v in dice_summary.items()},
                    "entropy": {k: round(v, 4) for k, v in ent_summary.items()}})
    return dice_summary, organ_ent


def O5_decoder_gain(val_loader, effi, full):
    subj_r_pearson, subj_r_spearman = [], []
    g_ent, g_pos, g_neg, g_net = [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].cuda()
            lbl = batch["label"].squeeze(1).long().cpu().squeeze()
            pred_full = infer(full, img).argmax(1).cpu().squeeze()
            logits_e = infer(effi, img)
            pred_effi = logits_e.argmax(1).cpu().squeeze()
            prob_e = logits_e.softmax(1).cpu()
            ent = -(prob_e * torch.log(prob_e + 1e-8)).sum(1).squeeze()
            pos = ((pred_full == lbl) & (pred_effi != lbl)).float()
            neg = ((pred_full != lbl) & (pred_effi == lbl)).float()
            net = pos - neg
            # np.quantile: torch.quantile errors on tensors > 2^24 elements
            edges = np.quantile(ent.flatten().numpy(), np.linspace(0, 1, 21))
            s_ent, s_net = [], []
            for b in range(20):
                q_lo, q_hi = float(edges[b]), float(edges[b + 1])
                mask = (ent >= q_lo) & (ent < q_hi)
                if mask.sum() > 100:
                    s_ent.append(ent[mask].mean().item()); s_net.append(net[mask].mean().item())
                    g_ent.append(s_ent[-1]); g_net.append(net[mask].mean().item())
                    g_pos.append(pos[mask].mean().item()); g_neg.append(neg[mask].mean().item())
            if len(s_ent) >= 5:
                subj_r_pearson.append(pearsonr(s_ent, s_net)[0])
                subj_r_spearman.append(spearmanr(s_ent, s_net)[0])
    if len(subj_r_pearson) == 0:
        print("[O5] no subject produced >=5 valid entropy bins; skipping"); return
    r_mean = float(np.mean(subj_r_pearson)); r_std = float(np.std(subj_r_pearson))
    rng = np.random.default_rng(0)
    boot = [np.mean(rng.choice(subj_r_pearson, len(subj_r_pearson), replace=True)) for _ in range(2000)]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    mean_pos, mean_neg = float(np.mean(g_pos)), float(np.mean(g_neg))
    print(f"[O5] subj Pearson r={r_mean:.3f}+/-{r_std:.3f}  95%CI[{ci_lo:.3f},{ci_hi:.3f}]")
    print(f"     mean pos_rate={mean_pos:.5f}  neg_rate={mean_neg:.5f}  "
          f"{'GO' if (ci_lo>0 and mean_pos>mean_neg) else 'NO-GO'}")
    idx = np.argsort(g_ent)
    xe = np.array(g_ent)[idx]
    plt.figure(figsize=(8, 5))
    plt.plot(xe, np.array(g_pos)[idx], "g--o", ms=3, alpha=.6, label="Positive rate")
    plt.plot(xe, np.array(g_neg)[idx], "r--o", ms=3, alpha=.6, label="Negative rate")
    plt.plot(xe, np.array(g_net)[idx], "b-o", ms=4,
             label=f"Net gain (subj-r={r_mean:.2f}[{ci_lo:.2f},{ci_hi:.2f}])")
    plt.axhline(0, color="k", lw=.8, ls=":")
    plt.xlabel("Mean entropy (bin)"); plt.ylabel("Rate"); plt.title("O5: Decoder Gain vs Uncertainty")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{OBS_DIR}/O5_decoder_gain.png", dpi=150); plt.close()
    save_obs("O5", {"subj_pearson_r_mean": r_mean, "subj_pearson_r_ci": [float(ci_lo), float(ci_hi)],
                    "mean_positive_rate": mean_pos, "mean_negative_rate": mean_neg,
                    "bin_ent": xe.tolist(), "bin_net": np.array(g_net)[idx].tolist(),
                    "go": bool(ci_lo > 0 and mean_pos > mean_neg)})


def O9_opportunity(val_loader, effi, full):
    rng = np.random.default_rng(0)
    budgets = np.array([5, 10, 20, 30, 50])
    ent_rec, rnd_rec = [], []
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].cuda()
            lbl = batch["label"].squeeze(1).long().cpu().squeeze()
            pred_full = infer(full, img).argmax(1).cpu().squeeze()
            logits_e = infer(effi, img)
            pred_effi = logits_e.argmax(1).cpu().squeeze()
            prob_e = logits_e.softmax(1).cpu()
            ent = -(prob_e * torch.log(prob_e + 1e-8)).sum(1).squeeze()
            pos = ((pred_full == lbl) & (pred_effi != lbl)).float()
            body = (lbl > 0) | (pred_full > 0) | (pred_effi > 0)
            eb = ent[body].numpy(); pb = pos[body].numpy()
            total = pb.sum()
            if total == 0:
                continue
            order = np.argsort(eb)[::-1]
            ent_rec.append([pb[order[:max(1, int(len(order) * q / 100))]].sum() / total for q in budgets])
            rnd_rec.append(np.mean([[pb[rng.choice(len(pb), max(1, int(len(pb) * q / 100)), replace=False)].sum() / total
                                     for q in budgets] for _ in range(100)], axis=0))
    ent_arr = np.asarray(ent_rec); rnd_arr = np.asarray(rnd_rec)
    if ent_arr.size == 0:
        print("[O9] no positive decoder transitions found; skipping"); return
    diffs = np.array([(ent_arr[rng.integers(len(ent_arr), size=len(ent_arr))]
                       - rnd_arr[rng.integers(len(rnd_arr), size=len(rnd_arr))]).mean(0) for _ in range(2000)])
    lo, hi = np.percentile(diffs, [2.5, 97.5], axis=0)
    print(f"[O9] budgets={budgets.tolist()}")
    print(f"     entropy_recovery={ent_arr.mean(0).round(3).tolist()}")
    print(f"     random_recovery ={rnd_arr.mean(0).round(3).tolist()}")
    print(f"     diff CI lower   ={lo.round(3).tolist()}")
    go = bool(lo[np.isin(budgets, [10, 20, 30])].max() > 0)
    print(f"     {'GO' if go else 'NO-GO'} (entropy beats random at 10-30% budget, CI_lo>0)")
    eb2, rb2 = [], []
    for _ in range(2000):
        i = rng.integers(len(ent_arr), size=len(ent_arr))
        eb2.append(ent_arr[i].mean(0)); rb2.append(rnd_arr[i].mean(0))
    elo, ehi = np.percentile(eb2, [2.5, 97.5], axis=0); rlo, rhi = np.percentile(rb2, [2.5, 97.5], axis=0)
    plt.figure(figsize=(7, 5))
    plt.plot(budgets, ent_arr.mean(0) * 100, "o-", color="steelblue", label="Entropy")
    plt.fill_between(budgets, elo * 100, ehi * 100, alpha=.2, color="steelblue")
    plt.plot(budgets, rnd_arr.mean(0) * 100, "o--", color="gray", label="Random")
    plt.fill_between(budgets, rlo * 100, rhi * 100, alpha=.15, color="gray")
    plt.xlabel("Selected union-foreground voxels (%)")
    plt.ylabel("Positive transitions recovered (%)"); plt.title("O9: Selective-Allocation Opportunity")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{OBS_DIR}/O9_opportunity.png", dpi=150); plt.close()
    save_obs("O9", {"budgets_pct": budgets.tolist(),
                    "entropy_recovery_mean": ent_arr.mean(0).round(4).tolist(),
                    "random_recovery_mean": rnd_arr.mean(0).round(4).tolist(),
                    "diff_ci_lower": lo.round(4).tolist(), "go": go})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/root/autodl-tmp/btcv-synapse")
    p.add_argument("--output", default="/root/output")
    p.add_argument("--dataset", default="BTCV13")
    args_ns = p.parse_args()
    set_determinism(seed=0)
    os.makedirs(OBS_DIR, exist_ok=True)
    device = torch.device("cuda:0")

    e0 = sorted(glob.glob(f"{args_ns.output}/E0*/3DUXNET/{args_ns.dataset}/best_metric_model.pth"))
    e1 = sorted(glob.glob(f"{args_ns.output}/E1*/3DUXNET_EffiDec3D/{args_ns.dataset}/best_metric_model.pth"))
    assert e0, f"No E0 checkpoint under {args_ns.output}/E0*/3DUXNET/{args_ns.dataset}/"
    assert e1, f"No E1 checkpoint under {args_ns.output}/E1*/3DUXNET_EffiDec3D/{args_ns.dataset}/"
    print(f"E0: {e0[-1]}\nE1: {e1[-1]}")
    full = load_ckpt(build_model("full", device), e0[-1], device)
    effi = load_ckpt(build_model("effi", device), e1[-1], device)

    ld_args = argparse.Namespace(root=args_ns.root, dataset=args_ns.dataset,
                                 mode="validation", crop_sample=4, img_size=[96, 96, 96])
    _, val_samples, _ = data_loader(ld_args)
    _, val_transform = data_transforms(ld_args)
    val_files = [{"image": im, "label": lb}
                 for im, lb in zip(val_samples["images"], val_samples["labels"])]
    val_loader = DataLoader(Dataset(data=val_files, transform=val_transform),
                            batch_size=1, shuffle=False, num_workers=2)
    post_pred = AsDiscrete(argmax=True, to_onehot=14)
    post_lbl = AsDiscrete(to_onehot=14)
    print(f"Val cases: {len(val_files)}\n")

    O2_entropy_distribution(val_loader, effi)
    O3_unc_error_corr(val_loader, effi)
    O4_per_organ(val_loader, effi, post_pred, post_lbl)
    O5_decoder_gain(val_loader, effi, full)
    O9_opportunity(val_loader, effi, full)
    print(f"\nDone. Figures + results.json in {OBS_DIR}")


if __name__ == "__main__":
    main()
