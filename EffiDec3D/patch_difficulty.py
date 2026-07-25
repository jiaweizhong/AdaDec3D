#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch-difficulty distribution for the patch-level whole-model adaptivity study.

During sliding-window inference the model runs the FULL encoder+decoder on every
96^3 patch, including many trivially-easy (background / homogeneous) patches.
This script tiles each validation volume into 96^3 patches and measures, per patch:
  - fg_frac  : foreground voxel fraction (label > 0)          [cheap, label-only]
  - img_std  : intensity std of the patch                     [cheap, image-only, pre-forward]
  - entropy  : mean predictive entropy from E1                [needs model, --with-model]

It answers two questions:
  1. What fraction of patches are trivially easy? (compute headroom for a cheap path)
  2. Can a CHEAP pre-forward signal (img_std / fg_frac) predict patch difficulty
     (entropy)? -> feasibility of gating the encoder before running it.

Usage (from /root/AdaDec3D/EffiDec3D):
  python patch_difficulty.py --root /root/autodl-tmp/btcv-synapse            # cheap signals only (CPU-ok)
  python patch_difficulty.py --root ... --with-model --e1 "/root/output/E1*/3DUXNET_EffiDec3D/BTCV13/best_metric_model.pth"
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from load_datasets_transforms import data_loader, data_transforms

PATCH = 96
STRIDE = 48   # 50% overlap, approximates the sliding-window patch population


def iter_patches(vol_shape, patch=PATCH, stride=STRIDE):
    D, H, W = vol_shape
    zs = list(range(0, max(1, D - patch + 1), stride)) or [0]
    ys = list(range(0, max(1, H - patch + 1), stride)) or [0]
    xs = list(range(0, max(1, W - patch + 1), stride)) or [0]
    # ensure last patch reaches the border
    if zs[-1] + patch < D: zs.append(D - patch)
    if ys[-1] + patch < H: ys.append(H - patch)
    if xs[-1] + patch < W: xs.append(W - patch)
    for z in zs:
        for y in ys:
            for x in xs:
                yield z, y, x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/root/autodl-tmp/btcv-synapse")
    p.add_argument("--dataset", default="BTCV13")
    p.add_argument("--with-model", action="store_true", help="also compute E1 patch entropy (needs GPU)")
    p.add_argument("--e1", default="/root/output/E1*/3DUXNET_EffiDec3D/BTCV13/best_metric_model.pth")
    p.add_argument("--empty-thr", type=float, default=0.005, help="fg_frac below this = 'easy' patch")
    p.add_argument("--out", default="/root/obs/patch_difficulty.json")
    args = p.parse_args()
    set_determinism(seed=0)

    ld = argparse.Namespace(root=args.root, dataset=args.dataset, mode="validation",
                            crop_sample=4, img_size=[PATCH, PATCH, PATCH])
    _, val_samples, _ = data_loader(ld)
    _, val_transform = data_transforms(ld)
    files = [{"image": im, "label": lb}
             for im, lb in zip(val_samples["images"], val_samples["labels"])]
    loader = DataLoader(Dataset(data=files, transform=val_transform),
                        batch_size=1, shuffle=False, num_workers=2)

    model = None
    if args.with_model:
        from networks.UXNet_3D.network_backbone import UXNET_EffiDec3D
        from monai_utils.inferers.utils import sliding_window_inference_1out  # noqa (kept for parity)
        device = torch.device("cuda:0")
        ck = sorted(glob.glob(args.e1))
        assert ck, f"No E1 checkpoint at {args.e1}"
        model = UXNET_EffiDec3D(
            in_chans=1, out_chans=14, depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384],
            n_decoder_channels=48, drop_path_rate=0, layer_scale_init_value=1e-6,
            spatial_dims=3, skip_aggregation="addition", resolution_factor=2).to(device).eval()
        st = torch.load(ck[-1], map_location=device)
        model.load_state_dict(st.get("model_state_dict", st) if isinstance(st, dict) else st)
        print(f"E1: {ck[-1]}")

    fg_fracs, img_stds, ents = [], [], []
    n_patches = 0
    with torch.no_grad():
        for batch in loader:
            img = batch["image"][0, 0]      # [D,H,W]
            lbl = batch["label"][0, 0]
            D, H, W = img.shape
            for (z, y, x) in iter_patches((D, H, W)):
                ip = img[z:z+PATCH, y:y+PATCH, x:x+PATCH]
                lp = lbl[z:z+PATCH, y:y+PATCH, x:x+PATCH]
                fg_fracs.append(float((lp > 0).float().mean()))
                img_stds.append(float(ip.std()))
                n_patches += 1
                if model is not None:
                    patch = ip.unsqueeze(0).unsqueeze(0).cuda()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logit = model(patch)
                    prob = logit.softmax(1).float().cpu()
                    e = -(prob * torch.log(prob + 1e-8)).sum(1).mean().item()
                    ents.append(e)

    fg = np.array(fg_fracs); istd = np.array(img_stds)
    easy = float((fg < args.empty_thr).mean())
    print(f"\nTotal 96^3 patches: {n_patches}  (stride {STRIDE})")
    print(f"'Easy' patches (fg_frac < {args.empty_thr}): {easy:.1%}")
    print(f"fg_frac  pcts: " + ", ".join(f"p{p}={np.percentile(fg,p):.3f}" for p in [50,75,90,95]))
    res = {"n_patches": n_patches, "stride": STRIDE, "empty_thr": args.empty_thr,
           "easy_patch_frac": easy,
           "fg_frac_pcts": {p: float(np.percentile(fg, p)) for p in [50,75,90,95,99]}}

    if ents:
        ent = np.array(ents)
        r_std = pearsonr(istd, ent)[0]; r_fg = pearsonr(fg, ent)[0]
        rs_std = spearmanr(istd, ent)[0]; rs_fg = spearmanr(fg, ent)[0]
        low_ent = float((ent < np.percentile(ent, 25)).mean())
        print(f"\nPatch entropy (E1):  median={np.median(ent):.4f}")
        print(f"cheap-signal predictive power for difficulty (entropy):")
        print(f"  img_std : Pearson {r_std:.3f}  Spearman {rs_std:.3f}")
        print(f"  fg_frac : Pearson {r_fg:.3f}  Spearman {rs_fg:.3f}")
        print(f"=> if a cheap gate skips/downsizes the {easy:.0%} easy patches, "
              f"that fraction of full encoder+decoder compute is saved.")
        res.update({"entropy_median": float(np.median(ent)),
                    "corr_imgstd_entropy": {"pearson": float(r_std), "spearman": float(rs_std)},
                    "corr_fgfrac_entropy": {"pearson": float(r_fg), "spearman": float(rs_fg)}})
    else:
        print("\n(cheap signals only; re-run with --with-model for entropy correlation)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
