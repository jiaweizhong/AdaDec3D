#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 1 (motivation + definition): one validation case, six panels ---
image, ground truth, EffiDec3D prediction, full prediction, positive/negative
FLIP map, and efficient-model entropy. The caption should stress that the error
map, the uncertainty (entropy) map, and the flip map are three DIFFERENT things.

This is the qualitative teaser that run_observations.py does NOT produce.

Usage (from /root/AdaDec3D/EffiDec3D):
  python make_figure1.py --network 3DUXNET --dataset BTCV13 \
      --root /root/autodl-tmp/btcv-synapse --output /root/output \
      --e0_ckpt <full.pth> --e1_ckpt <effi.pth> \
      --case_idx 0 --axis 2 --out /root/obs-seed1/figure1_motivation.png
  # --slice omitted -> auto-pick the slice (along --axis) with the most flips.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch

from monai.data import DataLoader, Dataset
from monai.utils import set_determinism

from load_datasets_transforms import data_loader, data_transforms
from run_observations import (build_model, load_ckpt, infer,
                              DEFAULT_FEATURE_SIZE, FULL_NETWORK, EFFI_NETWORK)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/root/autodl-tmp/btcv-synapse")
    p.add_argument("--output", default="/root/output")
    p.add_argument("--dataset", default="BTCV13")
    p.add_argument("--network", default="3DUXNET")
    p.add_argument("--feature_size", type=int, default=None)
    p.add_argument("--e0_ckpt", required=True, help="full-model checkpoint")
    p.add_argument("--e1_ckpt", required=True, help="EffiDec3D checkpoint")
    p.add_argument("--case_idx", type=int, default=0, help="validation case index")
    p.add_argument("--axis", type=int, default=2, choices=[0, 1, 2],
                   help="slicing axis of the (D,H,W) volume")
    p.add_argument("--slice", type=int, default=None,
                   help="slice index; default = the slice with the most flips")
    p.add_argument("--zoom_half", type=int, default=24,
                   help="zoom half-window in voxels; crop is 2*zoom_half square "
                        "(16=tight 32^2, 24=48^2 default, 32=64^2 wide context)")
    p.add_argument("--out", default="/root/obs/figure1_motivation.png")
    p.add_argument("--skip_aggregation", default="concatenation",
                   choices=["addition", "concatenation"],
                   help="EffiDec3D skip aggregation; must match how --e1_ckpt was trained "
                        "(ignored for MedNeXt, which has no such knob)")
    args = p.parse_args()
    set_determinism(seed=0)
    device = torch.device("cuda:0")
    fsize = args.feature_size or DEFAULT_FEATURE_SIZE.get(args.network, 48)

    ld = argparse.Namespace(root=args.root, dataset=args.dataset, mode="validation",
                            crop_sample=4, img_size=[96, 96, 96])
    _, val_samples, out_classes = data_loader(ld)
    _, val_transform = data_transforms(ld)
    files = [{"image": im, "label": lb}
             for im, lb in zip(val_samples["images"], val_samples["labels"])]
    loader = DataLoader(Dataset(data=files, transform=val_transform),
                        batch_size=1, shuffle=False, num_workers=1)

    full = load_ckpt(build_model(args.network, "full", out_classes, device, feature_size=fsize),
                     args.e0_ckpt, device)
    effi = load_ckpt(build_model(args.network, "effi", out_classes, device, feature_size=fsize,
                                 skip_aggregation=args.skip_aggregation),
                     args.e1_ckpt, device)

    batch = next(b for i, b in enumerate(loader) if i == args.case_idx)
    img = batch["image"].cuda()
    lbl = batch["label"].squeeze(1).long().cpu().squeeze().numpy()
    with torch.no_grad():
        logits_e = infer(effi, img)
        pred_e = logits_e.argmax(1).cpu().squeeze().numpy()
        prob_e = logits_e.softmax(1).cpu()
        ent = (-(prob_e * torch.log(prob_e + 1e-8)).sum(1)).squeeze().numpy()
        pred_f = infer(full, img).argmax(1).cpu().squeeze().numpy()
    image = img.cpu().squeeze().numpy()

    pos = (pred_f == lbl) & (pred_e != lbl)          # positive flip
    neg = (pred_f != lbl) & (pred_e == lbl)          # negative flip

    # Pick the slice (along --axis) with the most flips; fall back to most foreground.
    def take(vol, k):
        return np.take(vol, k, axis=args.axis)
    if args.slice is None:
        counts = (pos | neg).sum(axis=tuple(a for a in range(3) if a != args.axis))
        k = int(counts.argmax()) if counts.max() > 0 else \
            int((lbl > 0).sum(axis=tuple(a for a in range(3) if a != args.axis)).argmax())
    else:
        k = args.slice

    im2, gt2 = take(image, k), take(lbl, k)
    pe2, pf2 = take(pred_e, k), take(pred_f, k)
    pos2, neg2, ent2 = take(pos, k), take(neg, k), take(ent, k)

    ncls = int(max(lbl.max(), pred_f.max(), pred_e.max())) + 1
    seg_cmap = ListedColormap(plt.cm.tab20(np.linspace(0, 1, max(ncls, 2))))
    seg_kw = dict(cmap=seg_cmap, vmin=0, vmax=max(ncls - 1, 1), interpolation="nearest")

    def seg_panel(ax, seg, title):
        ax.imshow(im2, cmap="gray")
        ax.imshow(np.ma.masked_where(seg == 0, seg), alpha=0.6, **seg_kw)
        ax.set_title(title); ax.axis("off")

    # flip-map RGBA (green = positive flip / full fixes, red = negative flip / full breaks)
    flip_rgb = np.zeros((*im2.shape, 4))
    flip_rgb[pos2] = [0.1, 0.8, 0.1, 0.9]
    flip_rgb[neg2] = [0.9, 0.1, 0.1, 0.9]

    # Locate one positive and one negative zoom region (largest connected flip cluster).
    from scipy.ndimage import label as _cc_label, center_of_mass
    half = args.zoom_half                            # zoom half-window (voxels); crop = 2*half square

    def _pick(mask):
        lab, n = _cc_label(mask)
        if n == 0:
            return None
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        cy, cx = center_of_mass(lab == int(sizes.argmax()))
        y0 = int(np.clip(cy - half, 0, im2.shape[0] - 2 * half))
        x0 = int(np.clip(cx - half, 0, im2.shape[1] - 2 * half))
        return y0, x0
    zoom_pos, zoom_neg = _pick(pos2), _pick(neg2)

    def _zoom(ax, box, label, color):
        if box is None:
            ax.text(0.5, 0.5, f"no {label}\non this slice", ha="center", va="center",
                    fontsize=9); ax.axis("off"); return
        y0, x0 = box; sl = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
        # interpolation="nearest": crisp voxel squares, no smoothing blur when upsampling
        # the ~32x32 crop (each square is one real voxel).
        ax.imshow(im2[sl], cmap="gray", interpolation="nearest")
        ax.imshow(flip_rgb[sl], interpolation="nearest")
        ax.contour(ent2[sl], levels=[np.percentile(ent2, 90)], colors="cyan", linewidths=0.8)
        ax.set_title(label, color=color, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(color); s.set_linewidth(2)

    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.05])
    ax_a = fig.add_subplot(gs[0, 0:2]); ax_b = fig.add_subplot(gs[0, 2:4]); ax_c = fig.add_subplot(gs[0, 4:6])
    ax_d = fig.add_subplot(gs[1, 0:2]); ax_e = fig.add_subplot(gs[1, 2:4]); ax_f = fig.add_subplot(gs[1, 4:6])
    ax_a.imshow(im2, cmap="gray"); ax_a.set_title("(a) image"); ax_a.axis("off")
    seg_panel(ax_b, gt2, "(b) ground truth")
    seg_panel(ax_c, pe2, "(c) Efficient decoder $E_1$ prediction")
    seg_panel(ax_d, pf2, "(d) full-decoder prediction")
    ax_e.imshow(im2, cmap="gray"); ax_e.imshow(flip_rgb)
    ax_e.set_title("(e) flip map: green=+, red=$-$"); ax_e.axis("off")
    # mark the two zoom windows on the flip map
    from matplotlib.patches import Rectangle
    for box, col in [(zoom_pos, "lime"), (zoom_neg, "red")]:
        if box is not None:
            y0, x0 = box
            ax_e.add_patch(Rectangle((x0, y0), 2 * half, 2 * half, fill=False,
                                     edgecolor=col, lw=1.5))
    im_ent = ax_f.imshow(ent2, cmap="magma")
    ax_f.set_title("(f) efficient-model entropy"); ax_f.axis("off")
    fig.colorbar(im_ent, ax=ax_f, fraction=0.046, pad=0.04)
    ax_g = fig.add_subplot(gs[2, 0:3]); ax_h = fig.add_subplot(gs[2, 3:6])
    _zoom(ax_g, zoom_pos, "(g) high entropy + positive flip", "green")
    _zoom(ax_h, zoom_neg, "(h) high entropy + negative flip", "firebrick")

    fig.suptitle(f"{args.network} / {args.dataset}  case {args.case_idx}  "
                 f"axis {args.axis} slice {k}  (error ≠ uncertainty ≠ flip)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=160); plt.close(fig)
    print(f"[figure1] pos flips={int(pos.sum())} neg flips={int(neg.sum())} "
          f"-> slice {k} (axis {args.axis}); saved {args.out}")


if __name__ == "__main__":
    main()
