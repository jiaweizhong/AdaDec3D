#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 4 (generalization): aggregate the two converged headline numbers across
the L-shaped protocol's cells. Reads each cell's results.json and plots, on shared
axes, the H1 global net flip rate (with subject 95% CI; near zero = bidirectional
cancellation) and the P1 direction AUROC (pos-vs-neg discrimination; near 0.5 =
uncertainty predicts instability, not improvement), for the dataset axis (one
backbone across datasets) and the architecture axis (one dataset across backbones).

Runs after the cells exist; no models needed.

Usage (from /root/AdaDec3D/EffiDec3D):
  python aggregate_generality.py \
    --dataset_cells "BTCV=/root/obs-seed1" "FeTA=/root/obs-uxnet-feta" "HepaticVessel=/root/obs-uxnet-hv" \
    --arch_cells "3D UX-Net=/root/obs-seed1" "SwinUNETR=/root/obs-swin" "MedNeXt=/root/obs-mednext" \
    --budget 20 --out /root/obs/figure4_generality.png
Each token is LABEL=PATH, where PATH is an obs dir (results.json appended) or a json file.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "results.json")
    with open(path) as f:
        return json.load(f)


def _cell_metrics(res, budget=None):
    """Return (net, net_lo, net_hi, dir_auroc, dir_lo, dir_hi) for one cell; NaN if
    absent. net = H1 global net flip rate (O5); dir_auroc = P1 pos-vs-neg AUROC (O3).
    ``budget`` is accepted for CLI compatibility but no longer used."""
    nan = float("nan")
    h1 = res.get("O5", {})
    net = h1.get("subject_net_rate_mean", nan)
    net_ci = h1.get("subject_net_rate_ci", [nan, nan])
    p1 = res.get("O3", {})
    dr = p1.get("pos_vs_neg_auroc_mean", nan)
    dr_ci = p1.get("pos_vs_neg_auroc_ci", [nan, nan])
    return net, net_ci[0], net_ci[1], dr, dr_ci[0], dr_ci[1]


def _panel(ax, cells, budget, title):
    labels, net, lo, hi, r, rlo, rhi = [], [], [], [], [], [], []
    for tok in cells:
        label, _, path = tok.partition("=")
        try:
            m = _cell_metrics(_load(path), budget)
        except FileNotFoundError:
            m = (float("nan"),) * 6
        labels.append(label)
        net.append(m[0] * 100 if m[0] == m[0] else np.nan)
        lo.append(m[1] * 100 if m[1] == m[1] else np.nan)
        hi.append(m[2] * 100 if m[2] == m[2] else np.nan)
        r.append(m[3]); rlo.append(m[4]); rhi.append(m[5])
    x = np.arange(len(labels))
    net = np.array(net, float); lo = np.array(lo, float); hi = np.array(hi, float)
    yerr = np.abs(np.vstack([net - lo, hi - net]))
    ax.bar(x, net, 0.6, color="steelblue", alpha=.8,
           yerr=yerr, capsize=4, label="H1 net flip rate (%)")
    ax.axhline(0, color="k", lw=.6, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("global net flip rate (%)"); ax.set_title(title)
    ax2 = ax.twinx()
    r = np.array(r, float); rerr = np.abs(np.vstack([r - np.array(rlo, float), np.array(rhi, float) - r]))
    ax2.errorbar(x, r, yerr=rerr, fmt="D", color="darkorange", capsize=3,
                 label="P1 direction AUROC")
    ax2.axhline(0.5, color="darkorange", lw=.6, ls=":")
    ax2.set_ylabel("direction AUROC (pos-vs-neg)", color="darkorange"); ax2.set_ylim(0.3, 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_cells", nargs="+", default=[], help="LABEL=PATH per dataset (fixed backbone)")
    p.add_argument("--arch_cells", nargs="+", default=[], help="LABEL=PATH per backbone (fixed dataset)")
    p.add_argument("--budget", type=int, default=20)
    p.add_argument("--out", default="/root/obs/figure4_generality.png")
    args = p.parse_args()
    n = int(bool(args.dataset_cells)) + int(bool(args.arch_cells))
    if n == 0:
        raise SystemExit("provide --dataset_cells and/or --arch_cells")
    fig, axs = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)
    col = 0
    if args.dataset_cells:
        _panel(axs[0, col], args.dataset_cells, args.budget,
               "Dataset axis (fixed backbone)"); col += 1
    if args.arch_cells:
        _panel(axs[0, col], args.arch_cells, args.budget,
               "Architecture axis (fixed dataset)")
    fig.suptitle("Generalization of the decoder-flip finding across the L-shape")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150); plt.close(fig)
    print(f"[figure4] saved {args.out}")


if __name__ == "__main__":
    main()
