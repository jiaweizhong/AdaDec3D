#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 4 (generalization): aggregate the headline flip metrics across the
L-shaped protocol's cells. Reads each cell's results.json and plots, on shared
axes, the O9 region net-flip utility at the predeclared budget (with 95% CI) and
the O5 subject-level entropy-net correlation, for the dataset axis (one backbone
across datasets) and the architecture axis (one dataset across backbones).

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


def _cell_metrics(res, budget):
    """Return (o9_net, o9_lo, o9_hi, o5_r, o5_lo, o5_hi) for one cell; NaN if absent."""
    nan = float("nan")
    o9n = o9lo = o9hi = nan
    o9 = res.get("O9_corrected")
    if o9 and "budgets_pct" in o9 and budget in o9["budgets_pct"]:
        i = o9["budgets_pct"].index(budget)
        reg = o9.get("region_selector", {})
        if reg.get("entropy_net_utility_mean"):
            o9n = reg["entropy_net_utility_mean"][i]
            o9lo = reg.get("entropy_net_utility_ci_lower", [nan] * (i + 1))[i]
            o9hi = reg.get("entropy_net_utility_ci_upper", [nan] * (i + 1))[i]
    o5 = res.get("O5", {})
    o5r = o5.get("subj_pearson_r_mean", nan)
    o5ci = o5.get("subj_pearson_r_ci", [nan, nan])
    return o9n, o9lo, o9hi, o5r, o5ci[0], o5ci[1]


def _panel(ax, cells, budget, title):
    labels, o9, lo, hi, r, rlo, rhi = [], [], [], [], [], [], []
    for tok in cells:
        label, _, path = tok.partition("=")
        try:
            m = _cell_metrics(_load(path), budget)
        except FileNotFoundError:
            m = (float("nan"),) * 6
        labels.append(label)
        o9.append(m[0] * 100 if m[0] == m[0] else np.nan)
        lo.append(m[1] * 100 if m[1] == m[1] else np.nan)
        hi.append(m[2] * 100 if m[2] == m[2] else np.nan)
        r.append(m[3]); rlo.append(m[4]); rhi.append(m[5])
    x = np.arange(len(labels))
    o9 = np.array(o9, float); lo = np.array(lo, float); hi = np.array(hi, float)
    yerr = np.abs(np.vstack([o9 - lo, hi - o9]))
    ax.bar(x, o9, 0.6, color="steelblue", alpha=.8,
           yerr=yerr, capsize=4, label=f"O9 net@{budget}% (%)")
    ax.axhline(0, color="k", lw=.6, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(f"net flip recovered @{budget}% (%)"); ax.set_title(title)
    ax2 = ax.twinx()
    r = np.array(r, float); rerr = np.abs(np.vstack([r - np.array(rlo, float), np.array(rhi, float) - r]))
    ax2.errorbar(x, r, yerr=rerr, fmt="D", color="darkorange", capsize=3, label="O5 subj $r(H,U)$")
    ax2.set_ylabel("subject $r(H,U)$", color="darkorange"); ax2.set_ylim(-0.2, 1.05)


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
