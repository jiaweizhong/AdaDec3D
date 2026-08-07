#!/usr/bin/env python3
"""Regenerate paper figures directly from results/*/results.json -- no model, GPU, or
MONAI required (only numpy + matplotlib). Useful after cosmetic figure changes or to
rebuild the merged concentration panel that run_observations.py does not itself emit.

Produces, under elsarticle/figures/:
  * Figure 3  H1_global_flips.png   per backbone, R+/R-/R_net labels
  * Figure 5  O_anatomy.png         per backbone, de-cluttered size-vs-benefit scatter
  * Figure 6  concentration_merged.png   three backbones' oracle curves on one axis

Run from anywhere:  python EffiDec3D/regen_paper_figures.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO = Path(__file__).resolve().parent.parent          # .../AdaDec3D
RESULTS = REPO / "results"
FIGS = REPO / "elsarticle" / "figures"

# (results dir, paper figures dir, backbone label, color)
CELLS = [
    ("obs-uxnet-concat", "obs-uxnet-concat", "3D UX-Net",    "#4682b4"),
    ("obs-swin-concat",  "obs-swin-concat",  "SwinUNETR",    "#e08214"),
    ("obs-mednext-fs32", "obs-mednext",      "MedNeXt-M-K3", "#2e8b57"),
]


def load(cell_results):
    return json.load(open(RESULTS / cell_results / "results.json", encoding="utf-8"))


def regen_h1(r, out_png):
    o5 = r["O5"]
    mean_pos, mean_neg = o5["mean_positive_rate"], o5["mean_negative_rate"]
    net_mean, ci = o5["subject_net_rate_mean"], o5["subject_net_rate_ci"]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.yaxis.grid(True, ls=":", lw=.5, alpha=.6, zorder=0)
    ax.bar([0, 1, 2], [mean_pos, mean_neg, net_mean], width=0.5,
           color=["#2e8b57", "#b22222", "#4682b4"], alpha=.9,
           edgecolor="black", linewidth=0.6, zorder=3)
    ax.errorbar([2], [net_mean], yerr=[[net_mean - ci[0]], [ci[1] - net_mean]],
                fmt="none", ecolor="black", capsize=4, lw=1.2, zorder=4)
    ax.axhline(0, color="black", lw=.7, ls=":")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["$R_{+}$", "$R_{-}$", "$R_{\\mathrm{net}}$"])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("flip rate"); ax.set_title("H1: global decoder flip rates")
    fmt = mticker.ScalarFormatter(useMathText=True); fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def declutter(ax, xs, ys, labels, fig, fontsize=8):
    texts = []
    for x, y, lab in zip(xs, ys, labels):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        t = ax.annotate(lab, xy=(x, y), xytext=(7, 0), textcoords="offset points",
                        fontsize=fontsize, ha="left", va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.5, color="0.55",
                                        shrinkA=0, shrinkB=3))
        t.set_clip_on(False)
        texts.append(t)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for _ in range(400):
        bbs = [t.get_window_extent(r) for t in texts]
        moved = False
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if bbs[i].overlaps(bbs[j]):
                    hi, lo = (i, j) if bbs[i].y0 >= bbs[j].y0 else (j, i)
                    xh, yh = texts[hi].xyann; texts[hi].xyann = (xh, yh + 1.2)
                    xl, yl = texts[lo].xyann; texts[lo].xyann = (xl, yl - 1.2)
                    moved = True
        if not moved:
            break
        fig.canvas.draw()


def regen_anatomy(r, out_png):
    oa = r["O_anatomy"]
    names, pos_m, neg_m = oa["organ"], oa["positive_rate"], oa["negative_rate"]
    net_m, size, rho = oa["net_rate"], oa["organ_size"], oa["size_net_spearman"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    xs = np.arange(len(names))
    axs[0].bar(xs - 0.2, pos_m, 0.4, color="seagreen", alpha=.7, label="positive")
    axs[0].bar(xs + 0.2, neg_m, 0.4, color="firebrick", alpha=.7, label="negative")
    axs[0].plot(xs, net_m, "b-o", ms=3, label="net")
    axs[0].axhline(0, color="k", lw=.6, ls=":")
    axs[0].set_xticks(xs); axs[0].set_xticklabels(names, rotation=45, ha="right")
    axs[0].set_ylabel("flip rate (union fg)")
    axs[0].set_title("(a) Per-organ decoder benefit"); axs[0].legend()
    axs[1].scatter(size, net_m, s=32, color="steelblue", zorder=3,
                   edgecolor="white", linewidth=0.5)
    axs[1].set_xscale("log"); axs[1].axhline(0, color="k", lw=.6, ls=":")
    axs[1].set_xlabel("organ size (voxels, log)"); axs[1].set_ylabel("net flip rate")
    axs[1].set_title(f"(b) Size vs. benefit (Spearman ${rho:.2f}$)")
    finite = [s for s in size if np.isfinite(s) and s > 0]
    if finite:
        axs[1].set_xlim(min(finite) / 1.6, max(finite) * 2.4)
    ymin, ymax = min(net_m), max(net_m); pad = 0.12 * (ymax - ymin + 1e-9)
    axs[1].set_ylim(ymin - pad, ymax + pad)
    declutter(axs[1], size, net_m, names, fig, fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def regen_concentration_merged(out_png):
    """Three backbones' oracle concentration curves on one single-column axis. The curves
    are near-identical (all reach ~98% coverage at a 5% budget), so we distinguish them by
    colour + line style + marker and add a zoomed inset over the 5--10% region where they
    actually separate."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    styles = [("-", "o"), ("--", "s"), ("-.", "^")]
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    curves = []
    for (cell_results, _, label, color), (ls, mk) in zip(CELLS, styles):
        op = load(cell_results)["O_pareto"]
        bud = [0] + list(op["budgets_pct"])
        y = [0.0] + [v * 100 for v in op["oracle"]["net_recovered_mean"]]
        ax.plot(bud, y, ls=ls, marker=mk, ms=5, lw=1.8, color=color, label=label, zorder=3)
        curves.append((bud, y, color, ls, mk))
    ax.plot([0, 50], [0, 50], ls=":", color="gray", lw=1.2, label="random", zorder=2)
    ax.axhline(80, color="0.4", lw=.6, ls=":", zorder=1)
    ax.axvline(5, color="0.4", lw=.6, ls=":", zorder=1)
    ax.text(6, 72, r"$K_{80}\leq 5\%$", fontsize=9)
    ax.set_xlim(0, 50); ax.set_ylim(0, 103)
    ax.set_xlabel("Fraction of regions given the full decoder (%)")
    ax.set_ylabel("Positive net-gain covered (%)")
    ax.grid(True, ls=":", lw=.5, alpha=.5)
    ax.legend(loc="center right", fontsize=8, framealpha=.9)
    # zoomed inset on the 5--10% separation
    axins = inset_axes(ax, width="42%", height="40%", loc="lower right", borderpad=1.1)
    for bud, y, color, ls, mk in curves:
        axins.plot(bud, y, ls=ls, marker=mk, ms=5, lw=1.8, color=color)
    axins.set_xlim(4.4, 10.6); axins.set_ylim(96.8, 100.35)
    axins.grid(True, ls=":", lw=.4, alpha=.5)
    axins.tick_params(labelsize=7)
    axins.set_title("zoom: 5--10\\%", fontsize=7)
    mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.55", lw=0.8)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


if __name__ == "__main__":
    for cell_results, cell_figs, label, _ in CELLS:
        r = load(cell_results)
        regen_h1(r, FIGS / cell_figs / "H1_global_flips.png")
        regen_anatomy(r, FIGS / cell_figs / "O_anatomy.png")
        print(f"[{label}] H1 + O_anatomy -> figures/{cell_figs}/")
    regen_concentration_merged(FIGS / "concentration_merged.png")
    print("merged concentration -> figures/concentration_merged.png")
