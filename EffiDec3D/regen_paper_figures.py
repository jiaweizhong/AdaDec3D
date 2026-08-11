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
DPI = 200                                               # crisper than the 150 obs default

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
    ax.set_xticklabels(["$R_{+}$", "$R_{-}$", "$R_{\\mathrm{net}}$"], fontsize=15)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("flip rate", fontsize=13); ax.set_title("H1: global decoder flip rates", fontsize=13)
    ax.tick_params(axis="y", labelsize=11)
    fmt = mticker.ScalarFormatter(useMathText=True); fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.get_offset_text().set_fontsize(11)
    fig.tight_layout(); fig.savefig(out_png, dpi=DPI); plt.close(fig)


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
    axs[0].set_xticks(xs); axs[0].set_xticklabels(names, rotation=45, ha="right", fontsize=10)
    axs[0].set_ylabel("flip rate (union fg)", fontsize=12)
    axs[0].tick_params(axis="y", labelsize=10)
    axs[0].set_title("(a) Per-organ decoder benefit", fontsize=13); axs[0].legend(fontsize=11)
    axs[1].scatter(size, net_m, s=38, color="steelblue", zorder=3,
                   edgecolor="white", linewidth=0.5)
    axs[1].set_xscale("log"); axs[1].axhline(0, color="k", lw=.6, ls=":")
    axs[1].set_xlabel("organ size (voxels, log)", fontsize=12)
    axs[1].set_ylabel("net flip rate", fontsize=12)
    axs[1].tick_params(labelsize=10)
    axs[1].set_title(f"(b) Size vs. benefit (Spearman ${rho:.2f}$)", fontsize=13)
    finite = [s for s in size if np.isfinite(s) and s > 0]
    if finite:
        axs[1].set_xlim(min(finite) / 1.6, max(finite) * 2.4)
    ymin, ymax = min(net_m), max(net_m); pad = 0.12 * (ymax - ymin + 1e-9)
    axs[1].set_ylim(ymin - pad, ymax + pad)
    declutter(axs[1], size, net_m, names, fig, fontsize=10)
    fig.tight_layout(); fig.savefig(out_png, dpi=DPI); plt.close(fig)


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
    ax.set_xlabel("Fraction of regions given the full decoder (%)", fontsize=12)
    ax.set_ylabel("Positive net-gain covered (%)", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.grid(True, ls=":", lw=.5, alpha=.5)
    ax.legend(loc="center", bbox_to_anchor=(0.63, 0.70), fontsize=10, framealpha=.95)
    # zoomed inset on the 5--10% separation
    axins = inset_axes(ax, width="42%", height="38%", loc="lower right", borderpad=1.1)
    for bud, y, color, ls, mk in curves:
        axins.plot(bud, y, ls=ls, marker=mk, ms=5, lw=1.8, color=color)
    axins.set_xlim(4.4, 10.6); axins.set_ylim(96.8, 100.35)
    axins.grid(True, ls=":", lw=.4, alpha=.5)
    axins.tick_params(labelsize=7)
    axins.set_title("zoom: 5-10%", fontsize=7)
    mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.55", lw=0.8)
    fig.tight_layout(); fig.savefig(out_png, dpi=DPI); plt.close(fig)


def regen_recovery_merged(out_png):
    """Figure 8: entropy- vs random-guided macro-Dice gap recovery for all three backbones
    on one single-column axis, normalized to percent of the full-minus-efficient Dice gap.
    Solid = entropy-guided routing (colour per backbone); dashed = random baseline."""
    from matplotlib.lines import Line2D
    markers = ["o", "s", "^"]
    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    for (cell_results, _, label, color), mk in zip(CELLS, markers):
        rr = load(cell_results)["R_recovery"]
        effi, full = rr["effi_dice"], rr["full_dice"]
        gap = full - effi
        bud = [0] + list(rr["budgets_pct"])
        ent = [0.0] + [(d - effi) / gap * 100 for d in rr["entropy"]["dice_mean"]]
        rnd = [0.0] + [(d - effi) / gap * 100 for d in rr["random"]["dice_mean"]]
        ax.plot(bud, ent, "-", marker=mk, ms=5, lw=2.0, color=color, zorder=3)
        ax.plot(bud, rnd, "--", marker=mk, ms=4, lw=1.4, color=color, alpha=.55, zorder=2)
    ax.axhline(100, color="0.35", ls=":", lw=.9)
    ax.text(0.6, 103, "full decoder (100% gap)", fontsize=9, color="0.3")
    ax.axvline(20, color="0.6", ls=":", lw=.7)
    ax.set_xlim(0, 30); ax.set_ylim(0, 118)
    ax.set_xlabel("Spatial budget routed to full decoder (%)", fontsize=12)
    ax.set_ylabel("Macro-Dice gap recovered (%)", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.grid(True, ls=":", lw=.5, alpha=.5)
    handles = [Line2D([0], [0], color=c, lw=2.2, marker=m)
               for (_, _, _, c), m in zip(CELLS, markers)]
    handles += [Line2D([0], [0], color="0.45", lw=2, ls="-"),
                Line2D([0], [0], color="0.45", lw=1.5, ls="--")]
    labels = [lab for _, _, lab, _ in CELLS] + ["entropy-guided", "random"]
    ax.legend(handles, labels, fontsize=11, loc="center right", framealpha=.95)
    fig.tight_layout(); fig.savefig(out_png, dpi=DPI); plt.close(fig)


def regen_h1_merged(out_png):
    """Figure 3: global flip rates for all three backbones in one grouped-bar panel.
    Grouped by quantity (R+, R-, R_net); one bar per backbone. R_net carries the
    subject-bootstrap 95% CI, making the net-neutrality (CI crosses 0) visible against
    the much larger bidirectional activity (R+ ~ R-)."""
    cats = ["$R_{+}$", "$R_{-}$", "$R_{\\mathrm{net}}$"]
    x = np.arange(3); width = 0.26
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.yaxis.grid(True, ls=":", lw=.5, alpha=.6, zorder=0)
    for i, (cell_results, _, label, color) in enumerate(CELLS):
        o5 = load(cell_results)["O5"]
        net = o5["subject_net_rate_mean"]; ci = o5["subject_net_rate_ci"]
        vals = [o5["mean_positive_rate"], o5["mean_negative_rate"], net]
        off = (i - 1) * width
        ax.bar(x + off, vals, width, color=color, label=label,
               edgecolor="black", linewidth=0.5, zorder=3)
        ax.errorbar(x[2] + off, net, yerr=[[net - ci[0]], [ci[1] - net]],
                    fmt="none", ecolor="black", capsize=3, lw=1.0, zorder=4)
    ax.axhline(0, color="black", lw=.7, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=15)
    ax.set_ylabel("flip rate", fontsize=13); ax.tick_params(axis="y", labelsize=11)
    fmt = mticker.ScalarFormatter(useMathText=True); fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt); ax.yaxis.get_offset_text().set_fontsize(11)
    ax.set_title("Global decoder flip rates: activity vs. net gain", fontsize=12)
    ax.legend(fontsize=10, framealpha=.95)
    fig.tight_layout(); fig.savefig(out_png, dpi=DPI); plt.close(fig)


def regen_boundary_merged(out_png):
    """Figure 6: boundary-resolved decoder benefit for all three backbones on one figure,
    split into the paper's two quantities -- (a) net gain and (b) activity (P+N) -- versus
    distance to the ground-truth boundary, arranged vertically (2 rows x 1 col) for single-column layout."""
    markers = ["o", "s", "^"]
    fig, axs = plt.subplots(2, 1, figsize=(4.0, 4.0), sharex=True)
    bins = None
    for (cell_results, _, label, color), mk in zip(CELLS, markers):
        hb = load(cell_results)["H2_boundary"]
        bins = hb["distance_bins"]
        x = np.arange(len(bins))
        netf = hb["net_rate"]; ci = hb["net_rate_ci"]
        net = [v * 100 for v in netf]
        yerr = [[(netf[i] - ci[i][0]) * 100 for i in range(len(x))],
                [(ci[i][1] - netf[i]) * 100 for i in range(len(x))]]
        act = [(p + q) * 100 for p, q in zip(hb["positive_rate"], hb["negative_rate"])]
        axs[0].errorbar(x, net, yerr=yerr, marker=mk, ms=4, lw=1.5, color=color,
                        capsize=2.5, label=label, zorder=3)
        axs[1].plot(x, act, marker=mk, ms=4, lw=1.5, color=color, label=label, zorder=3)
    axs[0].axhline(0, color="k", lw=.7, ls=":")
    axs[0].set_title("(a) Net gain  $R_{+}\\!-\\!R_{-}$", fontsize=9.5, pad=3)
    axs[1].set_title("(b) Activity  $R_{+}\\!+\\!R_{-}$", fontsize=9.5, pad=3)
    axs[0].set_ylabel("rate (%)", fontsize=9)
    axs[1].set_ylabel("rate (%)", fontsize=9)
    for ax in axs:
        ax.set_xticks(np.arange(len(bins))); ax.set_xticklabels(bins, fontsize=9)
        ax.tick_params(labelsize=8, pad=2)
        ax.grid(True, ls=":", lw=.5, alpha=.5)
    axs[1].set_xlabel("distance to GT boundary (voxels)", fontsize=9, labelpad=3)
    axs[0].legend(fontsize=8, framealpha=.95, loc="upper right")
    fig.tight_layout(pad=0.3); fig.savefig(out_png, dpi=DPI); plt.close(fig)


if __name__ == "__main__":
    for cell_results, cell_figs, label, _ in CELLS:
        r = load(cell_results)
        regen_h1(r, FIGS / cell_figs / "H1_global_flips.png")
        regen_anatomy(r, FIGS / cell_figs / "O_anatomy.png")
        print(f"[{label}] H1 + O_anatomy -> figures/{cell_figs}/")
    regen_concentration_merged(FIGS / "concentration_merged.png")
    print("merged concentration -> figures/concentration_merged.png")
    regen_recovery_merged(FIGS / "recovery_merged.png")
    print("merged recovery -> figures/recovery_merged.png")
    regen_boundary_merged(FIGS / "boundary_merged.png")
    print("merged boundary -> figures/boundary_merged.png")
    regen_h1_merged(FIGS / "h1_merged.png")
    print("merged H1 -> figures/h1_merged.png")
