"""Regenerate ``figures/Figura5.png`` and ``figures/Figura6.png``.

Both figures are drawn from ``results/`` and therefore change only when the
simulation is re-run.  House style, applied to both: 300 dpi, ~7.0 x 4.4 in,
9 pt type, no figure title (the caption lives in the manuscript), a
colour-blind-safe palette, and marker *and* line-style coding so the figures
survive greyscale printing.

Usage::

    python plot_figures.py [--results results] [--out ../figures]
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from maimo import carbon as cb
from maimo.config import DEFAULT
from maimo.stats import summarise, t_critical

# Okabe-Ito, the standard colour-blind-safe qualitative palette.
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73",
             "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

RC = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "0.6",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "lines.linewidth": 1.2,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

FIGSIZE = (7.0, 4.4)

# Marker and line style per policy, chosen so that no two adjacent points in
# the Pareto plot share a glyph.
STYLE: Dict[str, Tuple[str, str]] = {
    "B1": ("o", "-"), "B2": ("s", "--"), "B3": ("^", "-."),
    "B4": ("v", ":"), "B5": ("D", "-"), "B6": ("P", "--"),
    "B7": ("X", "-."), "B8": ("<", ":"), "B9": (">", "-"),
    "B10": ("*", "-"),
}
COLOUR: Dict[str, str] = {
    "B1": OKABE_ITO[6], "B2": OKABE_ITO[3], "B3": OKABE_ITO[2],
    "B4": OKABE_ITO[1], "B5": OKABE_ITO[7], "B6": OKABE_ITO[5],
    "B7": OKABE_ITO[4], "B8": "#7F7F7F", "B9": OKABE_ITO[1],
    "B10": OKABE_ITO[0],
}


def load(results: str, pid: str) -> dict:
    with open(f"{results}/{pid}.json", encoding="utf-8") as f:
        return json.load(f)


def stat(rec: dict, metric: str, tc: float):
    return summarise(np.asarray(rec[metric], dtype=float), tc)


def pareto_front(x: Sequence[float], y: Sequence[float],
                 keep: Sequence[bool] | None = None) -> List[int]:
    """Indices of the non-dominated points when *both* axes are minimised.

    ``keep`` restricts the frontier to a feasible subset; the excluded points
    are still plotted, they simply cannot define the frontier.
    """
    order = [i for i in np.argsort(x) if keep is None or keep[i]]
    front, best = [], np.inf
    for i in order:
        if y[i] < best - 1e-12:
            front.append(int(i))
            best = y[i]
    return front


# ---------------------------------------------------------------------------
# Figure 5: latency-energy Pareto frontier
# ---------------------------------------------------------------------------
def figure5(results: str, out: str) -> str:
    """Latency-energy plane with the frontier of the quality-feasible set.

    Taken on their own, latency and energy are both minimised by the
    device-only baseline, which would make the two-dimensional frontier a
    single point and would say nothing useful.  What separates the policies is
    that device-only buys its position by answering with a 50 M INT4 model.
    The figure therefore draws the frontier over the policies that meet the
    deployment's task-success floor and plots the rest as hollow markers, so a
    reader can see both the trade-off and the price of leaving the constraint
    out.  This is the honest reading required by the CONTRACT: MAIMO is a
    Pareto-optimal operating point, not a universal dominator.
    """
    tc = t_critical(DEFAULT.n_seeds)
    floor = DEFAULT.accuracy_floor_pct
    ids = [f"B{i}" for i in range(1, 11)]
    recs = {p: load(results, p) for p in ids}
    lat = {p: stat(recs[p], "latency_mean_ms", tc) for p in ids}
    ene = {p: stat(recs[p], "energy_j", tc) for p in ids}
    acc = {p: stat(recs[p], "accuracy_pct", tc) for p in ids}

    x = np.array([lat[p].mean for p in ids])
    y = np.array([ene[p].mean for p in ids])
    ok = np.array([acc[p].mean >= floor for p in ids])
    front = pareto_front(x, y, ok)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_yscale("log")

    if len(front) > 1:
        ax.plot(x[front], y[front], color="0.35", linestyle="-",
                linewidth=1.1, zorder=1)

    handles = []
    for i, p in enumerate(ids):
        mk, _ = STYLE[p]
        is_maimo = p == "B10"
        face = COLOUR[p] if ok[i] else "white"
        ax.errorbar(x[i], y[i], xerr=lat[p].ci95, yerr=ene[p].ci95,
                    fmt=mk, markerfacecolor=face, color=COLOUR[p],
                    markersize=14 if is_maimo else 6.5,
                    markeredgecolor="black" if is_maimo else COLOUR[p],
                    markeredgewidth=1.2 if is_maimo else 0.9,
                    ecolor=COLOUR[p], elinewidth=0.9, capsize=2.0,
                    zorder=5 if is_maimo else 3)
        handles.append(Line2D(
            [], [], color=COLOUR[p], marker=mk, linestyle="none",
            markerfacecolor=face, markersize=10 if is_maimo else 5.5,
            markeredgecolor="black" if is_maimo else COLOUR[p],
            markeredgewidth=1.2 if is_maimo else 0.9,
            label=f"{p} {recs[p]['name']}"))
    handles += [
        Line2D([], [], color="0.35", linestyle="-",
               label=f"frontier, task success $\\geq$ {floor:.0f} %"),
        Line2D([], [], color="0.35", marker="o", linestyle="none",
               markerfacecolor="white", markersize=5.5,
               label=f"hollow: below the {floor:.0f} % floor"),
    ]

    i10, i2 = ids.index("B10"), ids.index("B2")
    ax.annotate("MAIMO", xy=(x[i10], y[i10]),
                xytext=(x[i10] - 3.4, y[i10] * 4.2),
                arrowprops=dict(arrowstyle="->", lw=0.9, color="black"),
                fontsize=9.5, fontweight="bold")

    # The CONTRACT requires the figure itself to make the edge-only point.
    ax.annotate(f"edge-only: {y[i2]:.2f} J vs MAIMO's {y[i10]:.2f} J.\n"
                f"MAIMO sends 25 % of the load to the\n"
                f"cloud, lifting task success from\n"
                f"{acc['B2'].mean:.1f} % to {acc['B10'].mean:.1f} %.",
                xy=(x[i2], y[i2] * 0.82),
                xytext=(0.025, 0.42), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3",
                                connectionstyle="arc3,rad=-0.18"),
                fontsize=7.4, color="0.15", va="bottom")

    ax.set_xlabel("Mean end-to-end latency (ms)")
    ax.set_ylabel("Energy per inference (J, log scale)")
    ax.set_xlim(x.min() - 1.6, x.max() + 2.2)
    ax.set_ylim(min(y) * 0.35, max(y) * 3.4)
    ax.legend(handles=handles, loc="lower right", ncol=2,
              handletextpad=0.35, columnspacing=0.9, borderpad=0.5,
              labelspacing=0.26)

    # The orchestration policies sit in one tight cluster; without a zoom the
    # frontier and the CI bars are unreadable.
    zoom = [i for i, p in enumerate(ids) if p not in ("B1", "B2", "B3", "B6")
            and 0.6 * y[ids.index("B10")] < y[i] < 1.8 * y[ids.index("B10")]]
    if len(zoom) >= 3:
        zx, zy = x[zoom], y[zoom]
        mx = 0.10 * (zx.max() - zx.min() + 1e-9)
        ins = ax.inset_axes([0.545, 0.315, 0.435, 0.355])
        if len(front) > 1:
            ins.plot(x[front], y[front], color="0.35", linewidth=1.1,
                     zorder=1)
        for i in zoom:
            p = ids[i]
            mk, _ = STYLE[p]
            is_maimo = p == "B10"
            ins.errorbar(x[i], y[i], xerr=lat[p].ci95, yerr=ene[p].ci95,
                         fmt=mk, color=COLOUR[p],
                         markerfacecolor=COLOUR[p] if ok[i] else "white",
                         markersize=12 if is_maimo else 5.5,
                         markeredgecolor="black" if is_maimo else COLOUR[p],
                         markeredgewidth=1.1 if is_maimo else 0.9,
                         ecolor=COLOUR[p], elinewidth=0.8, capsize=1.8,
                         zorder=5 if is_maimo else 3)
            ins.annotate(p, xy=(x[i], y[i]), xytext=(3.0, 3.5),
                         textcoords="offset points", fontsize=6.4,
                         fontweight="bold" if is_maimo else "normal")
        ins.set_xlim(zx.min() - mx - 0.25, zx.max() + mx + 0.35)
        ins.set_ylim(zy.min() * 0.93, zy.max() * 1.07)
        ins.tick_params(labelsize=6.2, length=2.5, pad=1.5)
        ins.grid(alpha=0.2, linewidth=0.4)
        ins.set_title("zoom: orchestration policies (ms, J)", fontsize=6.8,
                      pad=2.0)
        for s in ins.spines.values():
            s.set_linewidth(0.7)

    path = os.path.join(out, "Figura5.png")
    with plt.rc_context(RC):
        fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 6: carbon
# ---------------------------------------------------------------------------
STRATEGY_ORDER = ["No shifting", "Temporal shifting", "Geographic shifting",
                  "Temporal + geographic"]


def figure6(results: str, out: str) -> str:
    tc = t_critical(DEFAULT.n_seeds)
    with open(f"{results}/carbon.json", encoding="utf-8") as f:
        car = json.load(f)
    vals = car["policies"]["B10"]["values"]
    ideal = car["policies"]["B10"]["idealised"]

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.06, 1.0],
                          height_ratios=[1.5, 1.0], hspace=0.55, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a2 = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[:, 1])

    # -- panel (a): intensity traces + strategy bars -------------------
    hours = np.arange(25.0)
    ls = ["-", "--", "-.", ":"]
    for i, (region, trace) in enumerate(car["region_traces"].items()):
        y = np.array(trace + trace[:1])
        tag = ""
        if region == car["host_region"]:
            tag = " (host)"
        elif region == car["clean_region"]:
            tag = " (cleanest)"
        ax_a.plot(hours, y, ls[i % 4], color=OKABE_ITO[[6, 1, 5, 3][i % 4]],
                  marker=["o", "s", "^", "D"][i % 4], markevery=4,
                  markersize=3.2, label=f"{region}{tag}")
    ax_a.set_xlabel("Hour of day (local time)")
    ax_a.set_ylabel("Intensity\n(g CO$_2$e/kWh)", fontsize=8)
    ax_a.set_xlim(0, 24)
    ax_a.set_xticks(np.arange(0, 25, 6))
    ax_a.legend(loc="upper left", ncol=2, handletextpad=0.4,
                columnspacing=0.9, fontsize=7)
    ax_a.set_ylim(0, 620)
    ax_a.set_title("(a) Grid carbon intensity of the regions used",
                   fontsize=9, loc="left")

    means = [summarise(vals[s], tc) for s in STRATEGY_ORDER]
    ax_a2.bar(np.arange(4), [m.mean for m in means],
              yerr=[m.ci95 for m in means],
              color=[OKABE_ITO[6], OKABE_ITO[2], OKABE_ITO[3], OKABE_ITO[5]],
              edgecolor="black", linewidth=0.5, capsize=2.5, width=0.68)
    ax_a2.set_xticks(np.arange(4))
    ax_a2.set_xticklabels(["none", "temporal", "geographic", "both"],
                          fontsize=7.5)
    ax_a2.set_ylabel("g CO$_2$e per\n1000 inferences", fontsize=8)
    ax_a2.set_ylim(0, max(m.mean + m.ci95 for m in means) * 1.22)
    ax_a2.tick_params(axis="y", labelsize=7.5)
    ax_a2.set_title("MAIMO carbon under each shifting strategy",
                    fontsize=8, loc="left")

    # -- panel (b): the 89 % claim, decomposed -------------------------
    base = summarise(vals["No shifting"], tc)
    both = summarise(vals["Temporal + geographic"], tc)
    ideal_s = summarise(vals["Idealised full migration"], tc)
    names = ["No shifting\n(reference)", "Realistic\nconstrained\n(temp.+geo.)",
             "Idealised full\nmigration\nUPPER BOUND"]
    mm = [base, both, ideal_s]
    colours = [OKABE_ITO[6], OKABE_ITO[3], OKABE_ITO[2]]
    bars = ax_b.bar(np.arange(3), [m.mean for m in mm],
                    yerr=[m.ci95 for m in mm], capsize=2.5,
                    color=colours, edgecolor="black", linewidth=0.6,
                    width=0.62)
    bars[2].set_hatch("//")
    for i, m in enumerate(mm):
        red = 100.0 * (base.mean - m.mean) / base.mean
        txt = "reference" if i == 0 else f"-{red:.0f} %"
        ax_b.text(i, m.mean + m.ci95 + 0.045 * base.mean, txt,
                  ha="center", fontsize=8,
                  fontweight="bold" if i == 2 else "normal")
    ax_b.set_xticks(np.arange(3))
    ax_b.set_xticklabels(names, fontsize=7.5)
    ax_b.set_ylabel("Operational carbon (g CO$_2$e per 1000 inferences)")
    ax_b.set_ylim(0, base.mean * 1.62)
    ax_b.set_title("(b) Decomposition of the reported reduction",
                   fontsize=9, loc="left")
    ax_b.legend(handles=[
        matplotlib.patches.Patch(facecolor=OKABE_ITO[2], hatch="//",
                                 edgecolor="black", linewidth=0.6,
                                 label="idealised UPPER BOUND: 100 % of the\n"
                                       f"load migrated to {car['clean_region']}"
                                       " at its daily\nminimum; ignores "
                                       "migration latency, egress\nenergy and "
                                       "data sovereignty. Not achieved.")],
        loc="upper center", fontsize=6.6, handlelength=1.3,
        borderpad=0.4, handletextpad=0.5)

    path = os.path.join(out, "Figura6.png")
    with plt.rc_context(RC):
        fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=os.path.join("..", "figures"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with plt.rc_context(RC):
        for p in (figure5(args.results, args.out),
                  figure6(args.results, args.out)):
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
