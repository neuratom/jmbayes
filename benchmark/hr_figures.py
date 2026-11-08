"""Hazard-ratio figures for the paper.

hr_other_data.png  forest plot, the two data sets other than PBC (aids, prothro): hazard
                   ratios in JMbayes2 and jmbayes, posterior mean +- 1.96 posterior SD of
                   the log hazard ratio (R's result files keep means and SDs only, so both
                   programs use the same interval), two seeds averaged
hr_scatter.png     every hazard ratio of the 21 further-feature models, jmbayes against
                   JMbayes2 (from <model>_equivalence.json, compare_cr.py)

Usage: python hr_figures.py
"""
import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE.parent / "paper" / "figures"
C_R, C_PY = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 (validated palette)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})

OTHER = {
    "aids": ("AIDS data (467 patients)",
             {"drugddI": "Didanosine vs zalcitabine", "prevOIAIDS": "Previous AIDS diagnosis",
              "value(CD4)": "sqrt(CD4) (per unit)"}),
    "prothro": ("Liver cirrhosis data (488 patients)",
                {"treatprednisone": "Prednisone vs placebo",
                 "value(pro)": "Prothrombin index (per unit)"}),
}


def _r_rows(run):
    """(name, mean, sd) of gammas and alphas from an R result file. R writes a block with
    one parameter without names, so names come from the Python file."""
    out = []
    for b in run["stats"]:
        if b["block"] in ("gammas", "alphas"):
            m, s = np.atleast_1d(b["Mean"]), np.atleast_1d(b["SD"])
            out += list(zip(m, s))
    return out


def _py_rows(run):
    return [(r["name"], r["Mean"], r["StDev"]) for k in ("gammas", "alphas") for r in run[k]]


def _load(model):
    R = [json.loads((HERE / f"R_{model}_seed{s}.json").read_text()) for s in (123, 456)]
    P = [json.loads((HERE / f"PY_{model}_seed{s}.json").read_text()) for s in (123, 456)]
    names = [n for n, _, _ in _py_rows(P[0])]
    r = np.mean([[(m, s) for m, s in _r_rows(x)] for x in R], 0)
    p = np.mean([[(m, s) for _, m, s in _py_rows(x)] for x in P], 0)
    return names, r, p


def fig_other():
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.6),
                             gridspec_kw=dict(height_ratios=[3, 2]))
    for ax, (model, (title, labels)) in zip(axes, OTHER.items()):
        names, r, p = _load(model)
        ys = np.arange(len(names))[::-1]
        for off, est, col, lab in ((0.14, r, C_R, "JMbayes2 (R)"),
                                   (-0.14, p, C_PY, "jmbayes (Python)")):
            for y, (m, s) in zip(ys, est):
                ax.plot([math.exp(m - 1.96 * s), math.exp(m + 1.96 * s)], [y + off] * 2,
                        color=col, lw=2, solid_capstyle="round")
                ax.plot(math.exp(m), y + off, "o", color=col, ms=6, mec="white", mew=1.5,
                        label=lab if y == ys[0] else None)
        ax.axvline(1, color=INK2, lw=0.8)
        ax.set_xscale("log")
        ticks = [0.5, 0.75, 1, 1.5, 2, 3]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(0.45, 3.4)
        ax.set_yticks(ys, [labels[n] for n in names])
        ax.set_ylim(ys[-1] - 0.5, ys[0] + 0.5)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
    axes[0].legend(frameon=False, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.32))
    axes[-1].set_xlabel("Hazard ratio (log scale), posterior mean and 95% interval")
    fig.tight_layout()
    fig.savefig(FIG / "hr_other_data.png", dpi=220)
    plt.close(fig)


def fig_scatter():
    hr = []
    for f in sorted(HERE.glob("*_equivalence.json")):
        if f.stem.split("_")[0] in ("cr",):          # the main paper's competing-risks run
            continue
        e = json.loads(f.read_text())
        if "hazard_ratios" in e:
            hr += [(h["R"], h["Python"]) for h in e["hazard_ratios"]]
    hr = np.array(hr)
    hr = hr[(hr > 0).all(1)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw=dict(width_ratios=[1, 1]))
    ax = axes[0]
    lo, hi = hr.min() / 1.5, hr.max() * 1.5
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, zorder=1)
    ax.scatter(hr[:, 0], hr[:, 1], s=14, color=C_PY, edgecolor="white", linewidth=0.6,
               zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Hazard ratio, JMbayes2")
    ax.set_ylabel("Hazard ratio, jmbayes")
    ax.set_title(f"{len(hr)} hazard ratios, 21 models", loc="left", fontsize=9, color=INK)
    ax = axes[1]
    rel = 100 * (hr[:, 1] / hr[:, 0] - 1)
    ax.hist(rel, bins=np.arange(-25, 25.5, 1), color=C_PY, edgecolor="white", linewidth=0.6)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("jmbayes vs JMbayes2, % difference")
    ax.set_ylabel("Hazard ratios")
    ax.grid(axis="x", visible=False)
    inside = np.mean(np.abs(rel) < 1) * 100
    ax.set_title(f"{inside:.0f}% within 1%", loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "hr_scatter.png", dpi=220)
    plt.close(fig)
    return hr, rel


if __name__ == "__main__":
    fig_other()
    hr, rel = fig_scatter()
    print(len(hr), "hazard ratios; |diff| < 1%:", np.mean(np.abs(rel) < 1),
          "< 5%:", np.mean(np.abs(rel) < 5), "max %:", np.abs(rel).max().round(1))
    big = np.argsort(-np.abs(rel))[:6]
    print("largest:", [(round(hr[i, 0], 3), round(hr[i, 1], 3), round(rel[i], 1)) for i in big])
