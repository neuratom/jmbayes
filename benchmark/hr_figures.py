"""Hazard-ratio figures for the paper. Every model is shown and compared the same way:
posterior means and SDs from two seeds per program, averaged.

forest.png      hazard ratios on the three data sets (PBC main model, AIDS, liver
                cirrhosis): posterior mean and 95% interval, mean +- 1.96 posterior SD of
                the log hazard ratio (R's result files keep means and SDs only, so both
                programs use the same interval)
hr_scatter.png  (a) every hazard ratio of every model, jmbayes against JMbayes2; (b) how
                close single runs are: JMbayes2 vs itself with another seed, and JMbayes2
                vs jmbayes

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

FOREST = [
    ("main", "(a) PBC (312 patients), main model",
     {"sexfemale": "Female sex", "value(log(serBilir))": "log serum bilirubin (per unit)",
      "value(prothrombin)": "Prothrombin time (per second)",
      "value(ascites)": "Ascites (per unit log-odds)"}),
    ("aids", "(b) AIDS (467 patients)",
     {"drugddI": "Didanosine vs zalcitabine", "prevOIAIDS": "Previous AIDS diagnosis",
      "value(CD4)": "sqrt(CD4) (per unit)"}),
    ("prothro", "(c) Liver cirrhosis (488 patients)",
     {"treatprednisone": "Prednisone vs placebo",
      "value(pro)": "Prothrombin index (per unit)"}),
]


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


def fig_forest():
    n = [len(lab) for _, _, lab in FOREST]
    fig, axes = plt.subplots(len(FOREST), 1, figsize=(7.2, 1.2 + 0.5 * sum(n)),
                             gridspec_kw=dict(height_ratios=n), sharex=True)
    for ax, (model, title, labels) in zip(axes, FOREST):
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
        ticks = [0.25, 0.5, 1, 2, 4]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(0.22, 4.4)
        ax.set_yticks(ys, [labels[n] for n in names])
        ax.set_ylim(ys[-1] - 0.5, ys[0] + 0.5)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="upper center", ncol=2)
    axes[-1].set_xlabel("Hazard ratio (log scale), posterior mean and 95% interval")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / fig.get_figheight()))
    fig.savefig(FIG / "forest.png", dpi=220)
    plt.close(fig)


def _single_runs():
    """|% difference| of each hazard ratio between single runs: JMbayes2 vs JMbayes2 with
    another seed, and JMbayes2 vs jmbayes with the same seed number."""
    def r_means(run):
        return np.array([x for b in run["stats"] if b["block"] in ("gammas", "alphas")
                         for x in np.atleast_1d(b["Mean"]).astype(float)])

    def p_means(run):
        return np.array([r["Mean"] for k in ("gammas", "alphas") for r in run[k]])
    rr, rp = [], []
    for key in MODELS:
        R = [r_means(json.loads((HERE / f"R_{key}_seed{s}.json").read_text()))
             for s in (123, 456)]
        P = [p_means(json.loads((HERE / f"PY_{key}_seed{s}.json").read_text()))
             for s in (123, 456)]
        rr += list(100 * np.abs(np.expm1(R[1] - R[0])))
        rp += [*100 * np.abs(np.expm1(P[0] - R[0])), *100 * np.abs(np.expm1(P[1] - R[1]))]
    return np.array(rr), np.array(rp)


MODELS = ["main", "cr", "surv", "fam", "tvc", "icens", "crx", "bystr", "rec", "fam2", "fam3", "f8a", "f8b",
          "f8c", "tva", "bhw", "bhp", "bhn", "ms", "wts", "pen", "aids", "prothro"]


def fig_scatter():
    hr = []
    for key in MODELS:
        e = json.loads((HERE / f"{key}_equivalence.json").read_text())
        hr += [(h["R"], h["Python"]) for h in e["hazard_ratios"]]
    hr = np.array(hr)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
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
    ax.set_title(f"(a) {len(hr)} hazard ratios, all {len(MODELS)} models", loc="left",
                 fontsize=9, color=INK)
    ax = axes[1]
    rr, rp = _single_runs()
    grid = np.logspace(-3, 2.3, 400)
    for x, col, lab in ((rr, C_R, "JMbayes2 vs itself (other seed)"),
                        (rp, C_PY, "JMbayes2 vs jmbayes")):
        ax.plot(grid, [np.mean(x <= g) * 100 for g in grid], color=col, lw=2, label=lab)
    ax.set_xscale("log")
    ticks = [0.01, 0.1, 1, 10, 100]
    ax.set_xticks(ticks, [f"{t:g}%" for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(1e-3, 200)
    ax.set_ylim(0, 101)
    ax.set_xlabel("Difference between two single runs (|%|)")
    ax.set_ylabel("Hazard ratios within it (%)")
    ax.set_title("(b) Agreement between runs", loc="left", fontsize=9, color=INK)
    ax.legend(frameon=False, loc="upper left", fontsize=8, handlelength=1.0, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "hr_scatter.png", dpi=220)
    plt.close(fig)
    return hr, rr, rp


if __name__ == "__main__":
    fig_forest()
    hr, rr, rp = fig_scatter()
    for nm, x in (("R vs R", rr), ("R vs jmbayes", rp)):
        print(nm, len(x), "within 1%:", np.mean(x < 1).round(2), "5%:",
              np.mean(x < 5).round(2), "median:", np.median(x).round(2))
