"""Benchmark figure for the README: one grid, JMbayes2 (R) vs jmbayes2py (Python, GPU).

Columns: more patients (3,500 iterations) | more iterations (1,560 patients) |
long runs (1,872 patients). Rows: MCMC time, peak RAM.
Input: scaling_results.jsonl (scaling.py). Output: figures/benchmarks.png
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
C_R, C_PY = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.facecolor": "white"})

SC = {}
for line in open(HERE / "scaling_results.jsonl"):
    r = json.loads(line)
    SC[(r["impl"], r["k"], r["n_iter"])] = r          # last run of a job wins

IMPLS = (("R", C_R, "JMbayes2 (R)"), ("Python-GPU", C_PY, "jmbayes2py (GPU)"))
COLS = [("More patients (3,500 iterations)", "Patients",
         lambda impl: sorted((r["patients"], r) for (i, k, n), r in SC.items()
                             if i == impl and n == 3500)),
        ("More iterations (1,560 patients)", "Iterations per chain",
         lambda impl: sorted((n, r) for (i, k, n), r in SC.items() if i == impl and k == 5)),
        ("Long runs (1,872 patients)", "Iterations per chain",
         lambda impl: sorted((n, r) for (i, k, n), r in SC.items() if i == impl and k == 6))]
ROWS = [("mcmc_s", "MCMC time"), ("peak_ram_gb", "Peak RAM (GB)")]


def fmt_time(s):
    return f"{s:.0f} s" if s < 120 else f"{s / 60:.0f} min"


def main():
    cols = [c for c in COLS if any(c[2](impl) for impl, _, _ in IMPLS)]
    fig, axes = plt.subplots(2, len(cols), figsize=(3.4 * len(cols), 5.4), squeeze=False)
    for j, (title, xlabel, get) in enumerate(cols):
        for i, (key, ylabel) in enumerate(ROWS):
            ax = axes[i, j]
            for impl, col, lab in IMPLS:
                pts = get(impl)
                if not pts:
                    continue
                x = [p[0] for p in pts]
                y = [p[1][key] / (60 if key == "mcmc_s" and j == 2 else 1) for p in pts]
                ax.plot(x, y, "-o", color=col, lw=2, ms=5, mec="white", mew=1.2, label=lab)
                last = pts[-1][1][key]
                ax.annotate(fmt_time(last) if key == "mcmc_s" else f"{last:.1f} GB",
                            (x[-1], y[-1]), textcoords="offset points", xytext=(5, 0),
                            fontsize=8, color=INK, va="center")
            ax.set_ylim(bottom=0)
            ax.margins(x=0.18)
            ax.xaxis.set_major_formatter(FuncFormatter(
                lambda v, _: f"{v / 1000:.0f}k" if v >= 1000 and xlabel.startswith("It")
                else f"{v:,.0f}"))
            if i == 0:
                ax.set_title(title, fontsize=9.5, color=INK, loc="left")
            if i == 1:
                ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel if key != "mcmc_s" else
                          ("MCMC time (min)" if j == 2 else "MCMC time (s)"))
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    out = HERE / "figures" / "benchmarks.png"
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
