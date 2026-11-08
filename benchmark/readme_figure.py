"""Benchmark figure for the README: JMbayes2 (R) vs jmbayes (Python, GPU).

Left: more patients (3,500 iterations). Right: more iterations, one line per cohort size
(1,560 patients light, 3,120 patients full colour).
Top: MCMC time. Bottom: peak memory (RAM, and GPU memory for jmbayes).
With EXTRAPOLATE on, R's missing longest runs are extrapolated linearly and drawn dashed.
Input: scaling_results.jsonl (scaling.py). Output: figures/benchmarks.png
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
COLOR = {"R": ("#2a78d6", "#9cc3ef"), "Python-GPU": ("#eb6834", "#f5b493")}  # full, light
NAME = {"R": "JMbayes2", "Python-GPU": "jmbayes"}   # jmbayes CPU runs: report table only
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.facecolor": "white"})

EXTRAPOLATE = False            # R runs at 60k/90k are being measured, not guessed
EXTRAP = {"R"} if EXTRAPOLATE else set()
COHORTS = (1560, 3120)          # iteration sweeps; the largest is drawn in full colour
# timings known to be disturbed (the run overlapped with other work): time not plotted
BAD_TIME = {("R", 10, 60000)}
SC = {}
for line in open(HERE / "scaling_results.jsonl"):
    r = json.loads(line)
    SC[(r["impl"], r["k"], r["n_iter"])] = r          # last run of a job wins


def series(impl, patients=None):
    if patients is None:                               # patient sweep at 3,500 iterations
        rows = [(r["patients"], r) for (i, k, n), r in SC.items() if i == impl and n == 3500]
    else:
        rows = [(n, r) for (i, k, n), r in SC.items()
                if i == impl and r["patients"] == patients]
    return sorted(rows, key=lambda t: t[0])


def extrapolate(pts, key, x_new):
    (x1, r1), (x2, r2) = pts[-2], pts[-1]
    return r2[key] + (r2[key] - r1[key]) / (x2 - x1) * (x_new - x2)


def fmt(v, key, minutes):
    if key == "mcmc_s":
        return f"{v / 60:.0f} min" if minutes else f"{v:.0f} s"
    return f"{v:.2f} GB" if v < 1 else f"{v:.1f} GB"


def draw(ax, impl, pts, key, scale, minutes, color, lw, lab, x_max=None, note=True):
    if key == "mcmc_s":
        pts = [p for p in pts if (impl, p[1]["k"], p[1]["n_iter"]) not in BAD_TIME]
    x = [p[0] for p in pts]
    y = [p[1][key] for p in pts]
    ax.plot(x, [v / scale for v in y], "-o", color=color, lw=lw, ms=4.5, mec="white",
            mew=1.1, label=lab)
    end_x, end_y, est = x[-1], y[-1], False
    if x_max is not None and x[-1] < x_max and len(pts) >= 2:
        ye = extrapolate(pts, key, x_max)
        ax.plot([x[-1], x_max], [y[-1] / scale, ye / scale], "--", color=color, lw=lw * 0.8)
        ax.plot(x_max, ye / scale, "o", ms=6, mfc="white", mec=color, mew=1.6)
        end_x, end_y, est = x_max, ye, True
    if note:
        ax.annotate(("~" if est else "") + fmt(end_y, key, minutes)
                    + (" (extrapolated)" if est else ""),
                    (end_x, end_y / scale), textcoords="offset points",
                    xytext=(6, 9 if key == "peak_ram_gb" and impl == "Python-GPU" else 0),
                    fontsize=7.5, color=INK, va="center")
    if key == "peak_ram_gb" and impl == "Python-GPU":                 # GPU memory too
        g = [p[1]["gpu_peak_gb"] for p in pts]
        ax.plot(x, g, ":o", color=color, lw=lw * 0.8, ms=3, mec="white", mew=0.8,
                label=lab.replace("RAM", "GPU memory"))
        if note:
            ax.annotate(fmt(g[-1], key, minutes), (x[-1], g[-1]), textcoords="offset points",
                        xytext=(6, 1), fontsize=7.5, color=INK, va="center")


def main():
    rows = [("mcmc_s", "MCMC time"), ("peak_ram_gb", "Peak memory (GB)")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6))
    x_max = max(x for impl in COLOR for c in COHORTS for x, _ in series(impl, c))
    impls = [i for i in COLOR if series(i)]
    for i, (key, ylabel) in enumerate(rows):
        what = "" if key == "mcmc_s" else " RAM"
        # left: more patients
        ax = axes[i, 0]
        for impl in impls:
            draw(ax, impl, series(impl), key, 1, False, COLOR[impl][0], 2,
                 NAME[impl] + what)
        ax.set_ylabel("MCMC time (s)" if key == "mcmc_s" else ylabel)
        # right: more iterations, one line per cohort size
        ax = axes[i, 1]
        scale = 60 if key == "mcmc_s" else 1
        for impl in impls:
            for c in COHORTS:
                pts = series(impl, c)
                if not pts:
                    continue
                big = c == max(COHORTS)
                # label the big cohort; for R also any cohort that reaches further
                longest = max(series(impl, cc)[-1][0] for cc in COHORTS if series(impl, cc))
                draw(ax, impl, pts, key, scale, True, COLOR[impl][0 if big else 1],
                     2 if big else 1.4, f"{NAME[impl]}{what}, {c:,} patients",
                     x_max=x_max if impl in EXTRAP and (big or impl != "R") else None,
                     note=big or (impl == "R" and pts[-1][0] == longest))
        ax.set_ylabel("MCMC time (min)" if key == "mcmc_s" else ylabel)
        for j, a in enumerate(axes[i]):
            a.set_ylim(bottom=0)
            a.margins(x=0.2 if j else 0.16)
            a.legend(frameon=False, fontsize=7, loc="upper left" if j == 0 else ("center right" if i == 0 else "upper left"))
            a.xaxis.set_major_formatter(FuncFormatter(
                lambda v, _, j=j: f"{v / 1000:.0f}k" if j and v >= 1000 else f"{v:,.0f}"))
    axes[0, 0].set_title("More patients (3,500 iterations)", fontsize=10, color=INK, loc="left")
    axes[0, 1].set_title("More iterations", fontsize=10, color=INK, loc="left")
    axes[1, 0].set_xlabel("Patients")
    axes[1, 1].set_xlabel("Iterations per chain")
    fig.tight_layout()
    out = HERE / "figures" / "benchmarks.png"
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
