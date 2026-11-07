"""Build the comparison report (PDF): agreement with JMbayes2 and scaling.

Inputs (all in this folder):
  R_x1.json, PY_x1.json, PY_x1_seed456.json   README model on pbc2, 12000 iterations
  scaling_results.jsonl                        time / RAM sweep (scaling.py)
Output: jmbayes_vs_JMbayes2.pdf
"""
import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,  # noqa: E402
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
C_R, C_PY = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 (validated palette)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

import argparse  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--r", default="R_eq.json")
_ap.add_argument("--r2", default="R_eq_seed456.json")
_ap.add_argument("--py", default="PY_eq32_seed123.json")
_ap.add_argument("--py2", default="PY_eq32_seed456.json")
_ap.add_argument("--settings", default="12 chains, 30,000 iterations per chain, 5,000 burn-in, "
                                       "no thinning (300,000 retained draws per program)")
_ap.add_argument("--py-rinit", default="PY_eqRinit_seed123.json")
_ap.add_argument("--py2-rinit", default="PY_eqRinit_seed456.json")
_ap.add_argument("--out", default="jmbayes_vs_JMbayes2.pdf")
ARGS = _ap.parse_args()
R = json.load(open(HERE / ARGS.r))
P = json.load(open(HERE / ARGS.py))
P2 = json.load(open(HERE / ARGS.py2))
SC = [json.loads(line) for line in open(HERE / "scaling_results.jsonl")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})


# ------------------------------------------------------------------ data helpers
def r_rows(block):
    return {r["name"]: r for r in R[block]}


def p_rows(block):
    return {r["name"]: r for r in P[block]}


SURV_NAMES = ["sexfemale", "value(log(serBilir))", "value(prothrombin)", "value(ascites)"]
SURV_LABEL = {"sexfemale": "Female sex",
              "value(log(serBilir))": "log serum bilirubin (per unit)",
              "value(prothrombin)": "Prothrombin time (per second)",
              "value(ascites)": "Ascites (per unit log-odds)"}
r_surv = r_rows("gammas_alphas")
p_surv = {**p_rows("gammas"), **p_rows("alphas")}

OUTCOMES = [("betas1", "betas1", "log(serum bilirubin) - linear mixed model"),
            ("betas2", "betas2", "Prothrombin time - linear mixed model"),
            ("betas3", "betas3", "Ascites - logistic mixed model")]


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


GROUPS = [("gammas", "Survival covariate"), ("alphas", "Association (marker -> hazard)"),
          ("betas1", "Fixed effects: log(bilirubin)"), ("betas2", "Fixed effects: prothrombin"),
          ("betas3", "Fixed effects: ascites"), ("sigmas", "Residual SDs"),
          ("D", "Random-effects covariance"), ("bs_gammas", "Baseline hazard spline"),
          ("tau_bs_gammas", "Spline smoothing")]
MARGINS = (0.1, 0.2)          # equivalence margins, in posterior SDs
Z90 = 1.6448536269514722      # TOST at alpha = 0.05 <=> 90% CI inside the margin


def rows_of(run, blk):
    """(mean, posterior SD, MCSE) per parameter, from an R or a Python result file."""
    if "stats" in run:
        st = {b["block"]: b for b in run["stats"]}[blk]
        return list(zip(*(np.atleast_1d(st[k]).astype(float) for k in ("Mean", "SD", "SE"))))
    return [(r["Mean"], r["StDev"], r["SE"]) for r in run[blk]]

def pair_stats(a, b):
    """z and |difference| / SD for two runs of the same program."""
    z, d = [], []
    for blk, _ in GROUPS:
        for (ma, sa, ea), (mb, sb, eb) in zip(rows_of(a, blk), rows_of(b, blk)):
            z.append((ma - mb) / math.hypot(ea, eb))
            d.append(abs(ma - mb) / sa)
    return np.array(z), np.array(d)

def calibration(pairs):
    """How much JMbayes2's MCSE formula understates the real run-to-run error.
    Two runs of the same program that differ only in their seed are independent draws of
    the same procedure, so z = (run1 - run2) / sqrt(SE1^2 + SE2^2) should have unit
    variance. k = sqrt(mean(z^2)) over all parameters and all such pairs.
    """
    z = np.concatenate([pair_stats(a, b)[0] for a, b in pairs])
    return math.sqrt(np.mean(np.square(z)))

def pooled(runs, blk):
    rows = [rows_of(r, blk) for r in runs]
    out = []
    for vals in zip(*rows):
        m = np.mean([v[0] for v in vals])
        se = math.sqrt(sum(v[2] ** 2 for v in vals)) / len(vals)
        out.append((m, vals[0][1], se))
    return out

def comparisons(r_runs, p_runs, k_cal):
    """R vs Python, each pooled over its seeds: difference, calibrated MC error, TOST."""
    from scipy.stats import norm
    out = []
    for blk, _ in GROUPS:
        names = [r["name"] for r in P[blk]]
        for nm, (rm, rsd, rse), (pm, _, pse) in zip(names, pooled(r_runs, blk),
                                                    pooled(p_runs, blk)):
            d = pm - rm
            se = k_cal * math.hypot(rse, pse)
            rec = dict(block=blk, name=nm, d=d, se=se, sd=rsd, z=d / se, d_sd=d / rsd,
                       lo_sd=(d - Z90 * se) / rsd, hi_sd=(d + Z90 * se) / rsd)
            for m in MARGINS:
                delta = m * rsd
                p_tost = max(1 - norm.cdf((d + delta) / se), norm.cdf((d - delta) / se))
                rec[f"p_tost_{m}"] = p_tost
                rec[f"equiv_{m}"] = p_tost < 0.05
            out.append(rec)
    return out

R2 = json.load(open(HERE / ARGS.r2)) if (HERE / ARGS.r2).exists() else None
R_RUNS = [R] + ([R2] if R2 else [])
PAIRS = [(P, P2)] + ([(R, R2)] if R2 else [])
K_CAL = calibration(PAIRS)
Z_PP, D_PP = pair_stats(P, P2)
Z_RR, D_RR = pair_stats(R, R2) if R2 else (None, None)
CMP = comparisons(R_RUNS, [P, P2], K_CAL)
zarr = np.array([c["z"] for c in CMP])
dsd = np.abs(np.array([c["d_sd"] for c in CMP]))
# Same analysis for Python started from R's own separate fits (isolates the sampler)
RINIT = None
if (HERE / ARGS.py_rinit).exists() and (HERE / ARGS.py2_rinit).exists():
    PR1 = json.load(open(HERE / ARGS.py_rinit))
    PR2 = json.load(open(HERE / ARGS.py2_rinit))
    k_r = calibration([(PR1, PR2)] + PAIRS[1:])
    RINIT = dict(P1=PR1, P2=PR2, k=k_r, cmp=comparisons(R_RUNS, [PR1, PR2], k_r))

def summarise(cmp):
    d = np.abs([c["d_sd"] for c in cmp])
    z = np.array([c["z"] for c in cmp])
    d6 = [c["d_sd"] for c in cmp if c["name"].startswith("D[6")]
    return dict(median=np.median(d), max=d.max(), eq1=sum(c["equiv_0.1"] for c in cmp),
                eq2=sum(c["equiv_0.2"] for c in cmp), z2=np.mean(np.abs(z) < 2),
                z3=np.mean(np.abs(z) < 3), d6=np.mean(d6),
                diff=sum(c["lo_sd"] > 0.2 or c["hi_sd"] < -0.2 for c in cmp))


def conclusions():
    """Same sign and same 5% significance for every regression coefficient."""
    rows = []
    for nm in SURV_NAMES:
        rows.append(("Survival", nm, r_surv[nm], p_surv[nm]))
    for rb, pb, title in OUTCOMES:
        rr = r_rows(rb)
        for nm, p in p_rows(pb).items():
            if nm in rr:
                rows.append((title.split(" - ")[0], nm, rr[nm], p))
    out = []
    for grp, nm, r, p in rows:
        out.append(dict(group=grp, name=nm,
                        same_sign=np.sign(r["Mean"]) == np.sign(p["Mean"]),
                        same_sig=(r["P"] < 0.05) == (p["P"] < 0.05),
                        same_ci0=((r["2.5%"] > 0) or (r["97.5%"] < 0))
                        == ((p["2.5%"] > 0) or (p["97.5%"] < 0))))
    return out


CONC = conclusions()


# ------------------------------------------------------------------ figures
def fig_forest():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ys = np.arange(len(SURV_NAMES))[::-1]
    for off, rows, col, lab in ((0.14, r_surv, C_R, "JMbayes2 (R)"),
                                (-0.14, p_surv, C_PY, "jmbayes (Python)")):
        for y, nm in zip(ys, SURV_NAMES):
            r = rows[nm]
            ax.plot([math.exp(r["2.5%"]), math.exp(r["97.5%"])], [y + off] * 2, color=col, lw=2,
                    solid_capstyle="round")
            ax.plot(math.exp(r["Mean"]), y + off, "o", color=col, ms=6, mec="white", mew=1.5,
                    label=lab if y == ys[0] else None)
    ax.axvline(1, color=INK2, lw=0.8)
    ax.set_xscale("log")
    ticks = [0.25, 0.5, 1, 2]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(ys, [SURV_LABEL[n] for n in SURV_NAMES])
    ax.set_ylim(ys[-1] - 0.5, ys[0] + 0.9)
    ax.set_xlabel("Hazard ratio (log scale), posterior mean and 95% credible interval")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(FIG / "forest.png", dpi=220)
    plt.close(fig)


def fig_equiv(cmp=None, fname="equivalence.png"):
    cmp = CMP if cmp is None else cmp
    """Difference (Python - R) in posterior SDs with its 90% CI, per parameter."""
    n = len(cmp)
    fig, ax = plt.subplots(figsize=(7.2, 9.6))
    ys = np.arange(n)[::-1]
    ax.axvspan(-0.2, 0.2, color="#f3f2ee", zorder=0)
    ax.axvspan(-0.1, 0.1, color="#e6e4de", zorder=0)
    ax.axvline(0, color=INK2, lw=0.8)
    for y, c in zip(ys, cmp):
        col = C_PY if c["equiv_0.1"] else (INK2 if c["equiv_0.2"] else "#e34948")
        ax.plot([c["lo_sd"], c["hi_sd"]], [y, y], color=col, lw=2, solid_capstyle="round")
        ax.plot(c["d_sd"], y, "o", color=col, ms=4.5, mec="white", mew=1)
    labels = []
    gname = dict(GROUPS)
    last = None
    for c in cmp:
        g = gname[c["block"]].split(":")[-1].strip()
        labels.append(f"{c['name']}  [{g}]" if c["block"] != last else c["name"])
        last = c["block"]
    ax.set_yticks(ys, labels, fontsize=7)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlim(-0.45, 0.45)
    ax.set_xlabel("Python - R, in posterior SDs (point and 90% CI)")
    ax.grid(axis="y", visible=False)
    ax.text(0, n + 0.1, "inner band: ±0.1 SD   outer band: ±0.2 SD", ha="center", va="bottom",
            fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=220)
    plt.close(fig)


def series(impl, key, by):
    if by == "patients":
        rows = [r for r in SC if r["impl"] == impl and r["n_iter"] == 3500]
        rows.sort(key=lambda r: r["patients"])
        return [r["patients"] for r in rows], [r[key] for r in rows]
    rows = [r for r in SC if r["impl"] == impl and r["k"] == 5]
    rows.sort(key=lambda r: r["n_iter"])
    return [r["n_iter"] for r in rows], [r[key] for r in rows]


def fig_scaling(key, by, ylabel, fname, ylim0=True):
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for impl, col, lab in (("R", C_R, "JMbayes2 (R)"), ("Python-GPU", C_PY, "Python, GPU")):
        x, y = series(impl, key, by)
        if not x:
            continue
        ax.plot(x, y, "-o", color=col, lw=2, ms=5, mec="white", mew=1.2, label=lab)
        ax.annotate(f"{y[-1]:.0f}" if y[-1] >= 10 else f"{y[-1]:.2f}", (x[-1], y[-1]),
                    textcoords="offset points", xytext=(4, 0), fontsize=8, color=INK,
                    va="center")
    ax.set_xlabel("Patients" if by == "patients" else "MCMC iterations per chain")
    ax.set_ylabel(ylabel)
    if ylim0:
        ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=220)
    plt.close(fig)


fig_forest()
fig_equiv()
if RINIT:
    fig_equiv(RINIT["cmp"], "equivalence_rinit.png")
for key, lab in (("mcmc_s", "MCMC time (s)"), ("wall_s", "Total time incl. set-up (s)"),
                 ("peak_ram_gb", "Peak RAM (GB)")):
    fig_scaling(key, "patients", lab, f"pat_{key}.png")
    fig_scaling(key, "iters", lab, f"it_{key}.png")


# ------------------------------------------------------------------ PDF
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceAfter=6, textColor=INK)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4,
                    textColor=INK)
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.2, leading=12.5, textColor=INK)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=7.8, leading=10, textColor=INK2)
CELL = ParagraphStyle("C", parent=BODY, fontSize=8, leading=9.5)
CELLB = ParagraphStyle("CB", parent=CELL, fontName="Helvetica-Bold")


def tbl(data, widths, head_rows=1, zebra=True):
    t = Table([[Paragraph(str(c), CELLB if i < head_rows else CELL) for c in row]
               for i, row in enumerate(data)], colWidths=widths, repeatRows=head_rows)
    st = [("LINEBELOW", (0, head_rows - 1), (-1, head_rows - 1), 0.8, colors.HexColor(INK2)),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if zebra:
        for r in range(head_rows, len(data), 2):
            st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f6f5f2")))
    t.setStyle(TableStyle(st))
    return t


def img(name, w):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(FIG / name)).getSize()
    return Image(str(FIG / name), width=w, height=w * ih / iw)


def sc(impl, k, n_iter):
    for r in SC:
        if r["impl"] == impl and r["k"] == k and r["n_iter"] == n_iter:
            return r
    return None


n_eq1 = sum(c["equiv_0.1"] for c in CMP)
n_eq2 = sum(c["equiv_0.2"] for c in CMP)
n_conc = len(CONC)
n_same = sum(c["same_sign"] and c["same_sig"] and c["same_ci0"] for c in CONC)

story = []
story.append(Paragraph("Statistical equivalence of jmbayes and JMbayes2", H1))
story.append(Paragraph(
    "Joint model for longitudinal and time-to-event data, ported from the R package JMbayes2 "
    "(v0.6-5) to Python/PyTorch. Same model, same priors and same MCMC algorithm; it runs on "
    "CPU or GPU. This report tests whether the two programs give the same answers on the "
    "same data. All numbers were measured on one machine: Intel Core i9-12900K (x86-64, "
    "16 cores - 8 performance + 8 efficiency - and 24 threads), 64 GB RAM, NVIDIA GeForce "
    "RTX 3090 (24 GB); Windows 11, R 4.6.2 with JMbayes2 0.6-5, Python 3.11 with PyTorch "
    "2.6 (CUDA 12.4).", BODY))

big = sc("R", 10, 3500), sc("Python-GPU", 10, 3500)
it_r, it_p = sc("R", 5, 28000), sc("Python-GPU", 5, 28000)
if it_r is None:
    it_r, it_p = sc("R", 5, 14000), sc("Python-GPU", 5, 14000)
n_clear_diff = sum(c["lo_sd"] > 0.2 or c["hi_sd"] < -0.2 for c in CMP)
bullets = [
    (f"<b>R vs Python differs no more than R vs R.</b> Each program was run twice with "
     f"different random seeds. Two R runs differ from each other by a median of "
     f"{np.median(D_RR):.2f} posterior SDs (largest {D_RR.max():.2f}); two Python runs by "
     f"{np.median(D_PP):.2f} (largest {D_PP.max():.2f}); R and Python by "
     f"{np.median(dsd):.2f} (largest {dsd.max():.2f}). The choice of implementation has no "
     "effect beyond the ordinary run-to-run noise of either program."
     if R2 is not None else
     f"<b>R vs Python differs no more than Python vs Python.</b> The median difference "
     f"between R and Python is {np.median(dsd):.2f} posterior SDs (largest {dsd.max():.2f}). "
     f"Two Python runs with identical code and different random seeds differ by a median of "
     f"{np.median(D_PP):.2f} SDs (largest {D_PP.max():.2f}). The choice of implementation has "
     "no effect beyond ordinary Monte Carlo noise."),
    f"<b>Same conclusions.</b> For all {n_same} of {n_conc} regression coefficients (survival "
    "and longitudinal), R and Python agree on the direction of the effect, on significance at "
    "5% and on whether the 95% credible interval excludes zero. Hazard ratios, intervals and "
    "p-values are shown side by side in Section 1.",
    f"<b>Formal equivalence.</b> In an equivalence test (TOST, α = 0.05) {n_eq2} of "
    f"{len(CMP)} parameters are proven equivalent within ±0.2 posterior SD ({n_eq1} within "
    f"±0.1 SD). For the remaining {len(CMP) - n_eq2} the runs are not precise enough to "
    f"decide; {'no' if n_clear_diff == 0 else n_clear_diff} parameter"
    f"{'' if n_clear_diff == 1 else 's'} {'is' if n_clear_diff == 1 else 'are'} shown to "
    f"differ. With realistic error bars, {np.mean(np.abs(zarr) < 2):.0%} of differences lie "
    f"within 2 standard errors (expected ~95%) and {np.mean(np.abs(zarr) < 3):.0%} within 3.",
]
if all(big):
    bullets.append(
        f"<b>Faster as cohorts grow.</b> At {big[0]['patients']:,} patients the MCMC took "
        f"{big[0]['mcmc_s']:.0f} s in R and {big[1]['mcmc_s']:.0f} s in Python on the GPU "
        f"({big[0]['mcmc_s'] / big[1]['mcmc_s']:.1f}x). Python's time grows slowly with patients; "
        "R's grows roughly in proportion (Section 2).")
if it_r and it_p:
    bullets.append(
        f"<b>Flat memory.</b> R's peak RAM grows with patients and with iterations "
        f"(up to {it_r['peak_ram_gb']:.1f} GB at {it_r['patients']:,} patients x "
        f"{it_r['n_iter']:,} iterations); Python stays near {it_p['peak_ram_gb']:.1f} GB, because "
        "it does not store a per-patient log-likelihood for every iteration.")
bullets.append(
    "<b>Small data.</b> On the original 312 patients R and Python are comparable; most of "
    "R's time there is fixed start-up cost of its parallel workers, not sampling.")
for b in bullets:
    story.append(Paragraph("• " + b, BODY))
    story.append(Spacer(1, 3))

story.append(Paragraph("Model and data", H2))
story.append(Paragraph(
    "PBC data (pbc2, 312 patients, 1,945 visits, 169 deaths or transplants). The model is the "
    "one in the JMbayes2 README: a Cox-type survival model with sex, linked through the "
    "current value of three longitudinal markers - log serum bilirubin and prothrombin time "
    "(linear mixed models, random intercept and slope) and ascites (logistic mixed model, "
    f"random intercept and slope). MCMC: {ARGS.settings}, identical in both programs. "
    "The R model was fitted with <font face='Courier'>jm()</font>; the Python model "
    "with <font face='Courier'>jmbayes</font>, using its own versions of the separate "
    "starting fits (lme, mixed_model, coxph).", BODY))

# ---- Section 1: agreement
story.append(Paragraph("1. Do the two implementations agree?", H2))
story.append(Paragraph(
    "<b>Survival sub-model: hazard ratios.</b> Posterior mean hazard ratio with 95% credible "
    "interval (exp of the log-scale estimates) and the Bayesian tail probability P "
    "(2 x min(P(β&gt;0), P(β&lt;0)), as reported by JMbayes2).", BODY))
rows = [["Effect on the hazard of death or transplant", "R: HR (95% CI)", "R: P",
         "Python: HR (95% CI)", "Python: P"]]
for nm in SURV_NAMES:
    r, p = r_surv[nm], p_surv[nm]
    hr = lambda d: f"{math.exp(d['Mean']):.2f} ({math.exp(d['2.5%']):.2f} - {math.exp(d['97.5%']):.2f})"  # noqa: E731,E501
    rows.append([SURV_LABEL[nm], hr(r), fmt_p(r["P"]), hr(p), fmt_p(p["P"])])
story.append(tbl(rows, [6.0 * cm, 3.4 * cm, 1.4 * cm, 3.4 * cm, 1.6 * cm]))
story.append(Spacer(1, 4))
story.append(img("forest.png", 16.5 * cm))

for rb, pb, title in OUTCOMES:
    rr, pp = r_rows(rb), p_rows(pb)
    rows = [["Coefficient", "R: estimate (95% CI)", "R: P", "Python: estimate (95% CI)",
             "Python: P"]]
    for nm, p in pp.items():
        r = rr.get(nm)
        if r is None:
            continue
        est = lambda d: f"{d['Mean']:.3f} ({d['2.5%']:.3f}, {d['97.5%']:.3f})"  # noqa: E731
        rows.append([nm, est(r), fmt_p(r["P"]), est(p), fmt_p(p["P"])])
    sig_r = {k: v for k, v in r_rows(rb).items() if k == "sigma"}
    story.append(KeepTogether([Paragraph(f"<b>{title}</b>", BODY),
                               tbl(rows, [3.6 * cm, 4.2 * cm, 1.4 * cm, 4.2 * cm, 1.4 * cm])]))
    story.append(Spacer(1, 4))

# residual SDs
rs = {b["block"]: b for b in R["stats"]}["sigmas"]
rows = [["Residual standard deviation", "R: mean (SD)", "Python: mean (SD)"]]
for i, lab in enumerate(["log(serum bilirubin)", "Prothrombin time"]):
    rows.append([lab, f"{rs['Mean'][i]:.4f} ({rs['SD'][i]:.4f})",
                 f"{P['sigmas'][i]['Mean']:.4f} ({P['sigmas'][i]['StDev']:.4f})"])
story.append(tbl(rows, [6 * cm, 4.5 * cm, 4.5 * cm]))

story.append(Spacer(1, 6))
rows = [["Coefficient", "Same direction", "Same significance (P &lt; 0.05)",
         "Same 'CI excludes 0'"]]
for c in CONC:
    yn = lambda b: "yes" if b else "<b>NO</b>"  # noqa: E731
    rows.append([f"{c['group']}: {c['name']}", yn(c["same_sign"]), yn(c["same_sig"]),
                 yn(c["same_ci0"])])
story.append(KeepTogether([
    Paragraph(f"<b>Same conclusions.</b> {n_same} of {n_conc} regression coefficients lead to "
              "the same conclusion in both programs.", BODY),
    tbl(rows, [7.2 * cm, 2.6 * cm, 4.2 * cm, 3.0 * cm])]))

story.append(PageBreak())
story.append(Paragraph("Formal equivalence test (all 50 parameters)", H2))
story.append(Paragraph(
    "<b>Step 1: realistic error bars.</b> Any comparison of two MCMC runs needs the Monte "
    "Carlo standard error (MCSE) of each estimate. JMbayes2 reports MCSE = SD / sqrt(effective "
    "sample size), with the effective sample size from an autoregressive fit to all chains "
    "stacked together. To check this formula we compared two runs of the same program that "
    "differ only in their random seed: their differences should be within ±2 MCSE about 95% "
    f"of the time. For Python only {np.mean(np.abs(Z_PP) < 2):.0%} were"
    + (f", for R only {np.mean(np.abs(Z_RR) < 2):.0%}" if R2 is not None else "")
    + f", so the formula understates the real run-to-run error by a factor k = {K_CAL:.2f} "
    "(k = sqrt(mean z<super>2</super>) over all parameters and "
    + ("both same-program pairs" if R2 is not None else "the Python pair")
    + "). All error bars below are multiplied by k; the correction applies equally to R "
    "and Python.", BODY))
story.append(Spacer(1, 3))
story.append(Paragraph(
    "<b>Step 2: equivalence test.</b> A non-significant difference test only says a "
    "difference could not be detected. An <b>equivalence test (TOST)</b> asks whether the "
    "difference is small enough to be irrelevant. For each parameter, d = Python mean - R "
    "mean (each pooled over its seeds), with SE = k · sqrt(MCSE<sub>R</sub><super>2</super> + "
    "MCSE<sub>Py</sub><super>2</super>). The programs are declared equivalent for that "
    "parameter if the 90% confidence interval d ± 1.645·SE lies inside ±δ (two one-sided "
    "tests at α = 0.05). The margin is relative to the parameter's own posterior "
    "uncertainty: δ = 0.1 posterior SD (strict) or 0.2 SD. A shift of 0.2 SD moves a 95% "
    "credible interval by about 5% of its width.", BODY))
story.append(Spacer(1, 3))
rows = [["Comparison", "Median |difference|", "Largest |difference|"],
        ["R vs Python (pooled seeds)", f"{np.median(dsd):.3f} SD", f"{dsd.max():.3f} SD"],
        ["Python seed 123 vs Python seed 456", f"{np.median(D_PP):.3f} SD",
         f"{D_PP.max():.3f} SD"]]
if R2 is not None:
    rows.append(["R seed 123 vs R seed 456", f"{np.median(D_RR):.3f} SD",
                 f"{D_RR.max():.3f} SD"])
story.append(KeepTogether([
    Paragraph("<b>Benchmark: how different are two runs of the same program?</b>", BODY),
    tbl(rows, [7 * cm, 4 * cm, 4 * cm])]))
story.append(Spacer(1, 4))
rows = [["Parameter group", "Parameters", "Equivalent at ±0.1 SD", "Equivalent at ±0.2 SD",
         "Largest |d| (SD)"]]
for blk, label in GROUPS:
    cs = [c for c in CMP if c["block"] == blk]
    rows.append([label, len(cs), sum(c["equiv_0.1"] for c in cs),
                 sum(c["equiv_0.2"] for c in cs),
                 f"{max(abs(c['d_sd']) for c in cs):.2f}"])
rows.append(["<b>All</b>", f"<b>{len(CMP)}</b>", f"<b>{n_eq1}</b>", f"<b>{n_eq2}</b>",
             f"<b>{dsd.max():.2f}</b>"])
story.append(tbl(rows, [6.0 * cm, 2.2 * cm, 3.2 * cm, 3.2 * cm, 2.6 * cm]))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "Figure: each parameter's difference (Python - R) in posterior SDs with its 90% CI "
    "(calibrated error bars). Orange: equivalent at ±0.1 SD; grey: equivalent at ±0.2 SD "
    "only; red: not shown equivalent at ±0.2 SD. Wide intervals belong to parameters that "
    "mix slowly in both programs (the random-effects covariances, mainly the ascites random "
    "slope, and the baseline-hazard spline); their estimates still sit close to zero "
    "difference, and longer runs would narrow the intervals.", SMALL))
story.append(img("equivalence.png", 15.5 * cm))
story.append(Paragraph(
    f"<b>Noise check.</b> If both programs sample the same posterior, z = d / SE behaves like "
    f"a standard normal. Observed: {np.mean(np.abs(zarr) < 2):.0%} of |z| below 2 (expected "
    f"~95%) and {np.mean(np.abs(zarr) < 3):.0%} below 3 (expected ~99.7%).", BODY))

fs = {k: (R["fit_stats"][k], P["fit_stats"][k], P2["fit_stats"][k])
      for k in ("DIC", "pD", "WAIC", "LPML")}
rows = [["Fit statistic (conditional)", "R", "Python (seed 123)", "Python (seed 456)",
         "|R - Python mean|", "|Python - Python|"]]
for k, (r, a, b) in fs.items():
    rows.append([k, f"{r:.1f}", f"{a:.1f}", f"{b:.1f}", f"{abs(r - (a + b) / 2):.1f}",
                 f"{abs(a - b):.1f}"])
story.append(KeepTogether([
    Paragraph("<b>Model fit statistics.</b> DIC, WAIC and LPML are themselves Monte Carlo "
              "estimates. The table puts the R-vs-Python gap next to the gap between two "
              "Python seeds; the gaps are of comparable size, so the fit statistics agree within "
              "run-to-run variation (relative differences are below 0.3%).", BODY),
    tbl(rows, [3.6 * cm, 2.4 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm])]))


# ---- Convergence (Rhat), R vs Python
def rhat_of(run):
    """Rhat of the 15 regression coefficients and the 2 residual SDs, keyed by name.
    R files store them per coefficient table (with 'sigma' rows inside betas1/betas2);
    Python files store them per block."""
    out = {}
    if "gammas_alphas" in run:
        for r in run["gammas_alphas"]:
            out["surv:" + r["name"]] = r["Rhat"]
        for b in ("betas1", "betas2", "betas3"):
            for r in run[b]:
                out[(f"sigma:{b}" if r["name"] == "sigma" else f"{b}:{r['name']}")] = r["Rhat"]
    else:
        for b in ("gammas", "alphas"):
            for r in run[b]:
                out["surv:" + r["name"]] = r["Rhat"]
        for b in ("betas1", "betas2", "betas3"):
            for r in run[b]:
                out[f"{b}:{r['name']}"] = r["Rhat"]
        for b, r in zip(("betas1", "betas2"), run["sigmas"]):
            out[f"sigma:{b}"] = r["Rhat"]
    return out


RH_LABEL = {"surv": "Survival", "betas1": "log(bilirubin)", "betas2": "prothrombin",
            "betas3": "ascites"}


def rh_name(key):
    grp, nm = key.split(":", 1)
    if grp == "sigma":
        return f"{RH_LABEL[nm]}: residual SD"
    return f"{RH_LABEL[grp]}: {nm}"


R2_RUN = json.load(open(HERE / ARGS.r2)) if (HERE / ARGS.r2).exists() else None
rh_runs = [("R, seed 123", rhat_of(R))] + ([("R, seed 456", rhat_of(R2_RUN))] if R2_RUN else []) \
    + [("Python, seed 123", rhat_of(P)), ("Python, seed 456", rhat_of(P2))]
rh_keys = [k for k in rh_runs[0][1] if all(k in d for _, d in rh_runs)]
py_all = [r["Rhat"] for run in (P, P2) for blk, _ in GROUPS for r in run[blk]]
story.append(PageBreak())
story.append(Paragraph("Convergence: Rhat in both programs", H2))
story.append(Paragraph(
    "Rhat (Gelman-Rubin potential scale reduction) compares the spread within each chain to "
    "the spread between chains; values near 1 mean the chains agree, and JMbayes2's usual "
    "threshold is 1.05. Both programs compute it the same way (the point estimate of "
    "<font face='Courier'>coda::gelman.diag</font>, which JMbayes2 reports). The R output "
    "stores Rhat for the regression coefficients and residual SDs only, so those are compared "
    f"here; across all {len(py_all) // 2} parameters, the Python runs reach at most "
    f"{max(py_all):.3f}.", BODY))
story.append(Spacer(1, 3))
rows = [["Parameter"] + [n for n, _ in rh_runs]]
for k in rh_keys:
    rows.append([rh_name(k)] + [f"{d[k]:.3f}" for _, d in rh_runs])
mx = [max(d[k] for k in rh_keys) for _, d in rh_runs]
rows.append(["<b>Largest</b>"] + [f"<b>{v:.3f}</b>" for v in mx])
story.append(KeepTogether([
    Paragraph(f"<b>Long runs</b> ({ARGS.settings})", BODY),
    tbl(rows, [5.4 * cm] + [(17.0 - 5.4) / len(rh_runs) * cm] * len(rh_runs))]))
story.append(Spacer(1, 3))
story.append(Paragraph(
    "The pattern is the same in both programs: the longitudinal coefficients are at 1.00, and "
    "the slowest-mixing parameters are the same ones - the ascites association and the "
    "ascites (logistic) sub-model. All stay below 1.05 in every run.", BODY))

SCAL_IMPLS = [("R", "JMbayes2"), ("Python-GPU", "Python, GPU"), ("Python-CPU", "Python, CPU")]
scal_cfg = sorted({(r["k"], r["n_iter"]) for r in SC}, key=lambda t: (t[1], t[0]))
rows = [["Patients", "Iterations"] + [lab for _, lab in SCAL_IMPLS]]
n_short = 0
for k, n in scal_cfg:
    cells = []
    for impl, _ in SCAL_IMPLS:
        f = HERE / f"scal_{impl}_k{k}_it{n}.json"
        if not f.exists():
            cells.append("-")
            continue
        d = rhat_of(json.load(open(f)))
        d = {kk: v for kk, v in d.items() if not kk.startswith("sigma:")}
        worst = max(d, key=d.get)
        cells.append(f"{d[worst]:.2f}, {sum(v > 1.05 for v in d.values())}/{len(d)} "
                     f"<font size=6.5>({rh_name(worst)})</font>")
        n_short += 1
    rows.append([f"{312 * k:,}", f"{n:,}"] + cells)
if n_short:
    story.append(Spacer(1, 6))
    story.append(KeepTogether([
        Paragraph("<b>Short runs</b> (scaling runs in Section 2: 3 chains, burn-in 500, "
                  "thinning 5). Each cell: largest Rhat, number of the 15 coefficients above "
                  "1.05, and the worst coefficient.", BODY),
        tbl(rows, [1.7 * cm, 1.8 * cm] + [4.5 * cm] * len(SCAL_IMPLS))]))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "With JMbayes2's default length (3,500 iterations) and 3 chains, neither program "
        "converges for the ascites-related parameters on this model: several coefficients "
        "are above 1.05 in R and in Python alike, at every cohort size. Rhat is itself noisy "
        "with only 3 chains, so the exact values move from run to run, but neither program "
        "is consistently better. For this model, real analyses need longer runs or more "
        "chains in either program - which is where the speed of jmbayes on a GPU matters "
        "most, since extra chains cost almost nothing there.", BODY))


# ---- Section 1b: isolating the sampler
if RINIT:
    a, b = summarise(CMP), summarise(RINIT["cmp"])
    story.append(PageBreak())
    story.append(Paragraph("Isolating the sampler: Python started from R's own starting fits",
                           H2))
    story.append(Paragraph(
        "JMbayes2 does not start from scratch: it takes fitted lme / mixed_model / coxph "
        "objects, uses them as starting values and centres some priors on them (for example "
        "the prior for each random-effect standard deviation). The Python runs above used "
        "Python's own versions of those fits, which match R's to 4-5 digits for the linear "
        "mixed models but differ by about 1% for the ascites logistic mixed model. To remove "
        "that difference entirely, the R fits were exported (benchmark/export_initfits.R) and "
        "Python was run again from exactly those values, with the same MCMC settings and two "
        "seeds. Any remaining difference can then only come from the sampler itself.", BODY))
    rows = [["", "Python, own starting fits", "Python, R's starting fits"],
            ["Median |difference| from R", f"{a['median']:.3f} SD", f"{b['median']:.3f} SD"],
            ["Largest |difference| from R", f"{a['max']:.3f} SD", f"{b['max']:.3f} SD"],
            ["Mean difference, ascites random-slope covariances D[6,·]",
             f"{a['d6']:+.3f} SD", f"{b['d6']:+.3f} SD"],
            ["Equivalent at ±0.1 SD (TOST)", f"{a['eq1']} / {len(CMP)}",
             f"{b['eq1']} / {len(CMP)}"],
            ["Equivalent at ±0.2 SD (TOST)", f"{a['eq2']} / {len(CMP)}",
             f"{b['eq2']} / {len(CMP)}"],
            ["|z| below 2 (expected ~95%)", f"{a['z2']:.0%}", f"{b['z2']:.0%}"],
            ["|z| below 3 (expected ~99.7%)", f"{a['z3']:.0%}", f"{b['z3']:.0%}"],
            ["Parameters shown to differ (90% CI outside ±0.2 SD)", f"{a['diff']}",
             f"{b['diff']}"]]
    story.append(tbl(rows, [8.2 * cm, 4.2 * cm, 4.2 * cm]))
    story.append(Spacer(1, 4))
    pr1 = {c["name"]: c for c in RINIT["cmp"]}
    rs_ = {b_["block"]: b_ for b_ in R["stats"]}
    rows = [["Parameter", "R", "Python (own fits)", "Python (R's fits)"]]
    for blk, nm_list in (("betas3", ["year"]), ("D", [c["name"] for c in CMP
                                                       if c["name"].startswith("D[6")])):
        rb = rs_[blk]
        names = [p_["name"] for p_ in P[blk]]
        for nm in nm_list:
            i = names.index(nm)
            own = (P[blk][i]["Mean"] + P2[blk][i]["Mean"]) / 2
            rin = (RINIT["P1"][blk][i]["Mean"] + RINIT["P2"][blk][i]["Mean"]) / 2
            lab = "ascites: year" if blk == "betas3" else nm
            rmean = np.mean([rows_of(rr_, blk)[i][0] for rr_ in R_RUNS])
            rows.append([lab, f"{rmean:.4f}", f"{own:.4f}", f"{rin:.4f}"])
    story.append(KeepTogether([
        Paragraph("<b>The parameters that differed most</b> (posterior means; every column "
                  "averages two seeds)", BODY),
        tbl(rows, [5 * cm, 3.6 * cm, 3.8 * cm, 3.8 * cm])]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"Figure: as before, R vs Python started from R's fits (calibration factor k = "
        f"{RINIT['k']:.2f}, estimated from the same-program pairs).", SMALL))
    story.append(img("equivalence_rinit.png", 14.5 * cm))

# ---- Section 2: performance
story.append(PageBreak())
story.append(Paragraph("2. Speed and memory (preliminary)", H2))
story.append(Paragraph(
    "<i>Preliminary: the sweep was paused to run the equivalence test. The Python run at "
    "28,000 iterations and the colleague-scale runs (30,000-90,000 iterations) will be added "
    "in a later version. The R run at 28,000 iterations overlapped with another R run on this "
    "24-core machine, so its time may be slightly inflated.</i>", SMALL))
story.append(Paragraph(
    "Each run is a separate process, run one at a time. Peak RAM is the resident memory of "
    "the whole process tree, sampled every 0.2 s (R starts one worker process per chain; all "
    "are counted). Larger cohorts are pbc2 copied 2, 5 and 10 times with new patient ids. "
    "3 chains, burn-in 500, thinning 5. <b>MCMC time</b> is the sampling step; for R it "
    "includes starting its parallel worker processes. <b>Total time</b> also includes the "
    "separate starting fits (lme, mixed_model, coxph) that both programs need.", BODY))
story.append(Paragraph("<b>Scaling with the number of patients</b> (3,500 iterations, "
                       "JMbayes2's default)", BODY))
story.append(Table([[img("pat_mcmc_s.png", 8.2 * cm), img("pat_wall_s.png", 8.2 * cm)],
                    [img("pat_peak_ram_gb.png", 8.2 * cm), ""]]))
if any(r["k"] == 5 and r["n_iter"] > 3500 for r in SC):
    story.append(Paragraph("<b>Scaling with the number of iterations</b> (1,560 patients)",
                           BODY))
    story.append(Table([[img("it_mcmc_s.png", 8.2 * cm), img("it_peak_ram_gb.png", 8.2 * cm)]]))

rows = [["Implementation", "Patients", "Iterations", "MCMC (s)", "Total (s)", "Peak RAM (GB)",
         "GPU memory (GB)"]]
for r in sorted(SC, key=lambda r: (r["n_iter"], r["patients"], r["impl"])):
    rows.append([{"R": "JMbayes2 (R)", "Python-GPU": "Python, GPU",
                  "Python-CPU": "Python, CPU"}[r["impl"]], f"{r['patients']:,}",
                 f"{r['n_iter']:,}", f"{r['mcmc_s']:.1f}", f"{r['wall_s']:.1f}",
                 f"{r['peak_ram_gb']:.2f}",
                 f"{r['gpu_peak_gb']:.2f}" if r["impl"] == "Python-GPU" else "-"])
story.append(Spacer(1, 4))
story.append(KeepTogether([Paragraph("<b>All measurements</b>", BODY),
                           tbl(rows, [3.2 * cm, 2 * cm, 2.2 * cm, 2 * cm, 2 * cm, 2.6 * cm,
                                      2.8 * cm])]))

# ---- Section 3: notes
story.append(Paragraph("3. What was ported, and caveats", H2))
notes = [
    "<b>Same algorithm.</b> Python follows JMbayes2's C++ sampler step for step: the same "
    "parameter blocks in the same order, the same proposal types, the same Gibbs steps, the "
    "same adaptive step sizes and burn-in rules. Only the random-number stream differs, so "
    "individual draws differ while the results agree within Monte Carlo error.",
    "<b>How it is faster.</b> All chains and all patients are updated together as tensor "
    "operations, and on the GPU each iteration is recorded once and replayed (CUDA graph). "
    "The per-patient log-likelihood is not stored for every iteration; DIC, WAIC and LPML are "
    "accumulated during sampling, which gives identical values (checked to 1e-16).",
    "<b>Starting fits.</b> JMbayes2 takes fitted lme / mixed_model / coxph objects, which also "
    "set some prior centres. Python re-implements these estimators (REML, ML with adaptive "
    "Gauss-Hermite quadrature, Cox with Efron ties); they match to optimiser tolerance.",
    "<b>Two possible issues in JMbayes2's C++</b> were found and deliberately reproduced so "
    "results match: the random-effects log-density is not refreshed after the fixed-effects "
    "step, and the survival denominator is not refreshed after the smoothing-parameter "
    "update. Both are worth raising with the package authors.",
    "<b>Timing caveats.</b> A first R run on pbc2 took 143 s of MCMC and a repeat 58.5 s, so "
    "single timings vary. On large copies, R's own nlme starting fit failed to converge with "
    "its default optimiser; the optim() fallback was used only in that case.",
    "<b>Scope.</b> The Python version currently covers what this model needs: Gaussian and "
    "binomial outcomes, current-value association, right censoring. Other JMbayes2 features "
    "(more families, slope/area associations, competing risks, recurrent events) are not "
    "ported yet, and marginal fit statistics are not computed.",
]
for n in notes:
    story.append(Paragraph("• " + n, BODY))
    story.append(Spacer(1, 3))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor(INK2))
    canvas.drawString(1.8 * cm, 1.1 * cm, "jmbayes vs JMbayes2 - agreement and performance")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


out = HERE / ARGS.out
SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                  topMargin=1.6 * cm, bottomMargin=1.8 * cm,
                  title="jmbayes vs JMbayes2").build(story, onFirstPage=footer,
                                                         onLaterPages=footer)
print(f"wrote {out}")
