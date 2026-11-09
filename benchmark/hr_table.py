"""LaTeX tables for the paper from the <model>_equivalence.json files written by
compare_cr.py: hazard ratios (posterior means of exp(coefficient), two seeds averaged) in
JMbayes2 and jmbayes, and the equivalence results per model.

Usage: python hr_table.py   (writes ../paper/hr_table.tex and ../paper/ext_table.tex)
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
PAPER = HERE.parent / "paper"
MODELS = [
    ("main", "Main model (PBC, 3 markers)"), ("cr", "Competing risks (PBC)"),
    ("surv", "Strata + late entry"), ("fam", "NB, Poisson, Student's $t$"),
    ("tvc", "Time-varying covariate"), ("icens", "Interval censoring"),
    ("crx", "Competing risks + entry + strata"), ("bystr", "Effects per stratum"),
    ("rec", "Recurrent events"), ("fam2", "Gamma, beta, cens.\\ normal"),
    ("fam3", "Binomial (trials), beta-binomial"),
    ("f8a", "Velocity, area"), ("f8b", "expit, Dexp, acceleration"),
    ("f8c", "Value by sex, Delta"), ("tva", "Time-varying association"),
    ("bhw", "Weibull baseline"), ("bhp", "Piecewise-constant baseline"),
    ("bhn", "Natural-spline baseline"), ("ms", "Multi-state"),
    ("wts", "Case weights"), ("pen", "Shrinkage priors"),
    ("aids", "AIDS data"), ("prothro", "Prothrombin data"),
]
MAX_PER_MODEL = 4


def tex(s):
    s = s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")
    return s if len(s) <= 34 else s[:31] + "\\ldots"



def _models():
    for key, label in MODELS:
        f = HERE / f"{key}_equivalence.json"
        if f.exists():
            yield label, json.loads(f.read_text())


def hr_table():
    head = r"\toprule Model & Term & JMbayes2 & jmbayes \\ \midrule"
    out = [r"\begin{longtable}{llrr}",
           r"\caption{Hazard ratios (posterior means, two seeds averaged) in JMbayes2 and "
           rf"jmbayes; up to {MAX_PER_MODEL} per model (all are in the repository).}}"
           r"\label{tab:hrall}\\",
           head + r" \endfirsthead", head + r" \endhead", r"\bottomrule \endlastfoot"]
    for label, e in _models():
        for i, h in enumerate(e["hazard_ratios"][:MAX_PER_MODEL]):
            out.append(f"{label if i == 0 else ''} & \\texttt{{{tex(h['name'])}}} & "
                       f"{h['R']:.3f} & {h['Python']:.3f} \\\\")
        out.append(r"\addlinespace")
    out.append(r"\end{longtable}")
    return "\n".join(out) + "\n"


def single_run_gaps(key):
    """|difference| per parameter between single runs, in posterior SDs: JMbayes2 vs
    jmbayes (same seed number, averaged over the two seeds), and JMbayes2 vs JMbayes2."""
    import numpy as np
    import compare_cr as C
    R = [json.loads((HERE / f"R_{key}_seed{s}.json").read_text()) for s in (123, 456)]
    P = [json.loads((HERE / f"PY_{key}_seed{s}.json").read_text()) for s in (123, 456)]
    rp, rr = [], []
    for blk in [b["block"] for b in R[0]["stats"]]:
        for a, b, c, d in zip(*(C.rows_of(x, blk) for x in (R[0], R[1], P[0], P[1]))):
            if a[1] > 0:
                rp.append((abs(a[0] - c[0]) + abs(b[0] - d[0])) / 2 / a[1])
                rr.append(abs(a[0] - b[0]) / a[1])
    return np.array(rp), np.array(rr)


def ext_table():
    """Main-text summary: what holds for every model."""
    head = (r"\toprule & & Shown to & \multicolumn{2}{c}{Difference (posterior SDs)} \\"
            "\n" r"\cmidrule(lr){4-5}" "\n"
            r"Model & Parameters & differ & Median & Largest \\ \midrule")
    out = [r"\begin{longtable}{lrrrr}",
           r"\caption{jmbayes against JMbayes2 for every model in this paper, judged the "
           r"same way: two runs per program with different seeds (12 chains of 30,000 "
           r"iterations for the first four models, 8 of 15,000 for the rest). Shown to "
           r"differ: parameters whose difference is shown to exceed 0.2 posterior SD (two "
           r"one-sided tests at 5\%). Difference: between a JMbayes2 run and a jmbayes run, "
           r"per parameter, in units of its posterior SD (median and largest over the "
           r"parameters). Equivalence counts per model are in Table~\ref{tab:extdetail}; "
           r"Section~\ref{sec:multi} repeats the comparison with 30 runs per program.}"
           r"\label{tab:ext}\\",
           head + r" \endfirsthead", head + r" \endhead", r"\bottomrule \endlastfoot"]
    for key, label in MODELS:
        f = HERE / f"{key}_equivalence.json"
        if not f.exists():
            continue
        e = json.loads(f.read_text())
        rp, _ = single_run_gaps(key)
        out.append(f"{label} & {e['n_parameters']} & {e['shown_to_differ']} & "
                   f"{float(sorted(rp)[len(rp) // 2]):.3f} & {rp.max():.3f} \\\\")
    out.append(r"\end{longtable}")
    return "\n".join(out) + "\n"


def ext_detail():
    head = (r"\toprule & & \multicolumn{2}{c}{Equivalent} & Incon- & & "
            r"\multicolumn{2}{c}{Largest diff.} \\" "\n"
            r"\cmidrule(lr){3-4}\cmidrule(lr){7-8}" "\n"
            r"Model & Param. & $\pm 0.1$ & $\pm 0.2$ & clusive & Differ & "
            r"R--Py & R--R \\ \midrule")
    out = [r"\begin{longtable}{lrrrrrrr}",
           r"\caption{Equivalence tests per model (the runs of Table~\ref{tab:ext}). Each "
           r"parameter is either "
           r"shown equivalent (within $\pm 0.2$ posterior SD; the $\pm 0.1$ count is a "
           r"subset), shown to differ (by more than 0.2 SD), or inconclusive: two runs per "
           r"program are too few to separate a difference below 0.2 SD from Monte Carlo "
           r"error, which happens where the chains mix slowly. No parameter was shown to "
           r"differ. Largest difference: between the programs (R vs Python, seeds "
           r"averaged) and between the two JMbayes2 runs of the same model (R vs R).}"
           r"\label{tab:extdetail}\\",
           head + r" \endfirsthead", head + r" \endhead", r"\bottomrule \endlastfoot"]
    for label, e in _models():
        n, e2, dif = e["n_parameters"], e["equivalent_0_2"], e["shown_to_differ"]
        out.append(f"{label} & {n} & {e['equivalent_0_1']} & {e2} & {n - e2 - dif} & "
                   f"{dif} & {e['max_abs_diff_sd']:.3f} & {e['R_vs_R']['max']:.3f} \\\\")
    out.append(r"\end{longtable}")
    return "\n".join(out) + "\n"


def _small(t):
    return r"\begingroup\small\setlength{\tabcolsep}{4pt}" + "\n" + t + r"\endgroup" + "\n"


def main():
    (PAPER / "hr_table.tex").write_text(_small(hr_table()))
    (PAPER / "ext_table.tex").write_text(ext_table())
    (PAPER / "ext_detail.tex").write_text(_small(ext_detail()))
    print("ok")


if __name__ == "__main__":
    main()
