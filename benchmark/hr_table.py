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
    return s if len(s) <= 38 else s[:35] + "\\ldots"



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


def ext_table():
    head = r"\toprule Model & Parameters & Equivalent & Differ & Largest \\ \midrule"
    out = [r"\begin{longtable}{lrrrr}",
           r"\caption{Equivalence of jmbayes and JMbayes2 for every model in this paper, "
           r"judged the same way: two seeds per program (12 chains of 30,000 iterations for the "
           r"first four models, 8 of 15,000 for the rest). Equivalent: shown equivalent within $\pm 0.1$ / $\pm 0.2$ "
           r"posterior SD. Differ: shown to differ by more than 0.2 SD. Largest: largest "
           r"absolute difference, in posterior SDs.}\label{tab:ext}\\",
           head + r" \endfirsthead", head + r" \endhead", r"\bottomrule \endlastfoot"]
    for label, e in _models():
        out.append(f"{label} & {e['n_parameters']} & {e['equivalent_0_1']} / "
                   f"{e['equivalent_0_2']} & {e['shown_to_differ']} & "
                   f"{e['max_abs_diff_sd']:.3f} \\\\")
    out.append(r"\end{longtable}")
    return "\n".join(out) + "\n"


def main():
    (PAPER / "hr_table.tex").write_text(hr_table())
    (PAPER / "ext_table.tex").write_text(ext_table())
    print("ok")


if __name__ == "__main__":
    main()
