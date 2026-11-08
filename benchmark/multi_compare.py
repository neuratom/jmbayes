"""Run-distribution comparison of jmbayes and JMbayes2 (run_multi_all.sh).

For each data set, each program was run N times with different seeds (independent
starting values and random numbers). Each run gives, per parameter, a posterior mean and
a posterior SD. Per parameter we compare the N jmbayes runs with the N JMbayes2 runs:

  posterior mean  Welch's t-test (do they differ?) and two one-sided tests (are they
                  equivalent within +-0.1 posterior SD?)
  posterior SD    the same, with an equivalence margin of +-10% of the posterior SD
  whole run-to-run distribution of the posterior mean
                  two-sample Kolmogorov-Smirnov test

Runs are independent, so these tests are valid without assumptions about MCMC
autocorrelation; the run-to-run spread is the Monte Carlo error itself. With many
parameters, about 5% of difference tests are expected at p < 0.05 when the programs are
identical, so counts are also given after Benjamini-Hochberg correction (5% FDR per data
set and test). A non-significant difference test alone does not show sameness; the
equivalence tests do.

Usage: python multi_compare.py   (prints the table, writes multi/summary.json and
                                  ../paper/multi_table.tex)
"""
import json
import pathlib

import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
D = HERE / "multi"
DATASETS = [("main", "PBC (312 patients, 3 markers)"), ("aids", "AIDS (467 patients)"),
            ("prothro", "Liver cirrhosis (488 patients)")]
MARGIN_MEAN = 0.1        # posterior SDs
MARGIN_SD = 0.1          # relative
ALPHA = 0.05


def r_params(run):
    out = []
    for b in run["stats"]:
        m, s = np.atleast_1d(b["Mean"]).astype(float), np.atleast_1d(b["SD"]).astype(float)
        out += [(b["block"], i, mi, si) for i, (mi, si) in enumerate(zip(m, s))]
    return out


def p_params(run, blocks):
    return [(blk, i, r["Mean"], r["StDev"]) for blk in blocks for i, r in enumerate(run[blk])]


def p_names(run, blocks):
    return [f"{blk}: {r['name']}" for blk in blocks for r in run[blk]]


def welch_tost(a, b, margin):
    """p of the two one-sided Welch tests that |mean(a) - mean(b)| < margin."""
    va, vb, na, nb = a.var(ddof=1), b.var(ddof=1), len(a), len(b)
    se = np.sqrt(va / na + vb / nb)
    d = a.mean() - b.mean()
    if se == 0:
        return 0.0 if abs(d) < margin else 1.0
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return max(stats.t.sf((d + margin) / se, df), stats.t.cdf((d - margin) / se, df))


def n_needed(rm, pm, sd):
    """Runs per program for 80% power of the equivalence test of the posterior mean
    (margin +-0.1 posterior SD, alpha 5%, true difference 0), from the observed
    run-to-run SD: n = 2 s^2 (z_0.95 + z_0.90)^2 / margin^2."""
    s2 = (rm.var(ddof=1) + pm.var(ddof=1)) / 2
    z = stats.norm.ppf(0.95) + stats.norm.ppf(0.90)
    return int(np.ceil(2 * s2 * z * z / (MARGIN_MEAN * sd) ** 2))


def bh(p):
    """Benjamini-Hochberg: True where rejected at 5% FDR."""
    p = np.asarray(p)
    order = np.argsort(p)
    thr = ALPHA * np.arange(1, len(p) + 1) / len(p)
    passed = p[order] <= thr
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    rej = np.zeros(len(p), bool)
    rej[order[:k]] = True
    return rej


def compare(model):
    R = [json.loads(f.read_text()) for f in sorted(D.glob(f"R_{model}_*.json"))]
    P = [json.loads(f.read_text()) for f in sorted(D.glob(f"PY_{model}_*.json"))]
    blocks = [b["block"] for b in R[0]["stats"]]
    rp = [r_params(r) for r in R]
    pp = [p_params(p, blocks) for p in P]
    assert all(len(x) == len(rp[0]) for x in rp + pp), "parameter counts differ"
    names = p_names(P[0], blocks)
    rows = []
    for j in range(len(rp[0])):
        rm = np.array([x[j][2] for x in rp])
        rs = np.array([x[j][3] for x in rp])
        pm = np.array([x[j][2] for x in pp])
        ps = np.array([x[j][3] for x in pp])
        if np.all(rs == 0) and np.all(ps == 0):        # fixed in both (none expected)
            continue
        sd = np.mean(np.concatenate([rs, ps]))
        rows.append(dict(
            name=names[j], post_sd=float(sd),
            d_mean_sd=float((pm.mean() - rm.mean()) / sd),
            p_mean=float(stats.ttest_ind(pm, rm, equal_var=False).pvalue),
            eq_mean=bool(welch_tost(pm, rm, MARGIN_MEAN * sd) < ALPHA),
            sd_ratio=float(ps.mean() / rs.mean()),
            p_sd=float(stats.ttest_ind(ps, rs, equal_var=False).pvalue),
            eq_sd=bool(welch_tost(ps, rs, MARGIN_SD * rs.mean()) < ALPHA),
            p_ks=float(stats.ks_2samp(pm, rm).pvalue),
            mc_sd_R=float(rm.std(ddof=1) / sd), mc_sd_Py=float(pm.std(ddof=1) / sd),
            n_needed=n_needed(rm, pm, sd)))
    for key in ("p_mean", "p_sd", "p_ks"):
        rej = bh([r[key] for r in rows])
        for r, x in zip(rows, rej):
            r[key + "_bh"] = bool(x)
    n = len(rows)
    summ = dict(
        runs_R=len(R), runs_Py=len(P), n_parameters=n,
        mean_p05=sum(r["p_mean"] < ALPHA for r in rows),
        mean_bh=sum(r["p_mean_bh"] for r in rows),
        mean_equiv=sum(r["eq_mean"] for r in rows),
        sd_p05=sum(r["p_sd"] < ALPHA for r in rows),
        sd_bh=sum(r["p_sd_bh"] for r in rows),
        sd_equiv=sum(r["eq_sd"] for r in rows),
        ks_p05=sum(r["p_ks"] < ALPHA for r in rows),
        ks_bh=sum(r["p_ks_bh"] for r in rows),
        max_abs_d_mean_sd=max(abs(r["d_mean_sd"]) for r in rows),
        underpowered=sum(r["n_needed"] > min(len(R), len(P)) for r in rows),
        max_n_needed=max(r["n_needed"] for r in rows),
        median_mc_sd_R=float(np.median([r["mc_sd_R"] for r in rows])),
        median_mc_sd_Py=float(np.median([r["mc_sd_Py"] for r in rows])),
        mcmc_time=dict(R=[r["mcmc_time"] for r in R], Python=[p["mcmc_time"] for p in P]))
    return summ, rows


def main():
    out, tex = {}, []
    for model, label in DATASETS:
        if not list(D.glob(f"R_{model}_*.json")) or not list(D.glob(f"PY_{model}_*.json")):
            continue
        s, rows = compare(model)
        out[model] = dict(summary=s, parameters=rows)
        n = s["n_parameters"]
        print(f"{label}: {s['runs_R']} R / {s['runs_Py']} Py runs, {n} parameters")
        print(f"  posterior mean: p<0.05 {s['mean_p05']}, after BH {s['mean_bh']}, "
              f"equivalent (+-0.1 SD) {s['mean_equiv']}; largest |d| "
              f"{s['max_abs_d_mean_sd']:.3f} SD")
        print(f"  posterior SD:   p<0.05 {s['sd_p05']}, after BH {s['sd_bh']}, "
              f"equivalent (+-10%) {s['sd_equiv']}")
        print(f"  KS (run means): p<0.05 {s['ks_p05']}, after BH {s['ks_bh']}")
        print(f"  power: {s['underpowered']} parameters need more runs for 80% power "
              f"(largest need {s['max_n_needed']} per program)")
        print(f"  run-to-run SD of the mean (median, in posterior SDs): R "
              f"{s['median_mc_sd_R']:.3f}, Py {s['median_mc_sd_Py']:.3f}")
        bad = [r for r in rows if r["p_mean_bh"] or r["p_sd_bh"] or r["p_ks_bh"]
               or not r["eq_mean"]]
        for r in bad[:8]:
            print(f"    {r['name']:40s} d={r['d_mean_sd']:+.3f} SD  p={r['p_mean']:.3g} "
                  f"eq={r['eq_mean']}  sd ratio {r['sd_ratio']:.3f} p_sd={r['p_sd']:.3g}")
        tex.append(f"{label} & {s['runs_R']} / {s['runs_Py']} & {n} & "
                   f"{s['mean_p05']} ({s['mean_bh']}) & {s['mean_equiv']} & "
                   f"{s['sd_p05']} ({s['sd_bh']}) & {s['sd_equiv']} & "
                   f"{s['ks_p05']} ({s['ks_bh']}) \\\\")
    (D / "summary.json").write_text(json.dumps(out, indent=1))
    head = (r"\toprule & Runs & & \multicolumn{2}{c}{Posterior mean} & "
            r"\multicolumn{2}{c}{Posterior SD} & Run distribution \\"
            "\n" r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-8}"
            "\n" r"Data set & R / Py & Parameters & Differ & Equivalent & Differ & "
            r"Equivalent & Differ \\ \midrule")
    (HERE.parent / "paper" / "multi_table.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\small\n"
        "\\caption{Run-distribution comparison: each program run " + str(
            max([v['summary']['runs_R'] for v in out.values()] or [0])) +
        " times per data set with different seeds (8 chains of 15,000 iterations). Per "
        "parameter, the posterior means and SDs of the jmbayes runs are compared with those "
        "of the JMbayes2 runs. Differ: number of parameters with $p < 0.05$ in Welch's "
        "$t$-test (posterior mean, SD) or the two-sample Kolmogorov--Smirnov test (run "
        "distribution of the posterior mean); in brackets, after Benjamini--Hochberg "
        "correction at 5\\% FDR. About 5\\% of parameters are expected below 0.05 by "
        "chance when the programs are identical. Equivalent: shown equivalent by two "
        "one-sided tests, within $\\pm 0.1$ posterior SD (mean) or $\\pm 10\\%$ "
        "(SD).}\\label{tab:multi}\n\\vspace{4pt}\n"
        "\\begin{tabular}{lrrrrrrr}\n" + head + "\n" + "\n".join(tex) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


if __name__ == "__main__":
    main()
