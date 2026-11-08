"""Equivalence of jmbayes and JMbayes2 for the extension models (same test as the main
report): competing risks, and the 0.6 additions.

Usage: python compare_cr.py [model]   (default cr)
Inputs: R_<model>_seed123.json, R_<model>_seed456.json (run_cr_R.R / run_ext_R.R /
run_ext2_R.R) and PY_<model>_seed123.json, PY_<model>_seed456.json (run_cr.py /
run_ext.py).
  cr:   PBC, death and transplantation as competing events (12 x 30,000 iterations)
  surv: delayed entry and a baseline hazard per treatment arm (12 x 30,000)
  fam:  negative binomial, Poisson and Student's t markers (12 x 30,000)
  0.7 models (8 x 15,000): tvc, icens, crx, rec, fam2, fam3, bystr (see run_ext.py)
Prints the summary and writes <model>_equivalence.json.
"""
import json
import math
import pathlib
import sys

import numpy as np
from scipy.stats import norm

HERE = pathlib.Path(__file__).resolve().parent
BLOCKS = ["gammas", "alphas", "betas1", "betas2", "sigmas", "D", "bs_gammas",
          "tau_bs_gammas"]
MARGINS = (0.1, 0.2)
Z90 = 1.6448536269514722


def rows_of(run, blk):
    """(mean, posterior SD, MCSE) per parameter, from an R or a Python result file."""
    if "stats" in run:
        st = {b["block"]: b for b in run["stats"]}[blk]
        return list(zip(*(np.atleast_1d(st[k]).astype(float) for k in ("Mean", "SD", "SE"))))
    return [(r["Mean"], r["StDev"], r["SE"]) for r in run[blk]]


def pair_stats(a, b, blocks):
    z, d = [], []
    for blk in blocks:
        for (ma, sa, ea), (mb, _, eb) in zip(rows_of(a, blk), rows_of(b, blk)):
            z.append((ma - mb) / math.hypot(ea, eb))
            d.append(abs(ma - mb) / sa)
    return np.array(z), np.array(d)


def pooled(runs, blk):
    rows = [rows_of(r, blk) for r in runs]
    return [(np.mean([v[0] for v in vals]), vals[0][1],
             math.sqrt(sum(v[2] ** 2 for v in vals)) / len(vals)) for vals in zip(*rows)]


def main(model="cr"):
    R = [json.load(open(HERE / f"R_{model}_seed{s}.json")) for s in (123, 456)]
    P = [json.load(open(HERE / f"PY_{model}_seed{s}.json")) for s in (123, 456)]
    BLOCKS = [b["block"] for b in R[0]["stats"]]
    for blk in BLOCKS:                     # same parameters in both programs
        assert len(rows_of(R[0], blk)) == len(rows_of(P[0], blk)), blk
    z_pp, d_pp = pair_stats(*P, BLOCKS)
    z_rr, d_rr = pair_stats(*R, BLOCKS)
    k = math.sqrt(np.mean(np.square(np.concatenate([z_pp, z_rr]))))
    names = [r["name"] for blk in BLOCKS for r in P[0][blk]]
    cmp = []
    for blk in BLOCKS:
        for (rm, rsd, rse), (pm, _, pse) in zip(pooled(R, blk), pooled(P, blk)):
            d = pm - rm
            se = k * math.hypot(rse, pse)
            rec = dict(block=blk, d_sd=d / rsd, z=d / se,
                       lo_sd=(d - Z90 * se) / rsd, hi_sd=(d + Z90 * se) / rsd)
            for m in MARGINS:
                delta = m * rsd
                p = max(1 - norm.cdf((d + delta) / se), norm.cdf((d - delta) / se))
                rec[f"equiv_{m}"] = bool(p < 0.05)
            cmp.append(rec)
    for c, nm in zip(cmp, names):
        c["name"] = nm
    dsd = np.abs([c["d_sd"] for c in cmp])
    # hazard ratios side by side (posterior means pooled over the two seeds)
    hr = []
    for blk in ("gammas", "alphas"):
        for nm, (rm, _, _), (pm, _, _) in zip([r["name"] for r in P[0][blk]],
                                              pooled(R, blk), pooled(P, blk)):
            hr.append(dict(name=nm, R=math.exp(rm), Python=math.exp(pm)))
    out = dict(
        n_parameters=len(cmp), k=float(k),
        equivalent_0_1=int(sum(c["equiv_0.1"] for c in cmp)),
        equivalent_0_2=int(sum(c["equiv_0.2"] for c in cmp)),
        shown_to_differ=int(sum(bool(c["lo_sd"] > 0.2 or c["hi_sd"] < -0.2) for c in cmp)),
        median_abs_diff_sd=float(np.median(dsd)), max_abs_diff_sd=float(dsd.max()),
        R_vs_R=dict(median=float(np.median(d_rr)), max=float(d_rr.max())),
        Py_vs_Py=dict(median=float(np.median(d_pp)), max=float(d_pp.max())),
        hazard_ratios=hr, parameters=cmp,
        fit_stats={kind: {prog: {k: float(np.mean([r[kind][k] for r in runs]))
                                 for k in ("DIC", "pD", "WAIC", "LPML")}
                          for prog, runs in (("R", R), ("Python", P))}
                   for kind in ("fit_stats", "fit_stats_marginal")
                   if all(kind in r for r in R + P)},
        mcmc_time=dict(R=[r["mcmc_time"] for r in R], Python=[p["mcmc_time"] for p in P]))
    (HERE / f"{model}_equivalence.json").write_text(json.dumps(out, indent=1))
    print(f"{out['equivalent_0_2']}/{len(cmp)} equivalent at +-0.2 SD, "
          f"{out['equivalent_0_1']} at +-0.1 SD, {out['shown_to_differ']} shown to differ "
          f"(k = {k:.2f})")
    print(f"median |d| R vs Python {out['median_abs_diff_sd']:.3f} SD (max "
          f"{out['max_abs_diff_sd']:.3f}); R vs R {out['R_vs_R']['median']:.3f} "
          f"(max {out['R_vs_R']['max']:.3f}); Py vs Py {out['Py_vs_Py']['median']:.3f} "
          f"(max {out['Py_vs_Py']['max']:.3f})")
    for h in hr:
        print(f"  {h['name']:40s} HR R {h['R']:.3f}  Python {h['Python']:.3f}")
    worst = sorted(cmp, key=lambda c: -abs(c["d_sd"]))[:5]
    for c in worst:
        print(f"  largest: {c['name']:30s} d = {c['d_sd']:+.3f} SD  "
              f"equiv@0.2 {c['equiv_0.2']}")
    for kind, v in out["fit_stats"].items():
        print(f"  {kind:20s} " + "  ".join(
            f"{k} R {v['R'][k]:.1f} / Py {v['Python'][k]:.1f}" for k in ("DIC", "WAIC", "LPML")))
    print("MCMC time (s):", out["mcmc_time"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cr")
