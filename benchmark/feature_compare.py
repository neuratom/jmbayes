"""Compare jmbayes2py with JMbayes2 on the feature models (see run_feature_R.R).

Usage: python feature_compare.py <model> [--runs 2]
Runs Python with the same MCMC settings as the R reference (two seeds), then reports,
per parameter, the difference in posterior SDs, z with Monte Carlo errors calibrated on
the Python seed pair, and an equivalence test (TOST, margin 0.2 posterior SD).
"""
import argparse
import json
import math
import pathlib

import numpy as np
from scipy.stats import norm

import jmbayes2py as jmb

HERE = pathlib.Path(__file__).resolve().parent


def models(name, pid):
    if name == "nhc":
        return dict(survival=jmb.Survival("Surv(years, status2) ~ drug + age", data=pid),
                    longitudinal=[
                        jmb.Longitudinal("log(serBilir) ~ year + drug + age", random="~ 1 | id"),
                        jmb.Longitudinal("albumin ~ year + sex", random="~ year | id")])
    if name == "splines":
        return dict(survival=jmb.Survival("Surv(years, status2) ~ sex", data=pid),
                    longitudinal=[jmb.Longitudinal("log(serBilir) ~ ns(year, 2) + sex",
                                                   random="~ ns(year, 2) | id")])
    if name == "forms":
        return dict(survival=jmb.Survival("Surv(years, status2) ~ sex", data=pid),
                    longitudinal=[jmb.Longitudinal("log(serBilir) ~ year * sex",
                                                   random="~ year | id"),
                                  jmb.Longitudinal("prothrombin ~ year", random="~ year | id")],
                    functional_forms="~ value(log(serBilir)) + slope(log(serBilir)) + "
                                     "area(prothrombin)")
    raise ValueError(name)


def py_rows(fit, blk):
    if blk == "sigmas":
        draws, names = fit.blocks["sigmas"]
        return fit._table("sigmas")
    return fit._table(blk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    a = ap.parse_args()
    R = [json.loads(p.read_text()) for p in sorted(HERE.glob(f"R_feature_{a.model}_seed*.json"))]
    s0 = R[0]["settings"]
    pbc2, pid = jmb.datasets.pbc2()
    fits = []
    for seed in (123, 456):
        f = jmb.jm(**models(a.model, pid), data=pbc2, time_var="year", n_iter=s0["n_iter"],
                   n_burnin=s0["n_burnin"], n_chains=s0["n_chains"], seed=seed, verbose=False)
        fits.append(f)
        f.save(str(HERE / f"PY_feature_{a.model}_seed{seed}.jmb"))
    blocks = [b["block"] for b in R[0]["stats"] if b["block"] != "D"] + ["D"]
    zpp, rows = [], []
    for blk in blocks:
        rstats = [{b["block"]: b for b in r["stats"]}[blk] for r in R]
        rm = np.mean([np.atleast_1d(s["Mean"]).astype(float) for s in rstats], 0)
        rsd = np.atleast_1d(rstats[0]["SD"]).astype(float)
        rse = np.sqrt(np.sum([np.atleast_1d(s["SE"]).astype(float) ** 2 for s in rstats], 0)) / len(R)
        t1, t2 = (f._table(blk) for f in fits)
        for i, name in enumerate(t1.index):
            m1, m2 = t1["Mean"].iloc[i], t2["Mean"].iloc[i]
            e1, e2 = t1["SE"].iloc[i], t2["SE"].iloc[i]
            zpp.append((m1 - m2) / math.hypot(e1, e2))
            rows.append((blk, name, rm[i], (m1 + m2) / 2, rsd[i], math.hypot(e1, e2) / 2, rse[i],
                         abs(m1 - m2) / rsd[i]))
    k = max(1.0, math.sqrt(np.mean(np.square(zpp))))   # never make error bars smaller
    print(f"\nmodel '{a.model}': {len(R)} R run(s), 2 Python runs, calibration k = {k:.2f}\n")
    print(f"{'block':8s} {'parameter':26s} {'R':>10s} {'Python':>10s} {'d/SD':>7s} {'z':>6s}"
          f" {'TOST0.2':>8s}")
    n_eq, dsd, zs = 0, [], []
    for blk, name, rm_, pm, sd, pse, rse_, _ in rows:
        d = pm - rm_
        se = k * math.hypot(pse, rse_)
        p = max(1 - norm.cdf((d + 0.2 * sd) / se), norm.cdf((d - 0.2 * sd) / se))
        n_eq += p < 0.05
        dsd.append(abs(d) / sd)
        zs.append(d / se)
        print(f"{blk:8s} {name:26s} {rm_:10.4f} {pm:10.4f} {d / sd:7.3f} {d / se:6.2f}"
              f" {'yes' if p < 0.05 else 'no':>8s}")
    dpp = [r[-1] for r in rows]
    print(f"\nmedian |R - Python| = {np.median(dsd):.3f} SD (max {max(dsd):.3f});"
          f" median |Python - Python| = {np.median(dpp):.3f} SD (max {max(dpp):.3f})")
    print(f"|z| < 2: {np.mean(np.abs(zs) < 2):.0%}   equivalent at ±0.2 SD: {n_eq}/{len(rows)}")
    print("fit stats R:", {k_: round(v, 1) for k_, v in R[0]["fit_stats"].items()},
          " Python:", {k_: round(v, 1) for k_, v in fits[0].fit_statistics.items()})


if __name__ == "__main__":
    main()
