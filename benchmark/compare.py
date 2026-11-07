"""Compare JMbayes2 (R) and jmbayes2py (Python) runs of the same model.

Agreement is judged against Monte Carlo noise: for each parameter
    z = (mean_Python - mean_R) / sqrt(SE_R^2 + SE_Python^2)
where SE is the Monte Carlo standard error of the posterior mean (JMbayes2's own
definition: SD / sqrt(effective sample size)). If both samplers target the same
posterior, z behaves like a standard normal: about 95% of |z| below 2.

Usage: python compare.py R_x1.json PY_x1.json
"""
import json
import sys

import numpy as np

BLOCKS = [("gammas", "gammas"), ("alphas", "alphas"), ("betas1", "betas1"),
          ("betas2", "betas2"), ("betas3", "betas3"), ("sigmas", "sigmas"),
          ("D", "D"), ("bs_gammas", "bs_gammas"), ("tau_bs_gammas", "tau_bs_gammas")]
LABEL = {"gammas": "Survival", "alphas": "Survival (association)",
         "betas1": "log(serBilir)", "betas2": "prothrombin", "betas3": "ascites",
         "sigmas": "Residual SDs", "D": "Random-effects covariance",
         "bs_gammas": "Baseline hazard", "tau_bs_gammas": "Baseline hazard smoothing"}


def main(r_file, p_file):
    R = json.load(open(r_file))
    P = json.load(open(p_file))
    rstats = {b["block"]: b for b in R["stats"]}
    rows = []
    for rb, pb in BLOCKS:
        r = rstats[rb]
        rm, rsd, rse = (np.atleast_1d(r[k]).astype(float) for k in ("Mean", "SD", "SE"))
        prow = P[pb]
        # R drops the name of single-value blocks; use the Python name there
        names = [p["name"] for p in prow] if len(rm) == 1 else np.atleast_1d(r["names"])
        for i, nm in enumerate(names):
            pm, psd, pse = prow[i]["Mean"], prow[i]["StDev"], prow[i]["SE"]
            z = (pm - rm[i]) / np.sqrt(rse[i] ** 2 + pse ** 2)
            rows.append((rb, str(nm), rm[i], pm, rsd[i], psd, z))

    print(f"\nR: {r_file}   Python: {p_file}")
    cur = None
    for b, nm, rm, pm, rsd, psd, z in rows:
        if b != cur:
            cur = b
            print(f"\n{LABEL[b]}")
            print(f"  {'parameter':24s}{'R mean':>11s}{'Py mean':>11s}{'R SD':>9s}{'Py SD':>9s}"
                  f"{'z':>7s}")
        flag = "  <-" if abs(z) > 3 else ""
        print(f"  {nm:24s}{rm:11.4f}{pm:11.4f}{rsd:9.4f}{psd:9.4f}{z:7.2f}{flag}")

    zs = np.array([r[-1] for r in rows])
    sdr = np.array([abs(r[3] - r[2]) / r[4] for r in rows])
    print(f"\n{len(zs)} parameters:  |z| < 2: {np.mean(np.abs(zs) < 2):.0%}   "
          f"|z| < 3: {np.mean(np.abs(zs) < 3):.0%}   "
          f"median |difference| = {np.median(sdr):.2f} posterior SDs, "
          f"max = {sdr.max():.2f}")

    print("\nConditional fit statistics")
    for k in ("DIC", "pD", "LPML", "WAIC"):
        print(f"  {k:5s} R {R['fit_stats'][k]:12.2f}   Python {P['fit_stats'][k]:12.2f}")

    rt, pt = R["timing"], P["timing"]
    print("\nTiming (seconds)")
    print(f"  separate fits / setup : R {rt['separate_fits']:8.1f}   Python {pt['setup']:8.1f}")
    print(f"  MCMC                  : R {rt['mcmc']:8.1f}   Python {pt['mcmc']:8.1f} "
          f"({pt['device']})   -> {rt['mcmc'] / pt['mcmc']:.1f}x")
    print(f"  jm() total            : R {rt['jm_total']:8.1f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
