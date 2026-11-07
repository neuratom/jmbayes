"""Fit the README joint model on pbc2, fully in Python.

R equivalent (README.Rmd):
    CoxFit <- coxph(Surv(years, status2) ~ sex, data = pbc2.id)
    fm1 <- lme(log(serBilir) ~ year * sex, data = pbc2, random = ~ year | id)
    fm2 <- lme(prothrombin ~ year * sex, data = pbc2, random = ~ year | id)
    fm3 <- mixed_model(ascites ~ year + sex, data = pbc2,
                       random = ~ year | id, family = binomial())
    jointFit <- jm(CoxFit, list(fm1, fm2, fm3), time_var = "year",
                   n_iter = 12000L, n_burnin = 2000L, n_thin = 5L)

Usage:  python run_pbc2.py [--device cpu|cuda] [--n-iter N] [--n-burnin N] [--n-thin N]
"""
import argparse
import json
import pathlib
import time

import pandas as pd
import torch
import jmbayes as jmb
from jmbayes import var

from jmbayes.model import jm_setup
from jmbayes.sampler import JointModelSampler
from jmbayes import summary as sm

HERE = pathlib.Path(__file__).resolve().parent


year, sex = var("year"), var("sex")
MARKERS = [jmb.Marker(var("serBilir").log(), fixed=year * sex, random=year),
           jmb.Marker("prothrombin", fixed=year * sex, random=year),
           jmb.Marker("ascites", fixed=year + sex, random=year, family="binomial")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-iter", type=int, default=12000)
    ap.add_argument("--n-burnin", type=int, default=2000)
    ap.add_argument("--n-thin", type=int, default=5)
    ap.add_argument("--n-chains", type=int, default=3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--init-from", default=None,
                    help="JSON of separate fits exported from R (benchmark/export_initfits.R); "
                         "use them instead of the Python fits")
    ap.add_argument("--replicate", type=int, default=1,
                    help="copy pbc2 this many times with new ids (benchmarking larger data)")
    ap.add_argument("--out", default=str(HERE / "results_pbc2.json"))
    ap.add_argument("--dtype", default="float32", choices=["float64", "float32"],
                    help="floating-point precision of the sampler")
    a = ap.parse_args()

    pbc2, pbc2_id = jmb.datasets.pbc2()
    if a.replicate > 1:
        def rep(d):
            parts = []
            for r in range(a.replicate):
                c = d.copy()
                c["id"] = c["id"].astype(int) + 10000 * r
                parts.append(c)
            return pd.concat(parts, ignore_index=True)
        pbc2, pbc2_id = rep(pbc2), rep(pbc2_id)

    t0 = time.perf_counter()
    md = jm_setup(
        pbc2, pbc2_id,
        long_models=[dict(response=m.response, fixed=m.fixed, random=m.random, family=m.family)
                     for m in MARKERS],
        surv=dict(time="years", event="status2", covariates=jmb.Event("years", "status2",
                                                                       "sex").covariates),
        id_var="id", time_var="year",
        initial_fits=json.loads(pathlib.Path(a.init_from).read_text()) if a.init_from else None)
    t_setup = time.perf_counter() - t0
    print(f"setup (initial fits + design matrices): {t_setup:.1f}s")

    smp = JointModelSampler(md, n_chains=a.n_chains, device=a.device, seed=a.seed,
                            dtype=getattr(torch, a.dtype))
    print(f"running on {smp.dev}")
    out = smp.run(n_iter=a.n_iter, n_burnin=a.n_burnin)
    blocks = sm.collect(md, out, n_thin=a.n_thin)
    ll_mean = sm.loglik_at_means(smp, md, blocks, out)
    fit = sm.fit_stats(out["fitstats_acc"], out["n_keep"], ll_mean)
    sm.print_summary(md, blocks, fit, out["elapsed"], a.n_chains, a.n_iter, a.n_burnin,
                     a.n_thin)

    res = {k: sm.table(*v) for k, v in blocks.items()}
    res["fit_stats"] = fit
    res["timing"] = dict(setup=t_setup, mcmc=out["elapsed"], device=out["device"],
                         gpu_peak_mb=out["gpu_peak_mb"])
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
