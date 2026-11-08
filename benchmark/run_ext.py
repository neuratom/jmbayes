"""jmbayes side of the comparison for the features added in 0.6 (same models as
run_ext_R.R): 'surv' (delayed entry + strata) and 'fam' (negative binomial, Poisson and
Student's t markers).

Usage: python run_ext.py <surv|fam> <out.json> [--n-iter 15000 --n-burnin 2500
       --n-chains 8 --seed 123]
"""
import argparse
import json

import numpy as np

import jmbayes as jmb
from jmbayes import summary as sm
from jmbayes import var


def model_and_data(which):
    visits, patients = jmb.datasets.pbc2()
    visits, patients = visits.copy(), patients.copy()
    visits["bili_count"] = np.round(visits["serBilir"])
    year, sex = var("year"), var("sex")
    if which == "surv":
        patients["entry"] = 0.2 * patients["years"]
        event = jmb.Event("years", "status2", sex, start="entry", strata="drug")
        markers = [jmb.Marker(var("serBilir").log(), fixed=year * sex, random=year),
                   jmb.Marker("prothrombin", fixed=year * sex, random=year)]
    elif which == "fam":
        event = jmb.Event("years", "status2", sex)
        markers = [jmb.Marker("platelets", fixed=year * sex, random=year,
                              family="negative_binomial"),
                   jmb.Marker("bili_count", fixed=year + sex, random=year, family="poisson"),
                   jmb.Marker("prothrombin", fixed=year * sex, random=year,
                              family="student_t", df=4)]
    else:
        raise SystemExit("model must be 'surv' or 'fam'")
    return jmb.JointModel(event, markers, id="id", time="year"), visits, patients


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--n-iter", type=int, default=15000)
    ap.add_argument("--n-burnin", type=int, default=2500)
    ap.add_argument("--n-chains", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--initial-fits", default=None,
                    help="separate fits exported from R (export_initfits_ext.R)")
    a = ap.parse_args()
    model, visits, patients = model_and_data(a.model)
    init = None
    if a.initial_fits:
        with open(a.initial_fits) as f:
            init = json.load(f)
    fit = model.fit(visits, patients, n_iter=a.n_iter, n_burnin=a.n_burnin, n_thin=1,
                    n_chains=a.n_chains, seed=a.seed, dtype=a.dtype, verbose=False,
                    initial_fits=init)
    res = {k: sm.table(*v) for k, v in fit.blocks.items()}
    res["fit_stats"] = {k: float(v) for k, v in fit.fit_statistics.items()}
    res["GK_k"] = fit.md.GK_k
    res["mcmc_time"] = fit.timing["mcmc"]
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"done: MCMC {fit.timing['mcmc']:.1f}s")


if __name__ == "__main__":
    main()
