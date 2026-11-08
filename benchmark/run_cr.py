"""jmbayes side of the competing-risks comparison (same model as run_cr_R.R).

Usage: python run_cr.py <out.json> [--n-iter 15000 --n-burnin 2500 --n-chains 8 --seed 123]
"""
import argparse
import json

import pandas as pd

import jmbayes as jmb
from jmbayes import summary as sm
from jmbayes import var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n-iter", type=int, default=15000)
    ap.add_argument("--n-burnin", type=int, default=2500)
    ap.add_argument("--n-chains", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--dtype", default="float32")
    a = ap.parse_args()
    visits, patients = jmb.datasets.pbc2()
    # causes in the factor's level order, as JMbayes2 orders strata(CR)
    patients["status"] = pd.Categorical(patients["status"], ["alive", "transplanted", "dead"])
    year, sex = var("year"), var("sex")
    model = jmb.JointModel(
        event=jmb.Event("years", "status", sex, censored="alive"),
        markers=[jmb.Marker(var("serBilir").log(), fixed=year * sex, random=year),
                 jmb.Marker("prothrombin", fixed=year * sex, random=year)],
        id="id", time="year")
    fit = model.fit(visits, patients, n_iter=a.n_iter, n_burnin=a.n_burnin, n_thin=1,
                    n_chains=a.n_chains, seed=a.seed, dtype=a.dtype, verbose=False)
    res = {k: sm.table(*v) for k, v in fit.blocks.items()}
    res["fit_stats"] = {k: float(v) for k, v in fit.fit_statistics.items()}
    nd = visits[visits.id.isin([25, 93]) & (visits.year < 5)].copy()
    nd["years"] = 5.0
    nd["status"] = "alive"
    pr = fit.predict(nd, times=[6, 8, 10])
    res["predict"] = pr.to_dict(orient="list")
    res["mcmc_time"] = fit.timing["mcmc"]
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"done: MCMC {fit.timing['mcmc']:.1f}s")


if __name__ == "__main__":
    main()
