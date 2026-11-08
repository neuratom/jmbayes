"""jmbayes side of the comparisons for the features added in 0.6 and 0.7 (same models
as run_ext_R.R and run_ext2_R.R):
  surv  delayed entry + strata            fam   negative binomial, Poisson, Student's t
  tvc   time-varying covariate (counting-process rows)
  icens interval censoring                crx   competing risks + entry + strata
  rec   recurrent + terminal events (gap timescale, frailty; the JMbayes2 vignette data)
  fam2  gamma, beta and censored normal   fam3  binomial with trials, beta-binomial
  bystr strata with stratum-specific effects
Data sets: ext_data.py and make_rec_data.R (benchmark/data_ext/).

Usage: python run_ext.py <model> <out.json> [--n-iter 15000 --n-burnin 2500
       --n-chains 8 --seed 123 --initial-fits R_<model>_initfits.json]
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
        return _model_07(which, year, sex)
    return jmb.JointModel(event, markers, id="id", time="year"), visits, patients


def _model_07(which, year, sex):
    import pandas as pd
    from pathlib import Path
    d = Path(__file__).parent / "data_ext"
    visits, _ = jmb.datasets.pbc2()
    patients = pd.read_csv(d / "patients.csv")
    lbili = jmb.Marker(var("serBilir").log(), fixed=year * sex, random=year)
    time = "year"
    if which == "tvc":
        surv = pd.read_csv(d / "tvc_surv.csv")
        event = jmb.Event("tstop", "status2", sex + var("edema_tv"), start="tstart")
        markers, patients = [lbili], surv
    elif which == "icens":
        patients = pd.read_csv(d / "icens_surv.csv")
        event = jmb.Event("t1", "ic_status", sex, time2="t2")
        markers = [lbili]
    elif which == "crx":
        event = jmb.Event("years", "status", sex, censored="alive", start="entry",
                          strata="drug")
        markers = [lbili]
    elif which == "rec":
        visits = pd.read_csv(d / "rec_long.csv")
        patients = pd.read_csv(d / "rec_surv.csv")
        t = var("time")
        event = jmb.Event("tstop", "status", var("group") + var("age"), start="tstart",
                          strata="strata", by_strata=True, recurrent="gap")
        markers = [jmb.Marker("y", fixed=t, random=t)]
        time = "time"
    elif which == "fam2":
        visits = pd.read_csv(d / "fam_visits.csv")
        markers = [jmb.Marker("serBilir", fixed=year * sex, random=year, family="gamma"),
                   jmb.Marker("alb_p", fixed=year + sex, random=year, family="beta"),
                   jmb.Marker("lalk", fixed=year + sex, random=year,
                              family="censored_normal", censoring="lalk_cens")]
        event = jmb.Event("years", "status2", sex)
    elif which == "fam3":
        visits = pd.read_csv(d / "fam_visits.csv")
        markers = [jmb.Marker("count2", fixed=year + sex, family="binomial", trials=2),
                   jmb.Marker("count3", fixed=year + sex, family="beta_binomial", trials=3)]
        event = jmb.Event("years", "status2", sex)
    elif which == "bystr":
        event = jmb.Event("years", "status2", sex, strata="drug", by_strata=True)
        markers = [lbili, jmb.Marker("prothrombin", fixed=year * sex, random=year)]
    else:
        raise SystemExit(f"unknown model {which!r}")
    v0, _ = jmb.datasets.pbc2()
    for df in (visits, patients):         # CSV loses the factor level order (R's)
        for col in ("sex", "drug", "status"):
            if col in df and col in v0 and hasattr(v0[col], "cat") and                     not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.Categorical(df[col].astype(str),
                                         categories=list(v0[col].cat.categories))
    return jmb.JointModel(event, markers, id="id", time=time), visits, patients


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
    if fit.marginal_fit_statistics:
        res["fit_stats_marginal"] = {k: float(v)
                                     for k, v in fit.marginal_fit_statistics.items()}
    res["GK_k"] = fit.md.GK_k
    res["mcmc_time"] = fit.timing["mcmc"]
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"done: MCMC {fit.timing['mcmc']:.1f}s")


if __name__ == "__main__":
    main()
