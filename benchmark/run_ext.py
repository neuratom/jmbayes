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
    elif which in ("aids", "prothro"):
        return _model_other(which)
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
    elif which in MODELS_08:
        return _model_08(which, year, sex)
    else:
        raise SystemExit(f"unknown model {which!r}")
    v0, _ = jmb.datasets.pbc2()
    for df in (visits, patients):         # CSV loses the factor level order (R's)
        for col in ("sex", "drug", "status"):
            if col in df and col in v0 and hasattr(v0[col], "cat") and                     not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.Categorical(df[col].astype(str),
                                         categories=list(v0[col].cat.categories))
    return jmb.JointModel(event, markers, id="id", time=time), visits, patients


def _model_other(which):
    """Data other than PBC (make_other_data.R; same models in run_ext3_R.R)."""
    import pandas as pd
    from pathlib import Path
    d = Path(__file__).parent / "data_ext"
    lv = pd.read_csv(d / f"{which}_long.csv")
    sv = pd.read_csv(d / f"{which}_surv.csv")
    levels = dict(drug=["ddC", "ddI"], prevOI=["noAIDS", "AIDS"],
                  treat=["placebo", "prednisone"])
    for df in (lv, sv):
        for c, lev in levels.items():
            if c in df:
                df[c] = pd.Categorical(df[c], categories=lev)
    if which == "aids":
        t = var("obstime")
        event = jmb.Event("Time", "death", var("drug") + var("prevOI"))
        marker = jmb.Marker("CD4", fixed=t + (t & var("drug")), random=t)
        return jmb.JointModel(event, marker, id="patient", time="obstime"), lv, sv
    t = var("time")
    event = jmb.Event("Time", "death", var("treat"))
    marker = jmb.Marker("pro", fixed=t * var("treat"), random=t)
    return jmb.JointModel(event, marker, id="id", time="time"), lv, sv


MODELS_08 = ("f8a", "f8b", "f8c", "tva", "bhw", "bhp", "bhn", "ms", "wts", "pen")
# fit() options per 0.8 model (as jm() arguments in run_ext3_R.R)
FIT_08 = dict(bhw=dict(base_hazard="Weibull"), bhp=dict(base_hazard="piecewise constant"),
              bhn=dict(base_hazard="natural cubic spline"),
              ms=dict(base_hazard="Weibull"),
              pen=dict(penalty_alphas="horseshoe", penalty_gammas="ridge"))


def _model_08(which, year, sex):
    """0.8 models (functional forms, time-varying association, baseline hazards,
    multi-state, weights, shrinkage priors); see run_ext3_R.R."""
    import pandas as pd
    from pathlib import Path
    from jmbayes import assoc as A
    from jmbayes import ns
    d = Path(__file__).parent / "data_ext"
    visits, patients = jmb.datasets.pbc2()
    patients = patients.copy()
    time = "year"
    resp = var("serBilir").log()
    ff = lambda txt: A.parse(txt, ["log(serBilir)"])["log(serBilir)"]  # noqa: E731
    spl = dict(fixed=ns(year, 2) + sex, random=ns(year, 2))   # (see NOTES: HC + splines)
    lin = dict(fixed=year * sex, random=year)
    event = jmb.Event("years", "status2", sex)
    if which == "f8a":
        markers = [jmb.Marker(resp, **spl, association=ff(
            "~ value(log(serBilir)) + velocity(log(serBilir), eps = 0.01, direction = "
            "'backward') + area(log(serBilir), time_window = 2)"))]
    elif which == "f8b":
        markers = [jmb.Marker(resp, **spl, association=ff(
            "~ vexpit(value(log(serBilir))) + Dexp(slope(log(serBilir))) + "
            "acceleration(log(serBilir))"))]
    elif which == "f8c":
        markers = [jmb.Marker(resp, **spl, association=ff(
            "~ value(log(serBilir)):sex + Delta(log(serBilir), time_window = 3)"))]
    elif which == "tva":
        markers = [jmb.Marker(resp, **lin, association=ff(
            "~ value(log(serBilir)) * ns(year, k = c(3, 6, 9), B = c(0, 14.5))"))]
    elif which in ("bhw", "bhp", "bhn"):
        markers = [jmb.Marker(resp, **lin)]
    elif which == "wts":
        patients["w"] = (patients["sex"] == "female").astype(float) + 1.0
        event = jmb.Event("years", "status2", sex, weights="w")
        markers = [jmb.Marker(resp, **lin)]
    elif which == "pen":
        event = jmb.Event("years", "status2", sex + var("age") + var("drug"))
        markers = [jmb.Marker(resp, **lin, association=["value", "slope", "area"]),
                   jmb.Marker("prothrombin", fixed=year * sex, random=year)]
    elif which == "ms":
        visits = pd.read_csv(d / "ms_long.csv")
        patients = pd.read_csv(d / "ms_surv.csv")
        t = var("time")
        event = jmb.Event("Tstop", "status", var("X"), start="Tstart", strata="transition",
                          by_strata=True)
        markers = [jmb.Marker("y", fixed=t, random=t)]
        time = "time"
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
                    initial_fits=init, **FIT_08.get(a.model, {}))
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
