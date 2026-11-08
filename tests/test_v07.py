"""Features added in 0.7: beta, gamma, unit Lindley, censored normal, beta-binomial and
binomial-with-trials markers; time-varying covariates (counting-process rows); interval
and left censoring; competing risks with entry times and strata; stratum-specific effects;
recurrent events with a frailty; survival models without covariates; marginal fit
statistics. Agreement of whole models with JMbayes2: benchmark/run_ext.py +
run_ext2_R.R + compare_cr.py. These tests check the pieces."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

import jmbayes as jmb
from jmbayes import families as F
from jmbayes import fits
from jmbayes import formula as fm
from jmbayes import var

BENCH = Path(__file__).resolve().parents[1] / "benchmark"
DATA = BENCH / "data_ext"
t64 = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float64)  # noqa: E731


# ------------------------------------------------------------------ densities
def _case(family, rng):
    eta = rng.normal(0.3, 0.6, 60)
    sig = {"beta": 6.0, "gamma": 2.5, "censored_normal": 0.8, "beta_binomial": 4.0}.get(
        family, 1.0)
    mu = 1 / (1 + np.exp(-eta))
    aux = None
    if family == "beta":
        y = rng.beta(mu * sig, (1 - mu) * sig)
        ref = stats.beta(mu * sig, (1 - mu) * sig).logpdf(y)
    elif family == "gamma":
        m = np.exp(eta)
        y = rng.gamma(sig, m / sig)
        ref = stats.gamma(sig, scale=m / sig).logpdf(y)
    elif family == "unit_lindley":
        y = rng.uniform(0.05, 0.95, 60)
        th = 1 / mu - 1
        ref = 2 * np.log(th) - np.log1p(th) - 3 * np.log1p(-y) - th * y / (1 - y)
    elif family == "censored_normal":
        y = eta + rng.normal(0, sig, 60)
        aux = rng.choice([0.0, 1.0, 2.0], 60)
        d = stats.norm(eta, sig)
        ref = np.where(aux == 0, d.logpdf(y), np.where(aux == 1, d.logcdf(y), d.logsf(y)))
    elif family == "beta_binomial":
        aux = rng.integers(1, 6, 60).astype(float)
        y = rng.binomial(aux.astype(int), mu).astype(float)
        ref = stats.betabinom(aux, mu * sig, (1 - mu) * sig).logpmf(y)
    else:                                                     # binomial with trials
        aux = rng.integers(1, 6, 60).astype(float)
        y = rng.binomial(aux.astype(int), mu).astype(float)
        ref = stats.binom(aux, mu).logpmf(y)
    return y, eta, sig, aux, ref


FAMS = ["beta", "gamma", "unit_lindley", "censored_normal", "beta_binomial", "binomial"]


@pytest.mark.parametrize("family", FAMS)
def test_log_densities_match_scipy(family):
    y, eta, sig, aux, ref = _case(family, np.random.default_rng(2))
    pre = F.precompute(family, t64(y), None if aux is None else t64(aux))
    got = F.log_dens(family, t64(y), t64(eta), t64(np.full(60, sig)), 0.0, pre).numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9)
    phi = np.log(sig)
    np.testing.assert_allclose(F.np_loglik(family, y, eta, phi, 0.0, aux), ref,
                               rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("family", FAMS)
def test_scores_are_derivatives(family):
    y, eta, sig, aux, _ = _case(family, np.random.default_rng(3))
    phi, h = np.log(sig), 1e-6
    num = (F.np_loglik(family, y, eta + h, phi, 0.0, aux)
           - F.np_loglik(family, y, eta - h, phi, 0.0, aux)) / (2 * h)
    sc, w = F.np_score_weight(family, y, eta, phi, 0.0, aux)
    np.testing.assert_allclose(sc, num, rtol=1e-5, atol=1e-6)
    assert (w > 0).all()


# ------------------------------------------------------------------ separate fits vs R
def test_counting_process_cox_and_interval_weibull_match_R():
    tv = pd.read_csv(DATA / "tvc_surv.csv")
    W = np.column_stack([(tv.sex == "female").astype(float), tv.edema_tv])
    g = fits.fit_coxph(tv.tstop.to_numpy(float), tv.status2.to_numpy(float), W,
                       start=tv.tstart.to_numpy(float))
    # coxph(Surv(tstart, tstop, status2) ~ sex + edema_tv, data = tvc_surv)
    np.testing.assert_allclose(g, REF["cox_tvc"], atol=1e-7)
    ic = pd.read_csv(DATA / "icens_surv.csv")
    d = ic.ic_status.to_numpy()
    T_left = np.where(d >= 2, ic.t1, 0.0)
    T_right = np.where(d == 3, ic.t2, np.where(d == 2, 0.0, ic.t1))
    g = fits.fit_weibull_interval(T_left, T_right, d,
                                  (ic.sex == "female").astype(float).to_numpy()[:, None])
    # survreg(Surv(t1, t2, ic_status, type = "interval") ~ sex): -coef[-1] / scale
    np.testing.assert_allclose(g, REF["weibull_icens"], atol=1e-5)


REF = dict(cox_tvc=[np.nan, np.nan], weibull_icens=[np.nan])     # filled below


def _load_ref():
    f = BENCH / "R_v07_reference.json"
    if f.exists():
        REF.update(json.loads(f.read_text()))


_load_ref()


# ------------------------------------------------------------------ interface
def test_parse_surv_full():
    d = fm.parse_surv_full('Surv(t1, t2, st, type = "interval") ~ sex')
    assert (d["time"], d["time2"], d["event"], d["start"]) == ("t1", "t2", "st", None)
    d = fm.parse_surv_full("Surv(a, b, s) ~ (group + age):strata(kind)")
    assert d["strata"] == "kind" and d["by_strata"] and d["start"] == "a"
    assert d["covariates"].variables() == ["age", "group"]
    d = fm.parse_surv_full("Surv(years, status2) ~ 1")
    assert d["covariates"].terms == ()


def test_rc_setup_matches_R():
    d = pd.read_csv(DATA / "rec_surv.csv")              # made by rc_setup() in R
    rc = d[d.strata == "R"][["id", "tstart", "tstop", "status", "group", "age"]]
    tr = d[d.strata == "T1"][["id", "tstop", "status", "group", "age"]]
    o = jmb.rc_setup(rc, tr, start="tstart", stop="tstop")
    r = d.sort_values(["id", "strata", "tstart"], kind="mergesort").reset_index(drop=True)
    cols = ["id", "tstart", "tstop", "status"]
    np.testing.assert_allclose(o[cols].to_numpy(float), r[cols].to_numpy(float))
    assert list(o.strata.astype(str)) == list(r.strata)


def test_marker_option_errors():
    with pytest.raises(ValueError, match="censoring"):
        jmb.Marker("y", family="censored_normal")
    with pytest.raises(ValueError, match="trials"):
        jmb.Marker("y", family="poisson", trials=3)
    with pytest.raises(ValueError, match="start"):
        jmb.Event("t", "s", "x", recurrent="gap")
    with pytest.raises(ValueError, match="strata"):
        jmb.Event("t", "s", "x", by_strata=True)


# ------------------------------------------------------------------ fits
@pytest.fixture(scope="module")
def pbc2():
    return jmb.datasets.pbc2()


def _fit(event, markers, visits, patients, time="year", **kw):
    m = jmb.JointModel(event, markers, id="id", time=time)
    return m.fit(visits, patients, n_iter=300, n_burnin=100, n_chains=2, device="cpu",
                 verbose=False, **kw)


def test_no_covariates_and_marginal_stats(pbc2):
    visits, patients = pbc2
    f = _fit(jmb.Event("years", "status2"), jmb.Marker(var("serBilir").log(),
                                                        fixed=var("year"), random=var("year")),
             visits, patients)
    assert len(f.blocks["gammas"][1]) == 0
    assert set(f.marginal_fit_statistics) == {"DIC", "pD", "LPML", "WAIC"}
    assert np.isfinite(list(f.marginal_fit_statistics.values())).all()
    assert f.marginal_fit_statistics["pD"] < f.fit_statistics["pD"]


def test_time_varying_covariate_and_interval_censoring(pbc2):
    visits, _ = pbc2
    year, sex = var("year"), var("sex")
    mk = jmb.Marker(var("serBilir").log(), fixed=year, random=year)
    tv = pd.read_csv(DATA / "tvc_surv.csv")
    f = _fit(jmb.Event("tstop", "status2", sex + var("edema_tv"), start="tstart"), mk,
             visits, tv)
    assert len(f.md.event) == len(tv)
    nd = visits[visits.id == 2].copy()
    nd["tstart"], nd["tstop"], nd["status2"], nd["edema_tv"] = 0.0, 3.0, 0, 0
    nd = nd[nd.year < 3]
    out = f.predict(nd, times=[4, 5], n_samples=20)
    assert out.pred_CIF.between(0, 1).all()
    ic = pd.read_csv(DATA / "icens_surv.csv")
    f = _fit(jmb.Event("t1", "ic_status", sex, time2="t2"), mk, visits, ic)
    assert (f.md.delta == 3).sum() == (ic.ic_status == 3).sum()
    assert np.isfinite(list(f.fit_statistics.values())).all()


def test_competing_risks_with_entry_and_strata_predict(pbc2):
    visits, patients = pbc2
    patients = patients.assign(entry=0.2 * patients.years)
    year = var("year")
    f = _fit(jmb.Event("years", "status", var("sex"), censored="alive", start="entry",
                       strata="drug"),
             jmb.Marker(var("serBilir").log(), fixed=year, random=year), visits, patients)
    assert f.md.n_strata == 4 and len(f.md.knots_strata) == 4
    nd = visits[visits.id.isin([2, 25])].copy()
    nd = nd[nd.year < 4]
    nd["years"], nd["status"], nd["entry"] = 4.0, "alive", 0.5
    out = f.predict(nd, times=[5, 7], n_samples=20)
    tot = out.groupby(["id", "years"]).pred_CIF.sum()
    assert out.pred_CIF.between(0, 1).all() and (tot <= 1).all()


def test_by_strata_and_recurrent(pbc2):
    visits, patients = pbc2
    year, sex = var("year"), var("sex")
    f = _fit(jmb.Event("years", "status2", sex, strata="drug", by_strata=True),
             jmb.Marker(var("serBilir").log(), fixed=year, random=year), visits, patients)
    assert f.blocks["gammas"][1] == ["sexfemale [placebo]", "sexfemale [D-penicil]"]
    assert len(f.blocks["alphas"][1]) == 2
    nd = visits[visits.id == 2].copy()
    nd["years"], nd["status2"] = 3.0, 0
    assert f.predict(nd[nd.year < 3], times=[4], n_samples=10).pred_CIF.between(0, 1).all()
    lv = pd.read_csv(DATA / "rec_long.csv")
    rs = pd.read_csv(DATA / "rec_surv.csv")
    t = var("time")
    f = _fit(jmb.Event("tstop", "status", var("group") + var("age"), start="tstart",
                       strata="strata", by_strata=True, recurrent="gap"),
             jmb.Marker("y", fixed=t, random=t), lv, rs, time="time")
    tab = f.survival_table()
    assert {"sigma_frailty", "frailty:T1"} <= set(tab.index)
    assert np.isfinite(list(f.marginal_fit_statistics.values())).all()
    with pytest.raises(NotImplementedError):
        f.predict(lv[lv.id == 1].assign(tstart=0.0, tstop=1.0, status=0, strata="R",
                                       group=0, age=50.0), n_samples=5)


def test_new_family_fit_and_predict(pbc2):
    v = pd.read_csv(DATA / "fam_visits.csv")
    _, patients = pbc2
    v["sex"] = pd.Categorical(v.sex, categories=["male", "female"])
    year, sex = var("year"), var("sex")
    f = _fit(jmb.Event("years", "status2", sex),
             [jmb.Marker("lalk", fixed=year, random=year, family="censored_normal",
                         censoring="lalk_cens"),
              jmb.Marker("count3", fixed=year, family="beta_binomial", trials=3),
              jmb.Marker("alb_p", fixed=year, family="beta")], v, patients)
    nd = v[v.id == 2].copy()
    nd["years"], nd["status2"] = 3.0, 0
    nd = nd[nd.year < 3]
    long = f.predict(nd, process="longitudinal", n_samples=10)
    assert long.pred_count3.between(0, 1).all() and long.pred_alb_p.between(0, 1).all()
    assert f.predict(nd, times=[4], n_samples=10).pred_CIF.between(0, 1).all()


# ------------------------------------------------------------------ GLMMs vs GLMMadaptive
def _glmm_cases():
    import sys
    sys.path.insert(0, str(BENCH))
    import run_ext
    out = []
    for name in ("fam2", "fam3"):
        f = BENCH / f"R_{name}_initfits.json"
        if not f.exists():
            continue
        R = json.loads(f.read_text())
        model, visits, _ = run_ext.model_and_data(name)
        out += [(m, r, visits) for m, r in zip(model.markers, R["outcomes"])]
    return out


@pytest.mark.parametrize("case", _glmm_cases(), ids=lambda c: c[0].family)
def test_glmm_at_least_as_good_as_GLMMadaptive(case):
    """Beta and binomial-with-trials fits match GLMMadaptive; for gamma, beta-binomial
    and censored normal jmbayes reaches a higher likelihood (GLMMadaptive stops short;
    for the censored normal by a wide margin, with a non-positive-definite Hessian)."""
    m, r, visits = case
    need = [m.response.var] + ([m.aux] if isinstance(m.aux, str) else [])
    d = visits[visits[need].notna().all(axis=1)]
    lev = {}
    X, Z = m.fixed.build(d, lev)[0], m.random.build(d, lev)[0]
    aux = (None if m.aux is None else d[m.aux].to_numpy(float) if isinstance(m.aux, str)
           else np.full(len(d), float(m.aux)))
    y = d[m.response.var].to_numpy(float)
    f = fits.fit_glmm(y, X, Z, d.id.to_numpy(), m.family, aux=aux)
    uniq = list(dict.fromkeys(d.id))
    g = fits._GLMM(y, X, Z, np.array([uniq.index(i) for i in d.id]), 11, False, m.family,
                   0.0, aux)

    def nll(b, phi, D):
        L = np.linalg.cholesky(np.atleast_2d(D))
        tri = [np.log(L[i, j]) if i == j else L[i, j] for i, j in zip(*np.tril_indices(len(L)))]
        th = np.array([*b, *([phi] if g.n_phi else []), *tri])
        g.modes[:] = 0
        g.negloglik(th, update_modes=True)
        return g.negloglik(th)

    n_r = nll(np.asarray(r["betas"]), r["log_sigma"], np.array(r["D"]))
    n_p = nll(f.betas, f.log_sigma, f.D)
    assert n_p <= n_r + 1e-4
    if m.family in ("beta", "binomial"):
        np.testing.assert_allclose(f.betas, r["betas"], atol=6e-3)
        np.testing.assert_allclose(f.D, np.array(r["D"]).reshape(f.D.shape), rtol=5e-3,
                                   atol=1e-4)
