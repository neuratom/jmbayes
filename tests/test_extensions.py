"""Features added in 0.6: Poisson, negative binomial and Student's t markers; strata;
delayed entry (left truncation). Agreement of the full models with JMbayes2 is checked
by benchmark/run_ext.py + run_ext_R.R + compare_cr.py; these tests check the pieces."""
import numpy as np
import pytest
import torch
from scipy import stats

import jmbayes as jmb
from jmbayes import families as F
from jmbayes import fits
from jmbayes import formula as fm
from jmbayes import var


# ------------------------------------------------------------------ densities
@pytest.mark.parametrize("family", ["poisson", "negative_binomial", "student_t"])
def test_log_densities_match_scipy(family):
    rng = np.random.default_rng(1)
    eta = rng.normal(1.0, 0.7, 50)
    mu = np.exp(eta)
    sigma, df = 3.5, 4.0
    if family == "student_t":
        y = eta + rng.standard_t(df, 50) * sigma
        ref = stats.t(df, loc=eta, scale=sigma).logpdf(y)
    else:
        y = rng.poisson(mu).astype(float)
        ref = (stats.poisson(mu).logpmf(y) if family == "poisson"
               else stats.nbinom(sigma, sigma / (sigma + mu)).logpmf(y))
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
    got = F.log_dens(family, t(y), t(eta), t(np.full(50, sigma)), df).numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(F.np_loglik(family, y, eta, np.log(sigma), df), ref,
                               rtol=1e-10, atol=1e-10)


# ------------------------------------------------------------------ separate fits vs R
@pytest.fixture(scope="module")
def pbc2():
    visits, patients = jmb.datasets.pbc2()
    visits, patients = visits.copy(), patients.copy()
    visits["bili_count"] = np.round(visits["serBilir"])
    patients["entry"] = 0.2 * patients["years"]
    return visits, patients


def test_cox_delayed_entry_and_strata_match_R(pbc2):
    _, p = pbc2
    W = np.column_stack([(p.sex == "female").astype(float), p.age])
    t, e, s0 = p.years.to_numpy(float), p.status2.to_numpy(float), p.entry.to_numpy(float)
    strata = (p.drug.astype(str) == "D-penicil").astype(int).to_numpy()
    # coxph(Surv(entry, years, status2) ~ sex + age [+ strata(drug)]) in R
    np.testing.assert_allclose(fits.fit_coxph(t, e, W, start=s0),
                               [-0.4258851625, 0.0143617442], atol=1e-7)
    np.testing.assert_allclose(fits.fit_coxph(t, e, W, strata),
                               [-0.4985773124, 0.0212838729], atol=1e-7)
    np.testing.assert_allclose(fits.fit_coxph(t, e, W, strata, s0),
                               [-0.4501727283, 0.0149285028], atol=1e-7)


def _glmm_data(visits, col):
    d = visits[visits[col].notna()]
    X = np.column_stack([np.ones(len(d)), d.year, (d.sex == "female").astype(float)])
    return d[col].to_numpy(float), X, np.ones((len(d), 1)), d.id.to_numpy()


def test_poisson_glmm_matches_GLMMadaptive(pbc2):
    y, X, Z, ids = _glmm_data(pbc2[0], "bili_count")
    f = fits.fit_glmm(y, X, Z, ids, "poisson")
    # mixed_model(bili_count ~ year + sex, random = ~ 1 | id, family = poisson())
    np.testing.assert_allclose(f.betas, [0.85205229, 0.16145876, -0.42665944], atol=2e-3)
    np.testing.assert_allclose(f.D[0, 0], 1.60655506, rtol=2e-3)
    assert f.log_sigma == -20.0


def test_negative_binomial_glmm_at_least_as_good_as_GLMMadaptive(pbc2):
    """GLMMadaptive stops slightly short of the maximum here; jmbayes's fit is close to
    it and has the higher likelihood (checked with 31 quadrature points too)."""
    y, X, Z, ids = _glmm_data(pbc2[0], "platelets")
    f = fits.fit_glmm(y, X, Z, ids, "negative_binomial")
    r = dict(betas=[5.41131951, -0.04208231, 0.11551163], phi=2.98284446, D=0.13852916)
    np.testing.assert_allclose(f.betas, r["betas"], atol=0.06)
    assert abs(f.log_sigma - r["phi"]) < 0.01 and abs(f.D[0, 0] - r["D"]) < 0.005
    uniq = list(dict.fromkeys(ids))
    m = fits._GLMM(y, X, Z, np.array([uniq.index(g) for g in ids]), 11, False,
                   "negative_binomial")
    theta = lambda b, phi, D: np.array([*b, phi, np.log(np.sqrt(D))])  # noqa: E731
    m.negloglik(theta(r["betas"], r["phi"], r["D"]), update_modes=True)
    nll_r = m.negloglik(theta(r["betas"], r["phi"], r["D"]))
    m.negloglik(theta(f.betas, f.log_sigma, f.D[0, 0]), update_modes=True)
    assert m.negloglik(theta(f.betas, f.log_sigma, f.D[0, 0])) <= nll_r


# ------------------------------------------------------------------ interface
def test_parse_surv_with_start_and_strata():
    t, e, cov, start, strata = fm.parse_surv("Surv(entry, years, status2) ~ sex + strata(drug)")
    assert (t, e, start, strata) == ("years", "status2", "entry", "drug")
    assert cov.variables() == ["sex"]
    assert fm.parse_surv("Surv(years, status2) ~ sex")[3:] == (None, None)
    ev = jmb.Survival("Surv(entry, years, status2) ~ sex + strata(drug)",
                      data=jmb.datasets.pbc2()[1].assign(entry=0.0)).to_event()
    assert (ev.start, ev.strata) == ("entry", "drug")


def test_family_errors(pbc2):
    visits, patients = pbc2
    with pytest.raises(ValueError, match="degrees of freedom"):
        jmb.Marker("prothrombin", family="student_t")
    with pytest.raises(ValueError, match="only used with"):
        jmb.Marker("prothrombin", family="gaussian", df=3)
    assert jmb.Marker("platelets", family="negative binomial").family == "negative_binomial"
    model = jmb.JointModel(jmb.Event("years", "status2", "sex"),
                           jmb.Marker("serBilir", family="poisson"), id="id", time="year")
    with pytest.raises(jmb.DataError, match="whole numbers"):
        model.fit(visits, patients, n_iter=200, n_burnin=100, verbose=False)


def test_entry_time_errors(pbc2):
    visits, patients = pbc2
    model = jmb.JointModel(jmb.Event("years", "status2", "sex", start="entry"),
                           jmb.Marker(var("serBilir").log()), id="id", time="year")
    bad = patients.assign(entry=patients["years"])
    with pytest.raises(jmb.DataError, match="before the follow-up"):
        model.fit(visits, bad, n_iter=200, n_burnin=100, verbose=False)
    with pytest.raises(NotImplementedError):
        jmb.Event("years", "status", "sex", time2="t2", start="entry")


# ------------------------------------------------------------------ fits
@pytest.fixture(scope="module")
def surv_fit(pbc2):
    visits, patients = pbc2
    year, sex = var("year"), var("sex")
    model = jmb.JointModel(jmb.Event("years", "status2", sex, start="entry", strata="drug"),
                           jmb.Marker(var("serBilir").log(), fixed=year * sex, random=year),
                           id="id", time="year")
    return model.fit(visits, patients, n_iter=400, n_burnin=100, n_chains=2, device="cpu",
                     verbose=False)


def test_strata_and_delayed_entry_fit(surv_fit, pbc2):
    md = surv_fit.md
    assert md.GK_k == 7                       # as jm() with Surv(start, stop, event)
    assert md.strata_levels == ["placebo", "D-penicil"]
    assert not np.allclose(md.knots_strata[0], md.knots_strata[1])
    assert surv_fit.blocks["bs_gammas"][0].shape[2] == 2 * 11
    assert surv_fit.blocks["tau_bs_gammas"][1] == ["tau_bs_gammas_1", "tau_bs_gammas_2"]
    assert np.isfinite(list(surv_fit.fit_statistics.values())).all()
    visits, _ = pbc2
    nd = visits[visits.id.isin([25, 93]) & (visits.year < 5)].copy()
    nd["years"], nd["status2"], nd["entry"] = 5.0, 0, 1.0
    out = surv_fit.predict(nd, times=[6, 8], n_samples=30)
    assert out["pred_CIF"].between(0, 1).all() and len(out) == 6
    with pytest.raises(ValueError, match="unknown 'drug'"):
        surv_fit.predict(nd.assign(drug="other"), times=[6], n_samples=10)


@pytest.fixture(scope="module")
def fam_fit(pbc2):
    visits, patients = pbc2
    year, sex = var("year"), var("sex")
    model = jmb.JointModel(
        jmb.Event("years", "status2", sex),
        [jmb.Marker("platelets", fixed=year + sex, family="negative_binomial"),
         jmb.Marker("bili_count", fixed=year + sex, family="poisson"),
         jmb.Marker("prothrombin", fixed=year + sex, family="student_t", df=4)],
        id="id", time="year")
    return model.fit(visits, patients, n_iter=400, n_burnin=100, n_chains=2, device="cpu",
                     verbose=False)


def test_new_families_fit_and_predict(fam_fit, pbc2):
    assert [o.has_sigma for o in fam_fit.md.outcomes] == [True, False, True]
    assert "sigma" in fam_fit.coefficients("platelets").index
    assert "sigma" not in fam_fit.coefficients("bili_count").index
    assert np.isfinite(list(fam_fit.fit_statistics.values())).all()
    visits, _ = pbc2
    nd = visits[(visits.id == 93) & (visits.year < 5)].copy()
    nd["years"], nd["status2"] = 5.0, 0
    risk = fam_fit.predict(nd, times=[6, 8], n_samples=30)
    assert risk["pred_CIF"].between(0, 1).all()
    long = fam_fit.predict(nd, process="longitudinal", n_samples=30)
    assert (long["pred_platelets"] > 0).all() and (long["pred_bili_count"] > 0).all()
