"""Features added in 0.8: the remaining functional forms (velocity, acceleration, Delta,
coefs; slope and area options; IE_time), transformations, products of forms and
interactions with covariates (time-varying associations), baseline-hazard bases
(Weibull, piecewise constant, natural splines, log time), multi-state models
(ms_setup), case weights, shrinkage priors, and the hierarchical-centering fix for
splines. Whole models are compared with JMbayes2 by benchmark/run_ext.py +
run_ext3_R.R + compare_cr.py."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import jmbayes as jmb
from jmbayes import assoc as A
from jmbayes import fits, ns, var
from jmbayes.model import (TimeDesigner, base_hazard_config, baseline_basis, baseline_knots,
                           jm_setup)

BENCH = Path(__file__).resolve().parents[1] / "benchmark"
DATA = BENCH / "data_ext"
REF = json.loads((BENCH / "R_v08_reference.json").read_text())


# ------------------------------------------------------------------ specification
def test_parse_functional_forms_like_R():
    r = "log(serBilir)"
    d = A.parse("~ value(log(serBilir)) * sex + vexpit(slope(log(serBilir), eps = 0.01)) "
                "+ Dexp(slope(log(serBilir))) + area(log(serBilir), time_window = 2) + "
                "velocity(log(serBilir), direction = 'backward')", [r])[r]
    labels = [a.label(r) for a in d]
    # model.matrix() order: main terms, then interactions (Dexp counts as two factors)
    assert labels == ["value(log(serBilir))", "vexpit(slope(log(serBilir), eps = 0.01))",
                      "area(log(serBilir), time_window = 2)",
                      'velocity(log(serBilir), direction = "backward")',
                      "value(log(serBilir))",
                      "vexp(value(log(serBilir))):slope(log(serBilir))"]
    assert d[4].by is not None and d[0].by is None
    with pytest.raises(NotImplementedError):
        A.parse("~ foo(log(serBilir))", [r])


def test_transforms_match_numpy():
    x = np.linspace(-2, 2, 9)
    for t in A.TRANSFORMS:
        xx = np.abs(x) + 0.1 if t in ("log", "log2", "log10", "sqrt") else x
        np.testing.assert_allclose(A.torch_transform(t, torch.tensor(xx)).numpy(),
                                   A.np_transform(t, xx), rtol=1e-12)


def test_form_designs_on_a_quadratic():
    """For y = t^2 (a single-row 'subject') the forms have closed forms."""
    d = pd.DataFrame({"_sid": [0, 0], "t": [0.0, 1.0]})
    des = TimeDesigner(d, "t")
    terms = jmb.design.Terms((((jmb.design.Column("t")),) * 2,), False)     # t:t = t^2
    T = np.array([[3.0]])
    f = lambda form, **kw: des.design(A.Form(form, **kw), terms, {}, T, None,  # noqa: E731
                                      form + str(kw))[0, 0]
    assert f("value") == pytest.approx(9.0)
    assert f("slope") == pytest.approx(6.0, rel=1e-6)
    assert f("slope", direction="backward", eps=0.01) == pytest.approx(5.99, rel=1e-6)
    assert f("acceleration") == pytest.approx(2.0, rel=1e-4)
    assert f("area") == pytest.approx(3.0, rel=1e-9)                       # mean of t^2 on [0,3]
    assert f("area", time_window=1.0) == pytest.approx((27 - 8) / 3, rel=1e-9)
    assert f("Delta", time_window=1.0) == pytest.approx(5.0)               # (9 - 4) / 1
    assert f("Delta", time_window=1.0, standardise=False) == pytest.approx(5.0)
    assert f("Delta") == pytest.approx(3.0)                                # (9 - 0) / 3


# ------------------------------------------------------------------ pieces vs R
@pytest.mark.parametrize("name,spec", [("ns_int", "natural cubic spline"),
                                       ("weibull", "Weibull"),
                                       ("pwc", "piecewise constant")])
def test_baseline_bases_match_R(name, spec):
    cfg = base_hazard_config(spec)
    B = baseline_basis(np.array(REF["x"]), baseline_knots(np.array([14.4]), cfg), cfg)
    np.testing.assert_allclose(B, np.array(REF[name]), atol=1e-12)


def test_weighted_cox_matches_R():
    _, p = jmb.datasets.pbc2()
    W = np.column_stack([(p.sex == "female").astype(float), p.age])
    w = (p.sex == "female").astype(float).to_numpy() + 1
    g = fits.fit_coxph(p.years.to_numpy(float), p.status2.to_numpy(float), W, weights=w)
    np.testing.assert_allclose(g, REF["cox_weighted"], atol=1e-7)


def test_ms_setup_rebuilds_the_vignette_rows():
    d = pd.read_csv(DATA / "ms_surv.csv")
    rows = []
    for i, g in d.groupby("id"):
        r1, r2 = g[g.transition == 1].iloc[0], g[g.transition == 2].iloc[0]
        r = g[g.transition == 3].iloc[0] if r1.status == 1 else r2
        rows.append(dict(id=i, ill_t=r1.Tstop, ill_s=int(r1.status), dth_t=r.Tstop,
                         dth_s=int(r.status), X=r1.X))
    w = pd.DataFrame(rows)
    tm = [[np.nan, 1, 2], [np.nan, np.nan, 3], [np.nan] * 3]
    o = jmb.ms_setup(w, [None, "ill_t", "dth_t"], [None, "ill_s", "dth_s"], tm, covs=["X"])
    d = d.sort_values(["id", "Tstart", "transition"], kind="mergesort")
    cols = ["id", "transition", "Tstart", "Tstop", "status", "X"]
    np.testing.assert_allclose(o[cols].to_numpy(float), d[cols].to_numpy(float))


def test_centering_leaves_predictors_unchanged_with_spline_interactions():
    """NOTES.md item 7: with ns(year, 2) * sex the fixed-effect Gibbs step must not move
    any linear predictor (JMbayes2's create_HC_X() gets this wrong)."""
    visits, patients = jmb.datasets.pbc2()
    year, sex = var("year"), var("sex")
    m = dict(response=var("serBilir").log(), fixed=ns(year, 2) * sex, random=ns(year, 2),
             forms=[A.value(), A.slope(), A.acceleration()])
    md = jm_setup(visits, patients, [m], dict(time="years", event="status2",
                                              covariates=sex), "id", "year")
    sid_H = md.row_patient[md.pt_row]
    for sl in md.outcomes[0].slots:
        M = np.einsum("nr,nrp->np", sl["Z_H"], md.X_dot[sid_H])
        np.testing.assert_allclose(sl["X_H"], M, atol=1e-12)


# ------------------------------------------------------------------ fits
@pytest.fixture(scope="module")
def pbc2():
    return jmb.datasets.pbc2()


def _fit(markers, visits, patients, event=None, **kw):
    m = jmb.JointModel(event or jmb.Event("years", "status2", var("sex")), markers,
                       id="id", time="year")
    return m.fit(visits, patients, n_iter=300, n_burnin=100, n_chains=2, device="cpu",
                 verbose=False, **kw)


def _pred(f, visits):
    nd = visits[visits.id == 2].copy()
    nd["years"], nd["status2"] = 3.0, 0
    return f.predict(nd[nd.year < 3], times=[4, 6], n_samples=10)


def test_time_varying_association_and_transforms(pbc2):
    visits, patients = pbc2
    year = var("year")
    f = _fit([jmb.Marker(var("serBilir").log(), fixed=year, random=year,
                         association=[A.vexpit(A.value()), A.Dexp(A.slope()),
                                      *(A.value() * ns(year, knots=[3, 6], boundary=[0, 15]))])],
             visits, patients)
    names = f.blocks["alphas"][1]
    assert names[:2] == ["vexpit(value(log(serBilir)))",
                         "vexp(value(log(serBilir))):slope(log(serBilir))"]
    assert len(names) == 2 + 1 + 3
    assert _pred(f, visits).pred_CIF.between(0, 1).all()


def test_base_hazards_weights_and_penalties(pbc2):
    visits, patients = pbc2
    patients = patients.assign(w=(patients.sex == "female") + 1.0)
    year = var("year")
    mk = [jmb.Marker(var("serBilir").log(), fixed=year, random=year,
                     association=["value", "slope"])]
    for kw in (dict(base_hazard="Weibull"), dict(base_hazard="piecewise constant"),
               dict(basis="ns", timescale="log"),
               dict(penalty_alphas="horseshoe"), dict(penalty_alphas="ridge")):
        f = _fit(mk, visits, patients, **kw)
        assert np.isfinite(list(f.fit_statistics.values())).all(), kw
        assert ("tau_bs_gammas" in f.blocks) == (kw.get("base_hazard") != "Weibull")
        assert _pred(f, visits).pred_CIF.between(0, 1).all()
    f = _fit(mk, visits, patients, event=jmb.Event("years", "status2", var("sex"),
                                                   weights="w"))
    assert np.isfinite(list(f.fit_statistics.values())).all()


def test_multi_state_fit():
    lv = pd.read_csv(DATA / "ms_long.csv")
    ms = pd.read_csv(DATA / "ms_surv.csv")
    t = var("time")
    m = jmb.JointModel(jmb.Event("Tstop", "status", var("X"), start="Tstart",
                                 strata="transition", by_strata=True),
                       jmb.Marker("y", fixed=t, random=t), id="id", time="time")
    f = m.fit(lv, ms, n_iter=300, n_burnin=100, n_chains=2, device="cpu", verbose=False,
              base_hazard="Weibull")
    assert f.blocks["gammas"][1] == ["X [1]", "X [2]", "X [3]"]
    assert f.md.GK_k == 7 and f.md.n_strata == 3


# ------------------------------------------------------------------ bugs found by the comparisons
def test_lgamma_densities_are_exact_in_float32():
    """NOTES.md item 11: in float32 the beta-binomial log-likelihood at phi = 1e6 came out
    hundreds of units above its value at phi = 441 (rounding), and the sampler drifted."""
    from jmbayes import families as F
    rng = np.random.default_rng(0)
    eta = rng.normal(-1.0, 1.5, 1900)
    y = rng.binomial(3, 1 / (1 + np.exp(-eta))).astype(float)
    for fam, phis in (("beta_binomial", (441.0, 1e6)), ("negative_binomial", (5.0, 1e5))):
        ll = {}
        for dt in (torch.float32, torch.float64):
            Y, E, N = (torch.tensor(v, dtype=dt) for v in (y, eta, np.full(len(y), 3.0)))
            pre = F.precompute(fam, Y, N if fam == "beta_binomial" else None)
            ll[dt] = [float(F.log_dens(fam, Y, E, torch.tensor(p, dtype=dt), None, pre).sum())
                      for p in phis]
        np.testing.assert_allclose(ll[torch.float32], ll[torch.float64], atol=1e-2)


def test_lme_reaches_the_maximum_on_aids():
    """NOTES.md item 12: statsmodels' L-BFGS stopped 739 REML units below R's lme()."""
    d = pd.read_csv(DATA / "aids_long.csv")
    X = np.column_stack([np.ones(len(d)), d.obstime, d.obstime * (d.drug == "ddI")])
    Z = np.column_stack([np.ones(len(d)), d.obstime])
    f = fits.fit_lme(d.CD4.to_numpy(), X, Z, d.patient.to_numpy())
    # R: lme(CD4 ~ obstime + obstime:drug, random = ~ obstime | patient, data = aids)
    np.testing.assert_allclose(f.D, [[21.06955, -0.1237379], [-0.1237379, 0.03020929]],
                               rtol=2e-3, atol=2e-5)
    assert np.exp(f.log_sigma) == pytest.approx(1.749791, rel=1e-3)
