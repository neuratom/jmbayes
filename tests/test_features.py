"""Phase-2 features: smoke tests, and equivalence with JMbayes2 (slow)."""
import json
import pathlib

import numpy as np
import pytest

import jmbayes2py as jmb
from jmbayes2py.splines import SplineSpec

HERE = pathlib.Path(__file__).resolve().parent


def feature_model(name, pid):
    """Same definitions as benchmark/run_feature_R.R."""
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


def test_ns_bs_bases_match_r():
    """Values printed by R: predict(ns(x, 2), xn) and predict(bs(x, 3), xn), x = pbc2$year."""
    x = jmb.datasets.pbc2()[0]["year"].to_numpy()
    xn = np.array([-0.5, 0.3, 7.1, 20.0])
    ns2 = SplineSpec.fit("ns", x, 2).basis(xn)
    assert ns2 == pytest.approx(np.array([[-0.07519493, 0.04054891], [0.04505069, -0.02425798],
                                          [0.55854229, -0.02734726], [-0.10874897, 1.73463206]]),
                                abs=1e-8)
    bs3 = SplineSpec.fit("bs", x, 3).basis(xn)
    assert bs3 == pytest.approx(np.array([[-0.1140116, 0.00390296, -0.00004454],
                                          [0.0611185, 0.00132811, 0.00000962],
                                          [0.3724789, 0.37748763, 0.12752123],
                                          [0.7426931, -2.52007831, 2.85034498]]), abs=1e-7)


@pytest.mark.parametrize("name", ["nhc", "splines", "forms"])
def test_feature_smoke(name, pbc2_data):
    pbc2, pid = pbc2_data
    fit = jmb.jm(**feature_model(name, pid), data=pbc2, time_var="year", n_iter=300,
                 n_burnin=150, device="cpu", verbose=False)
    hr = fit.hazard_ratios()
    assert np.isfinite(hr.values).all()
    if name == "nhc":
        assert fit.settings["n_iter"] == 300
        assert [fit.md.outcomes[0].x_names[c] for c in fit.md.outcomes[0].nhc] == ["year"]
    if name == "forms":
        assert list(hr.index[-3:]) == ["value(log(serBilir))", "slope(log(serBilir))",
                                       "area(prothrombin)"]


def test_uncorrelated_and_independent(pbc2_data):
    pbc2, pid = pbc2_data
    fit = jmb.jm(jmb.Survival("Surv(years, status2) ~ sex", data=pid),
                 [jmb.Longitudinal("log(serBilir) ~ year", random="~ year | id"),
                  jmb.Longitudinal("prothrombin ~ year", random="~ year || id")],
                 data=pbc2, time_var="year", which_independent="all", n_iter=300,
                 n_burnin=150, device="cpu", verbose=False)
    D = fit.random_effects_covariance()["Mean"]
    for name in ("D[3, 1]", "D[4, 1]", "D[3, 2]", "D[4, 2]", "D[4, 3]"):
        assert abs(D[name]) < 1e-12, name


@pytest.mark.slow
@pytest.mark.parametrize("name", ["nhc", "splines", "forms"])
def test_feature_equivalence_with_jmbayes2(name, pbc2_data):
    """Every parameter within 0.25 posterior SD of the JMbayes2 reference runs."""
    pbc2, pid = pbc2_data
    refs = [json.loads(p.read_text()) for p in sorted(HERE.glob(f"ref_R_feature_{name}_seed*.json"))]
    s0 = refs[0]["settings"]
    fit = jmb.jm(**feature_model(name, pid), data=pbc2, time_var="year", n_iter=s0["n_iter"],
                 n_burnin=s0["n_burnin"], n_chains=s0["n_chains"], verbose=False)
    for blk in [b["block"] for b in refs[0]["stats"]]:
        stats = [{b["block"]: b for b in r["stats"]}[blk] for r in refs]
        rm = np.mean([np.atleast_1d(s["Mean"]).astype(float) for s in stats], 0)
        rsd = np.atleast_1d(stats[0]["SD"]).astype(float)
        py = fit._table(blk)["Mean"].to_numpy()
        assert np.all(np.abs(py - rm) < 0.25 * rsd), (blk, py, rm)
