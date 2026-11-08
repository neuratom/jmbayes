"""User-facing API, fit statistics and R equivalence."""
import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import logsumexp

import jmbayes as jmb
from jmbayes import summary as sm
from jmbayes.sampler import JointModelSampler

from conftest import README_MODELS, ref, setup_args


def models():
    return [jmb.Longitudinal(**m) for m in README_MODELS]


@pytest.fixture(scope="module")
def short_fit(pbc2_data):
    pbc2, pid = pbc2_data
    return jmb.jm(jmb.Survival("Surv(years, status2) ~ sex", data=pid), models(),
                  data=pbc2, time_var="year", n_iter=400, n_burnin=200, device="cpu",
                  verbose=False)


def test_api_tables(short_fit, tmp_path):
    hr = short_fit.hazard_ratios()
    assert list(hr.index) == ["sexfemale", "value(log(serBilir))", "value(prothrombin)",
                              "value(ascites)"]
    assert (hr["2.5%"] <= hr["HR"]).all() and (hr["HR"] <= hr["97.5%"]).all()
    coef = short_fit.coefficients("log(serBilir)")
    assert list(coef.index) == ["(Intercept)", "year", "sexfemale", "year:sexfemale", "sigma"]
    assert short_fit.random_effects().shape == (312, 6)
    assert short_fit.random_effects_covariance().shape[0] == 21
    path = tmp_path / "fit.jmb"
    short_fit.save(str(path))
    again = jmb.load(str(path))
    pd.testing.assert_frame_equal(again.hazard_ratios(), hr)


def test_helpful_errors(pbc2_data):
    pbc2, pid = pbc2_data
    with pytest.raises(NotImplementedError):
        jmb.Longitudinal("serBilir ~ year", random="~ year | id", family="gamma")
    with pytest.raises(NotImplementedError):
        jmb.api.parse_functional_forms("~ velocity(y)", ["y"])
    assert jmb.Longitudinal("serBilir ~ year", random="~ year || id").uncorrelated
    with pytest.raises(ValueError):
        jmb.Survival("Surv(years, status) ~ sex", data=pid)    # status is not 0/1
    with pytest.raises(ValueError):
        jmb.Longitudinal("serBilir ~ year", random="year")


def test_streamed_fit_statistics_equal_full_table(pbc2_data):
    """DIC/WAIC/LPML from running sums == R's fit_stats() on the stored table."""
    pbc2, pid = pbc2_data
    md = jmb.api.jm_setup(pbc2, pid, *setup_args(), "id", "year")
    smp = JointModelSampler(md, 2, device="cpu")
    table = []
    orig = smp.accumulate

    def acc(s, keep):
        if bool(keep):
            table.append((s["ll_long"] + s["ll_surv"] + s["ll_re"]).clone())
        orig(s, keep)
    smp.accumulate = acc
    out = smp.run(n_iter=220, n_burnin=150, progress=False)
    lL = torch.cat([torch.stack([t[c] for t in table]) for c in range(2)]).numpy()
    blocks = sm.collect(md, out)
    ll_mean = sm.loglik_at_means(smp, md, blocks, out)
    D_bar = -2 * np.mean(lL.sum(1))
    pD = D_bar + 2 * ll_mean.sum()
    CC = math.log(lL.shape[0])
    expect = dict(DIC=pD + D_bar, pD=pD, LPML=np.sum(-logsumexp(-lL, 0) + CC),
                  WAIC=-2 * np.sum(logsumexp(lL, 0) - CC) + 2 * np.sum(lL.var(0, ddof=1)))
    got = sm.fit_stats(out["fitstats_acc"], out["n_keep"], ll_mean)
    for k, v in expect.items():
        assert got[k] == pytest.approx(v, rel=1e-10)


@pytest.mark.slow
def test_equivalence_with_jmbayes2(pbc2_data):
    """Refit pbc2 and compare with two JMbayes2 runs (12 chains x 30,000 iterations).

    Regression coefficients, association parameters and residual SDs must lie within
    0.35 posterior SD of the R estimate (two R runs differ by up to 0.14 SD; this run is
    shorter, so its own Monte Carlo error is larger).
    """
    pbc2, pid = pbc2_data
    fit = jmb.jm(jmb.Survival("Surv(years, status2) ~ sex", data=pid), models(),
                 data=pbc2, time_var="year", n_iter=12000, n_burnin=2000, n_thin=1,
                 n_chains=6, verbose=False)
    r_runs = [ref("ref_R_eq_seed123.json"), ref("ref_R_eq_seed456.json")]
    checked = 0
    for blk in ("gammas", "alphas", "betas1", "betas2", "betas3", "sigmas"):
        stats = [{b["block"]: b for b in r["stats"]}[blk] for r in r_runs]
        r_mean = np.mean([np.atleast_1d(s["Mean"]) for s in stats], 0)
        r_sd = np.atleast_1d(stats[0]["SD"])
        py = fit._table(blk)
        for i, (name, row) in enumerate(py.iterrows()):
            assert abs(row["Mean"] - r_mean[i]) < 0.35 * r_sd[i], (blk, name)
            checked += 1
    assert checked == 17         # 1 + 3 + 4 + 4 + 3 + 2
    hr = fit.hazard_ratios()
    assert hr.loc["value(log(serBilir))", "HR"] == pytest.approx(1.68, abs=0.08)
    assert hr.loc["value(ascites)", "HR"] == pytest.approx(1.77, abs=0.08)
