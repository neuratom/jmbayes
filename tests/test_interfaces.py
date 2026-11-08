"""The Python-object interface and the R-style string interface fit the same model."""
import numpy as np
import pandas as pd

import jmbayes as jmb
from jmbayes import var


def test_python_and_string_interfaces_give_identical_fits(pbc2_data):
    pbc2, pid = pbc2_data
    year, sex = var("year"), var("sex")
    opts = dict(n_iter=200, n_burnin=100, n_chains=2, device="cpu", verbose=False)

    py = jmb.JointModel(
        event=jmb.Event(time="years", event="status2", covariates=["sex", "drug"]),
        markers=[jmb.Marker(var("serBilir").log(), fixed=year.ns(2) * sex, random=year,
                            association=["value", "slope"]),
                 jmb.Marker("albumin", fixed=year + sex, uncorrelated=True)],
        id="id", time="year").fit(pbc2, pid, **opts)

    rs = jmb.jm(jmb.Survival("Surv(years, status2) ~ sex + drug", data=pid),
                [jmb.Longitudinal("log(serBilir) ~ ns(year, 2) * sex", random="~ year | id"),
                 jmb.Longitudinal("albumin ~ year + sex", random="~ 1 || id")],
                data=pbc2, time_var="year",
                functional_forms="~ value(log(serBilir)) + slope(log(serBilir))", **opts)

    assert py.md.beta_names == rs.md.beta_names
    assert py.md.assoc_names == rs.md.assoc_names
    for k in py.blocks:
        assert np.array_equal(py.blocks[k][0], rs.blocks[k][0]), k
    assert list(py.hazard_ratios().index) == ["sexfemale", "drugD-penicil",
                                              "value(log(serBilir))", "slope(log(serBilir))",
                                              "value(albumin)"]


def test_python_interface_errors(pbc2_data):
    import pytest
    with pytest.raises(NotImplementedError):
        jmb.Marker("serBilir", family="gamma")
    with pytest.raises(NotImplementedError):
        jmb.Marker("serBilir", association="velocity")
    with pytest.raises(TypeError):
        jmb.Marker(var("serBilir"))           # a term, not a response
    with pytest.raises(ValueError, match="16-bit"):
        pbc2, pid = pbc2_data
        jmb.JointModel(jmb.Event("years", "status2", "sex"), jmb.Marker("albumin"),
                       id="id", time="year").fit(pbc2, pid, n_iter=200, n_burnin=100,
                                                 dtype="bfloat16", verbose=False)
    with pytest.raises(jmb.DataError):
        pbc2, pid = pbc2_data
        jmb.JointModel(jmb.Event("years", "status2", "sex"),
                       jmb.Marker("nonexistent", fixed="year"), id="id",
                       time="year").fit(pbc2, pid, n_iter=200, n_burnin=100,
                                          verbose=False)


def test_missing_and_invalid_data_give_clear_errors(pbc2_data):
    """Nothing is imputed or dropped silently: bad input stops with a clear message."""
    import pytest
    pbc2, pid = pbc2_data
    model = jmb.JointModel(jmb.Event("years", "status2", ["sex", "age"]),
                           jmb.Marker(var("serBilir").log(), fixed="year + sex", random="year"),
                           id="id", time="year")
    fit = lambda v, p: model.fit(v, p, n_iter=200, n_burnin=100, verbose=False)  # noqa

    p = pid.copy(); p.loc[p.index[:3], "age"] = np.nan
    with pytest.raises(jmb.DataError, match=r"patients data has missing values in: age \(3"):
        fit(pbc2, p)
    p = pid.copy(); p.loc[p.index[0], "years"] = np.nan
    with pytest.raises(jmb.DataError, match="years"):
        fit(pbc2, p)
    v = pbc2.copy(); v.loc[v.index[:2], "sex"] = np.nan
    with pytest.raises(jmb.DataError, match=r"visits data has missing values in: sex \(2"):
        fit(v, pid)
    v = pbc2.copy(); v.loc[v.index[0], "serBilir"] = 0.0
    with pytest.raises(jmb.DataError, match="positive"):
        fit(v, pid)
    with pytest.raises(jmb.DataError, match="without any visit"):
        fit(pbc2[pbc2.id != pbc2.id.iloc[0]], pid)
    with pytest.raises(jmb.DataError, match="one row per patient"):
        fit(pbc2, pd.concat([pid, pid.iloc[:1]]))


def test_missing_marker_value_means_not_measured(pbc2_data):
    """An empty marker cell is allowed: that visit is not an observation of the marker."""
    pbc2, pid = pbc2_data
    assert pbc2["ascites"].isna().any()
    fit = jmb.JointModel(jmb.Event("years", "status2", "sex"),
                         jmb.Marker("ascites", fixed="year + sex", family="binomial"),
                         id="id", time="year").fit(pbc2, pid, n_iter=150, n_burnin=110,
                                                   device="cpu", verbose=False)
    assert len(fit.md.outcomes[0].y) == pbc2["ascites"].notna().sum()


def test_saved_loglik_table_reproduces_fit_statistics(pbc2_data):
    """save_loglik=True keeps the per-patient table; WAIC and LPML computed from it equal
    the ones jmbayes computes from running totals. By default nothing is kept."""
    from scipy.special import logsumexp
    pbc2, pid = pbc2_data
    model = jmb.JointModel(jmb.Event("years", "status2", "sex"),
                           jmb.Marker(var("serBilir").log(), fixed=var("year"),
                                      random=var("year")),
                           id="id", time="year")
    opts = dict(n_iter=300, n_burnin=100, n_chains=2, device="cpu", verbose=False)
    assert model.fit(pbc2, pid, **opts).loglik is None
    for dtype in ("float64", "float32"):
        fit = model.fit(pbc2, pid, save_loglik=True, dtype=dtype, **opts)
        ll = fit.loglik
        assert ll.shape == (2, 200, len(pid)) and np.isfinite(ll).all()
        assert len(fit.patient_ids) == len(pid)
        x = ll.reshape(-1, ll.shape[2]).astype(np.float64)       # draws x patients
        N = x.shape[0]
        lpml = np.sum(np.log(N) - logsumexp(-x, axis=0))
        waic = -2 * np.sum(logsumexp(x, axis=0) - np.log(N)) + 2 * np.sum(x.var(0, ddof=1))
        tol = 1e-8                     # totals are kept in float64 in both precisions
        assert np.isclose(lpml, fit.fit_statistics["LPML"], rtol=tol), dtype
        assert np.isclose(waic, fit.fit_statistics["WAIC"], rtol=tol), dtype
