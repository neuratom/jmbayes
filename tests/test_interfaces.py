"""The Python-object interface and the R-style string interface fit the same model."""
import numpy as np

import jmbayes2py as jmb
from jmbayes2py import var


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
        jmb.Marker("serBilir", family="poisson")
    with pytest.raises(NotImplementedError):
        jmb.Marker("serBilir", association="velocity")
    with pytest.raises(TypeError):
        jmb.Marker(var("serBilir"))           # a term, not a response
    with pytest.raises(KeyError):
        pbc2, pid = pbc2_data
        jmb.JointModel(jmb.Event("years", "status2", "sex"),
                       jmb.Marker("nonexistent", fixed="year"), id="id",
                       time="year").fit(pbc2, pid, n_iter=200, n_burnin=100,
                                          verbose=False)
