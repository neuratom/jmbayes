"""Competing risks: cause-specific hazards, as crisk_setup() + strata(CR) in JMbayes2."""
import numpy as np
import pandas as pd
import pytest
import torch

import jmbayes as jmb
from jmbayes import prediction as P
from jmbayes import var


def _model():
    year, sex = var("year"), var("sex")
    return jmb.JointModel(
        jmb.Event("years", "status", sex, censored="alive"),
        [jmb.Marker(var("serBilir").log(), fixed=year * sex, random=year)],
        id="id", time="year")


@pytest.fixture(scope="module")
def cr_fit(pbc2_data):
    pbc2, pid = pbc2_data
    pid = pid.copy()
    pid["status"] = pd.Categorical(pid["status"], ["alive", "transplanted", "dead"])
    return _model().fit(pbc2, pid, n_iter=400, n_burnin=100, n_chains=2, device="cpu",
                        verbose=False)


def test_cause_specific_parameters(cr_fit):
    assert cr_fit.md.causes == ["transplanted", "dead"]
    assert list(cr_fit.hazard_ratios().index) == [
        "sexfemale [transplanted]", "sexfemale [dead]",
        "value(log(serBilir)) [transplanted]", "value(log(serBilir)) [dead]"]
    assert cr_fit.blocks["bs_gammas"][0].shape[2] == 2 * 11       # a baseline per cause
    assert cr_fit.blocks["tau_bs_gammas"][1] == ["tau_bs_gammas_1", "tau_bs_gammas_2"]
    assert np.isfinite(list(cr_fit.fit_statistics.values())).all()


def test_cause_risks_add_up_to_overall_risk(cr_fit, pbc2_data):
    pbc2, _ = pbc2_data
    nd = pbc2[pbc2.id.isin([25, 93]) & (pbc2.year < 5)].copy()
    nd["years"], nd["status"] = 5.0, "alive"
    out = cr_fit.predict(nd, times=[6, 8, 10], n_samples=50)
    assert set(out["cause"]) == {"transplanted", "dead"}
    assert out.groupby(["id", "cause"])["pred_CIF"].apply(
        lambda x: np.all(np.diff(x.to_numpy()) >= 0)).all()
    draws, _ = P._posterior_draws(cr_fit, 50)
    grids = P._time_grid(cr_fit.md, np.array([5.0, 5.0]), [6, 8, 10])
    eng = P._Engine(cr_fit, P._NewData(cr_fit.md, nd, grids), draws, "cpu")
    b = eng.sample_b(20, torch.Generator().manual_seed(1))
    total = sum(eng.cif_cr(b)).numpy()
    H = eng.cumhaz_cr(b)
    overall = (1 - torch.exp(-(H - H[..., :1]))).numpy()
    assert np.abs(total - overall).max() < 0.01        # 7-point rule vs exact total


def test_competing_risk_errors(pbc2_data):
    pbc2, pid = pbc2_data
    pid2 = pid.assign(status=np.where(pid["status"] == "alive", "alive", "dead"))
    with pytest.raises(jmb.DataError, match="at least two causes"):
        _model().fit(pbc2, pid2, n_iter=200, n_burnin=100, verbose=False)
    with pytest.raises(jmb.DataError, match="coded 0/1"):
        jmb.JointModel(jmb.Event("years", "status", "sex"), jmb.Marker("albumin"),
                       id="id", time="year").fit(pbc2, pid, n_iter=200, n_burnin=100,
                                                 verbose=False)


def test_accuracy_not_for_competing_risks(cr_fit, pbc2_data):
    pbc2, _ = pbc2_data
    with pytest.raises(NotImplementedError, match="competing-risk"):
        cr_fit.tv_auc(pbc2, t_start=5, dt=2)
