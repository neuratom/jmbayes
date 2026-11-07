"""Dynamic predictions: behaviour, and agreement with JMbayes2's predict()."""
import copy

import numpy as np
import pytest

import jmbayes as jmb
from jmbayes.prediction import predict

from conftest import README_MODELS, ref


@pytest.fixture(scope="module")
def short_fit(pbc2_data):
    pbc2, pid = pbc2_data
    return jmb.jm(jmb.Survival("Surv(years, status2) ~ sex", data=pid),
                  [jmb.Longitudinal(**m) for m in README_MODELS], data=pbc2,
                  time_var="year", n_iter=400, n_burnin=200, device="cpu", verbose=False)


def new_patients(pbc2, ids=(25, 93), t0=5.0):
    nd = pbc2[pbc2["id"].isin(ids) & (pbc2["year"] < t0)].copy()
    nd["status2"] = 0.0
    nd["years"] = t0
    return nd


def test_event_predictions_are_well_formed(short_fit, pbc2_data):
    pbc2, _ = pbc2_data
    ev = short_fit.predict(new_patients(pbc2), times=np.linspace(5, 12, 8), n_samples=50,
                           n_mcmc=20, device="cpu")
    assert len(ev) == 16
    for _, g in ev.groupby("id"):
        assert g["pred_CIF"].iloc[0] == 0.0                     # no risk at the last time
        assert np.all(np.diff(g["pred_CIF"]) >= -1e-12)         # cumulative risk grows
        assert ((g["low_CIF"] <= g["pred_CIF"] + 1e-12)
                & (g["pred_CIF"] <= g["upp_CIF"] + 1e-12)).all()


def test_longitudinal_predictions_are_well_formed(short_fit, pbc2_data):
    pbc2, _ = pbc2_data
    nd = new_patients(pbc2)
    lg = short_fit.predict(nd, process="longitudinal", n_samples=50, n_mcmc=20, device="cpu")
    assert len(lg) == len(nd)
    asc = lg["pred_ascites"]
    assert ((asc > 0) & (asc < 1)).all()                        # probabilities


def test_event_column_must_be_zero(short_fit, pbc2_data):
    pbc2, _ = pbc2_data
    nd = new_patients(pbc2)
    nd["status2"] = 1.0
    with pytest.raises(ValueError):
        short_fit.predict(nd)


@pytest.mark.slow
def test_prediction_engine_matches_jmbayes2(short_fit, pbc2_data):
    """Run the Python engine on JMbayes2's own posterior draws (README model, all 6000
    draws). With R's way of sampling the random effects (r_sampling=True) the risks must
    match R's predict(); see NOTES.md for why the default differs."""
    pbc2, _ = pbc2_data
    f = copy.copy(short_fit)
    f.blocks = dict(short_fit.blocks)
    for k, v in ref("ref_R_draws_readme.json").items():
        a = np.asarray(v, float)
        f.blocks[k] = ((a[:, None] if a.ndim == 1 else a)[None], short_fit.blocks[k][1])
    r = np.array([x["pred_CIF"] for x in ref("ref_R_predict_all.json")["event"]])
    py = predict(f, new_patients(pbc2), times=np.linspace(5, 12, 8), n_samples=6000,
                 draws="first", r_sampling=True)["pred_CIF"].to_numpy()
    assert np.max(np.abs(py - r)) < 0.01
