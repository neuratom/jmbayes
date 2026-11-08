"""Prediction-accuracy measures (tvROC/tvAUC, tvBrier, calibration) against JMbayes2.

The reference (ref_R_accuracy.json, from benchmark/accuracy_R.R) holds JMbayes2's results
for two windows on pbc2 together with every predicted risk JMbayes2 computed inside
those functions. Feeding the same risks to the Python functions must reproduce
JMbayes2's numbers.
"""
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from jmbayes import accuracy

REF = json.loads((pathlib.Path(__file__).parent / "ref_R_accuracy.json").read_text())


@pytest.fixture(scope="module")
def short_fit(pbc2_data):
    import jmbayes as jmb
    from jmbayes import var
    pbc2, pid = pbc2_data
    year, sex = var("year"), var("sex")
    return jmb.JointModel(
        jmb.Event("years", "status2", sex),
        [jmb.Marker(var("serBilir").log(), fixed=year * sex, random=year),
         jmb.Marker("ascites", fixed=year + sex, random=year, family="binomial")],
        id="id", time="year").fit(pbc2, pid, n_iter=400, n_burnin=200, device="cpu",
                                  verbose=False)


def _subjects(s):
    subj = pd.DataFrame(s["subjects"])
    ids = subj["id"].astype(str).to_numpy()
    return ids, subj["Time"].to_numpy(float), subj["event"].to_numpy(float)


def _risk(call, ids, after=None):
    """Risk per patient from one recorded predict() call: the last prediction of each
    patient (after `after`, to skip the row at the start time)."""
    df = pd.DataFrame({"id": np.atleast_1d(call["id"]).astype(str),
                       "times": np.atleast_1d(call["times"]),
                       "pred": np.atleast_1d(call["pred"])})
    if after is not None:
        df = df[df["times"] > after]
    return df.groupby("id", sort=False)["pred"].last().reindex(ids).to_numpy(float)


def _main_and_censored(s, calls, horizon):
    ids, time, event = _subjects(s)
    risk = _risk(calls[0], ids, s["Tstart"])
    cens = _risk(calls[1], ids) if len(calls) > 1 else np.full(len(ids), np.nan)
    return risk, time, event, cens


@pytest.mark.parametrize("window", list(REF))
def test_roc_and_auc_match_jmbayes2(window):
    s = REF[window]
    for key, w in (("roc_mb", "model-based"), ("roc_ipcw", "IPCW")):
        if key not in s:
            continue
        ref = s[key]
        risk, time, event, cens = _main_and_censored(s, ref["calls"], s["Thoriz"])
        r = accuracy.roc_from_risks(risk, time, event, s["Thoriz"], w, cens)
        np.testing.assert_allclose(r["TP"], ref["TP"], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(r["FP"], ref["FP"], rtol=1e-10, atol=1e-12)
        assert r["Youden"] == pytest.approx(ref["Youden"])
        assert r["F1score"] == pytest.approx(ref["F1score"])
        assert accuracy.auc_from_roc(r["TP"], r["FP"]) == pytest.approx(ref["auc"], abs=1e-12)
        if w == "model-based":
            for k in ("nTP", "nFN", "nFP", "nTN"):
                np.testing.assert_allclose(r[k], ref[k], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("window", list(REF))
def test_brier_matches_jmbayes2(window):
    s = REF[window]
    ref = s["brier_mb"]
    risk, time, event, cens = _main_and_censored(s, ref["calls"], s["Thoriz"])
    with _no_check():
        mb = accuracy.brier_from_risks(risk, time, event, s["Thoriz"], "model-based", cens)
    assert mb["Brier"] == pytest.approx(ref["Brier"], abs=1e-12)
    assert [mb["nr"], mb["nint"], mb["ncens"]] == [ref["nr"], ref["nint"], ref["ncens"]]
    if "brier_ipcw" in s:
        risk, time, event, _ = _main_and_censored(s, s["brier_ipcw"]["calls"], s["Thoriz"])
        with _no_check():
            ipcw = accuracy.brier_from_risks(risk, time, event, s["Thoriz"], "IPCW")
        assert ipcw["Brier"] == pytest.approx(s["brier_ipcw"]["Brier"], abs=1e-12)
    if "brier_int" in s:          # Simpson's rule: Brier at the midpoint and the horizon
        calls = s["brier_int"]["calls"]
        rm, time, event, cm = _main_and_censored(s, calls[:2], s["Tmid"])
        rh, _, _, ch = _main_and_censored(s, calls[2:], s["Thoriz"])
        with _no_check():
            mid = accuracy.brier_from_risks(rm, time, event, s["Tmid"], "model-based", cm)
            hor = accuracy.brier_from_risks(rh, time, event, s["Thoriz"], "model-based", ch)
        assert 2 * mid["Brier"] / 3 + hor["Brier"] / 6 == pytest.approx(
            s["brier_int"]["Brier"], abs=1e-12)


@pytest.mark.parametrize("window", list(REF))
def test_calibration_matches_jmbayes2(window):
    s = REF[window]
    ref = s["calibration"]
    ids, time, event = _subjects(s)
    risk = pd.Series(ref["pi_u_t"], index=[str(i) for i in ref["ids"]]).reindex(ids)
    c = accuracy.calibration_from_risks(risk.to_numpy(float), time, event, s["Thoriz"])
    np.testing.assert_allclose(c["predicted"], ref["predicted"], rtol=1e-10)
    np.testing.assert_allclose(c["observed"], ref["observed"], atol=1e-6)
    obs = pd.Series(ref["obs_pi_u_t"], index=[str(i) for i in ref["ids"]])
    np.testing.assert_allclose(c["obs_pi_u_t"], obs.reindex(ids).to_numpy(), atol=1e-6)
    for k in ("ICI", "E50", "E90"):
        assert c[k] == pytest.approx(ref["metrics"][k], abs=1e-6)


def test_accuracy_methods_on_a_fit(short_fit, pbc2_data):
    """End to end on a short Python fit: the outputs are well-formed and sensible."""
    pbc2, _ = pbc2_data
    opts = dict(n_samples=100)
    auc = short_fit.tv_auc(pbc2, t_start=5, dt=2, **opts)
    assert 0.0 <= auc.auc <= 1.0 and auc.n_at_risk > 0
    roc = short_fit.tv_roc(pbc2, t_start=5, t_horizon=7, weights="IPCW", **opts)
    assert len(roc.table) == 101 and roc.table["sensitivity"].between(0, 1).all()
    br = short_fit.tv_brier(pbc2, t_start=5, dt=2, **opts)
    ib = short_fit.tv_brier(pbc2, t_start=5, dt=2, integrated=True, **opts)
    assert 0.0 < br.brier < 0.25 and 0.0 < ib.brier < br.brier
    cal = short_fit.calibration(pbc2, t_start=5, dt=2, **opts)
    assert 0.0 <= cal.ici <= cal.e90 <= 1.0
    assert len(cal.curve) == 101 and cal.curve["observed"].between(0, 1).all()
    with pytest.raises(ValueError):
        short_fit.tv_auc(pbc2, t_start=5)                 # neither t_horizon nor dt


class _no_check:
    """Silence the 'fewer than 5 events' warning in the reference comparisons."""

    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore")

    def __exit__(self, *a):
        return self._w.__exit__(*a)
