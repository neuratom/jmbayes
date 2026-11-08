"""Accuracy of dynamic predictions, as in JMbayes2 (R/accuracy_measures.R).

Given the measurements of each patient up to a time t_start, the model predicts the risk
of the event by a later time t_horizon. These functions check those predictions against
what happened:

* tv_roc / tv_auc: time-dependent sensitivity, specificity and AUC (discrimination)
* tv_brier:        time-dependent (or integrated) Brier score (overall accuracy)
* calibration:     observed vs predicted risk, with the ICI, E50 and E90 summaries

The definitions follow JMbayes2 exactly: the same at-risk set, the same 1e-6 shift of the
times, the same handling of patients censored between t_start and t_horizon
("model-based" weights use the model's own risk for them; "IPCW" uses Kaplan-Meier
weights of the censoring distribution), and the same default number of posterior samples
(200; 400 for the Brier score).

The *_from_risks functions take predicted risks directly (like tvMetric.default() in R),
which is how they are checked against JMbayes2 in the tests.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

THRESHOLDS = np.linspace(0.0, 1.0, 101)
_EPS = 1e-06                       # JMbayes2 shifts t_start and t_horizon by this


# ============================================================ building blocks
def _km_censoring(time, event, at):
    """Kaplan-Meier estimate of the censoring survival G(t) (censoring = event 0), at
    the times `at`, right-continuous like summary(survfit(...), times = at) in R."""
    time = np.asarray(time, float)
    cens = np.asarray(event, float) == 0
    ut = np.unique(time[cens])
    n = np.array([(time >= u).sum() for u in ut], float)
    d = np.array([((time == u) & cens).sum() for u in ut], float)
    surv = np.cumprod(1.0 - d / n) if len(ut) else np.array([])
    idx = np.searchsorted(ut, np.asarray(at, float), side="right") - 1
    return np.where(idx >= 0, surv[np.clip(idx, 0, None)] if len(ut) else 1.0, 1.0)


def _median_threshold(values):
    best = np.nanmax(values)
    return float(np.median(THRESHOLDS[values == best]))


def _check_weights(type_weights):
    t = str(type_weights).lower().replace("_", "-")
    if t in ("model-based", "model"):
        return "model-based"
    if t == "ipcw":
        return "IPCW"
    raise ValueError("weights must be 'model-based' or 'IPCW'")


# ============================================================ ROC / AUC
@dataclass
class ROC:
    """Time-dependent ROC curve (cumulative cases by t_horizon vs controls event-free)."""
    table: pd.DataFrame                     # threshold, sensitivity, specificity, counts
    youden_cutoff: float
    f1_cutoff: float
    t_start: float
    t_horizon: float
    n_at_risk: int
    weights: str

    @property
    def auc(self) -> float:
        return auc_from_roc(self.table["sensitivity"].to_numpy(),
                            1.0 - self.table["specificity"].to_numpy())

    def __repr__(self):
        d = self.table[["threshold", "sensitivity", "specificity"]].dropna()
        d = d.drop_duplicates(["sensitivity", "specificity"]).reset_index(drop=True)
        d[" "] = np.where(np.isclose(d["threshold"], self.youden_cutoff), "*", "")
        return (f"Time-dependent sensitivity and specificity at time {self.t_horizon:g}, "
                f"using information up to time {self.t_start:g} ({self.n_at_risk} patients "
                f"still at risk); censoring handled with {self.weights} weights.\n"
                f"AUC {self.auc:.4f}; '*' marks the Youden cut-off.\n{d.to_string()}")


@dataclass
class AUC:
    """Time-dependent area under the ROC curve."""
    auc: float
    t_start: float
    t_horizon: float
    n_at_risk: int
    weights: str

    def __float__(self):
        return float(self.auc)

    def __repr__(self):
        return (f"Time-dependent AUC: {self.auc:.4f} at time {self.t_horizon:g}, using "
                f"information up to time {self.t_start:g} ({self.n_at_risk} patients still "
                f"at risk; censoring handled with {self.weights} weights)")


def auc_from_roc(tp, fp) -> float:
    """Trapezoidal area under the ROC curve, as tvAUC() in JMbayes2."""
    tp, fp = np.asarray(tp, float), np.asarray(fp, float)
    return float(np.nansum(0.5 * np.diff(fp) * (tp[1:] + tp[:-1])))


def roc_from_risks(risk, time, event, t_horizon, weights="model-based",
                   censored_risk=None) -> dict:
    """ROC quantities from predicted risks, as tvROC.jm() computes them.

    risk:          predicted risk of the event by t_horizon for each patient at risk
    time, event:   each patient's observed event time and event indicator
    t_horizon:     the (already shifted) horizon, as JMbayes2 uses internally
    censored_risk: for "model-based" weights, the model's risk by t_horizon for patients
                   censored before t_horizon, given that they were event-free when
                   censored (aligned with `risk`; other entries are ignored)
    """
    weights = _check_weights(weights)
    risk = np.asarray(risk, float)
    time = np.asarray(time, float)
    event = np.asarray(event, float)
    check = (1.0 - risk)[:, None] < THRESHOLDS[None, :]
    if weights == "model-based":
        ind1 = (time < t_horizon) & (event == 1)
        ind2 = (time < t_horizon) & (event == 0)
        ind = (ind1 | ind2).astype(float)
        if ind2.any():
            ind[ind2] = np.asarray(censored_risk, float)[ind2]
        n_tp = (check * ind[:, None]).sum(0)
        n_fn = ind.sum() - n_tp
        tp = n_tp / ind.sum()
        n_fp = (check * (1.0 - ind)[:, None]).sum(0)
        n_tn = (1.0 - ind).sum() - n_fp
        fp = n_fp / (1.0 - ind).sum()
    else:
        ind1 = (time < t_horizon) & (event == 1)
        ind2 = time > t_horizon
        n_fp = (check * ind2[:, None]).sum(0)
        n_tn = ind2.sum() - n_fp
        fp = n_fp / ind2.sum()
        # JMbayes2 divides by w = 1 / G(T), i.e. weighs each case by G(T)
        w = 1.0 / _km_censoring(time, event, time[ind1])
        n_tp = (check[ind1] / w[:, None]).sum(0)
        n_fn = (1.0 / w).sum() - n_tp
        tp = n_tp / (1.0 / w).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = 2 * n_tp / (2 * n_tp + n_fn + n_fp)
    return dict(TP=tp, FP=fp, nTP=n_tp, nFN=n_fn, nFP=n_fp, nTN=n_tn,
                F1score=_median_threshold(f1), Youden=_median_threshold(tp - fp))


# ============================================================ Brier score
@dataclass
class Brier:
    """Time-dependent (or integrated) Brier score: mean squared error of the predicted
    risk. 0 is perfect; predicting 50% for everyone gives 0.25."""
    brier: float
    integrated: bool
    t_start: float
    t_horizon: float
    n_at_risk: int
    n_events: int
    n_censored: int
    weights: str

    def __float__(self):
        return float(self.brier)

    def __repr__(self):
        what = "Integrated Brier score" if self.integrated else "Brier score"
        when = (f"over [{self.t_start:g}, {self.t_horizon:g})" if self.integrated
                else f"at time {self.t_horizon:g}")
        return (f"{what}: {self.brier:.4f} {when}, for the {self.n_at_risk} patients at risk "
                f"at time {self.t_start:g} ({self.n_events} with an event and "
                f"{self.n_censored} censored before {self.t_horizon:g}; censoring handled "
                f"with {self.weights} weights)")


def brier_from_risks(risk, time, event, t_horizon, weights="model-based",
                     censored_risk=None) -> dict:
    """Brier score from predicted risks, as tvBrier.jm() computes it (arguments as in
    roc_from_risks)."""
    weights = _check_weights(weights)
    risk = np.asarray(risk, float)
    time = np.asarray(time, float)
    event = np.asarray(event, float)
    ind1 = (time < t_horizon) & (event == 1)          # event before the horizon
    ind2 = time > t_horizon                           # event-free at the horizon
    ind3 = (time < t_horizon) & (event == 0)          # censored before the horizon
    if ind1.sum() < 5:
        warnings.warn(f"there are fewer than 5 patients with an event before {t_horizon:g}",
                      RuntimeWarning)
    if weights == "model-based":
        res = np.nansum((1.0 - risk[ind1]) ** 2) + np.nansum(risk[ind2] ** 2)
        if ind3.any():
            w = np.asarray(censored_risk, float)[ind3]
            p = risk[ind3]
            res += np.nansum(w * (1.0 - p) ** 2 + (1.0 - w) * p ** 2)
        brier = res / len(ind1)
    else:
        w = np.zeros(len(time))
        w[ind1] = 1.0 / _km_censoring(time, event, time[ind1])
        if ind2.any():
            w[ind2] = 1.0 / _km_censoring(time, event, [t_horizon])[0]
        brier = float(np.mean((ind1.astype(float) - risk) ** 2 * w))
    return dict(Brier=float(brier), nr=len(time), nint=int(ind1.sum()),
                ncens=int(ind3.sum()))


# ============================================================ calibration
@dataclass
class Calibration:
    """Calibration of the predicted risk at t_horizon: the observed risk is a smooth
    function of the predicted risk, estimated with a Cox model on a natural spline of
    cloglog(predicted risk), as calibration_plot() in JMbayes2.

    ICI: mean |predicted - observed| over patients; E50, E90: median and 90th percentile.
    `curve` holds the calibration curve (observed vs predicted) for plotting; `patients`
    the predicted and observed risk of each patient."""
    ici: float
    e50: float
    e90: float
    curve: pd.DataFrame = field(repr=False)
    patients: pd.DataFrame = field(repr=False)
    t_start: float = 0.0
    t_horizon: float = 0.0
    df_ns: int = 3

    @property
    def metrics(self) -> pd.Series:
        return pd.Series({"ICI": self.ici, "E50": self.e50, "E90": self.e90})

    def __repr__(self):
        return (f"Calibration of the predicted risk at time {self.t_horizon:g} (information "
                f"up to time {self.t_start:g}, {len(self.patients)} patients):\n"
                f"  ICI {self.ici:.4f}   E50 {self.e50:.4f}   E90 {self.e90:.4f}\n"
                f"  (.curve: observed vs predicted risk for a calibration plot)")


def _ns_basis(x, knots):
    """Natural cubic spline basis without intercept (linear beyond the outer knots).
    Any basis of this space gives the same Cox model fit, so this matches R's
    nsk(x, df) given the same knots."""
    k = np.sort(np.asarray(knots, float))
    K = len(k)
    x = np.asarray(x, float)

    def d(j):
        return ((np.clip(x - k[j], 0, None) ** 3 - np.clip(x - k[-1], 0, None) ** 3)
                / (k[-1] - k[j]))

    cols = [x] + [d(j) - d(K - 2) for j in range(K - 2)]
    return np.column_stack(cols)


def _nsk_knots(x, df):
    """Knots of survival::nsk(x, df): boundary knots at the 5% and 95% quantiles, interior
    knots at quantiles of the values between them (splines::ns rules)."""
    b = np.quantile(x, [0.05, 0.95])
    inside = x[(x >= b[0]) & (x <= b[1])]
    n_int = df - 1
    probs = np.linspace(0, 1, n_int + 2)[1:-1]
    interior = np.quantile(inside, probs) if n_int > 0 else np.array([])
    return np.unique(np.concatenate([[b[0]], interior, [b[1]]]))


def calibration_from_risks(risk, time, event, t_horizon, df_ns=None) -> dict:
    """Calibration curve and ICI/E50/E90 from predicted risks, as calibration_plot() and
    calibration_metrics() in JMbayes2."""
    from jmbayes.fits import fit_coxph
    risk = np.asarray(risk, float)
    time = np.asarray(time, float)
    event = np.asarray(event, float)
    T = np.minimum(t_horizon, time)
    ev = event * (time <= t_horizon)
    if df_ns is None:
        df_ns = 3 if ev.sum() > 25 else 2
    cloglog = lambda p: np.log(-np.log(1.0 - p))           # noqa: E731
    x = cloglog(risk)
    knots = _nsk_knots(x, df_ns)
    B = _ns_basis(x, knots)
    mu, sd = B.mean(0), B.std(0)
    sd[sd == 0] = 1.0
    scale = lambda M: (M - mu) / sd                           # noqa: E731
    beta = fit_coxph(T, ev, scale(B))
    r = np.exp(scale(B) @ beta)
    H0 = 0.0                                                  # Efron baseline hazard
    for t in np.unique(T[ev == 1]):
        at_risk = T >= t
        dead = (T == t) & (ev == 1)
        nd = int(dead.sum())
        R, D = r[at_risk].sum(), r[dead].sum()
        H0 += sum(1.0 / (R - (k / nd) * D) for k in range(nd))

    def observed(p):
        return 1.0 - np.exp(-H0 * np.exp(scale(_ns_basis(cloglog(p), knots)) @ beta))

    q = np.quantile(risk, [0.01, 0.99])
    grid = np.linspace(q[0], q[1], 101)
    obs_i = observed(risk)
    diff = np.abs(risk - obs_i)
    return dict(observed=observed(grid), predicted=grid, obs_pi_u_t=obs_i,
                ICI=float(np.mean(diff)), E50=float(np.median(diff)),
                E90=float(np.quantile(diff, 0.9)), df_ns=df_ns)


# ============================================================ from a fitted model
def _window(t_start, t_horizon, dt):
    if t_horizon is None and dt is None:
        raise ValueError("give either t_horizon or dt")
    if t_horizon is not None and t_horizon <= t_start:
        raise ValueError("t_horizon must be larger than t_start")
    if t_horizon is None:
        t_horizon = t_start + dt
    return float(t_start), float(t_horizon)


class _AtRisk:
    """Patients still event-free at t_start, with their measurements up to t_start."""

    def __init__(self, fit, newdata, ts, th):
        if getattr(fit.md, "causes", None):
            raise NotImplementedError(
                "accuracy measures are not available for competing-risk models (nor in "
                "JMbayes2); use fit.predict() for each cause's cumulative incidence")
        meta = fit.md.meta
        self.id, self.tvar = meta["id_var"], meta["time_var"]
        self.T, self.ev = meta["surv"]["time"], meta["surv"]["event"]
        missing = [c for c in (self.id, self.tvar, self.T, self.ev) if c not in newdata]
        if missing:
            raise ValueError(f"newdata has no column(s) {missing}")
        nd = newdata[newdata[self.T] > ts]
        nd = nd[nd[self.tvar] <= ts]
        if not len(nd):
            raise ValueError("there are no data on patients who had an observed event time "
                             "after t_start and measurements before t_start")
        if not ((nd[self.T] < th) & (nd[self.ev] == 1)).any():
            raise ValueError("there are no events in the interval [t_start, t_horizon)")
        self.data = nd
        g = nd.groupby(self.id, sort=False)
        self.ids = np.asarray(pd.unique(nd[self.id]))
        self.time = g[self.T].last().reindex(self.ids).to_numpy(float)
        self.event = g[self.ev].last().reindex(self.ids).to_numpy(float)

    def _last_risk(self, pred, after):
        p = pred[pred[self.T] > after]
        s = p.groupby(self.id, sort=False)["pred_CIF"].last()
        return s

    def risk(self, fit, ts, th, **kw):
        """Risk by th, given event-free at ts and the measurements up to ts."""
        nd2 = self.data.copy()
        nd2[self.T] = ts
        nd2[self.ev] = 0
        pred = fit.predict(nd2, process="event", times=[th], **kw)
        s = self._last_risk(pred, ts).reindex(self.ids)
        if s.isna().any():
            raise ValueError("t_horizon is beyond the largest event time in the fitted data; "
                             "the model cannot predict that far")
        return s.to_numpy(float)

    def censored_risk(self, fit, th, **kw):
        """For patients censored before th: risk by th, given event-free when censored."""
        out = np.full(len(self.ids), np.nan)
        cens = (self.time < th) & (self.event == 0)
        if cens.any():
            sub = self.data[self.data[self.id].isin(self.ids[cens])]
            pred = fit.predict(sub, process="event", times=[th], **kw)
            s = pred.groupby(self.id, sort=False)["pred_CIF"].last()
            out[cens] = s.reindex(self.ids[cens]).to_numpy(float)
        return out


def tv_roc(fit, newdata, t_start, t_horizon=None, dt=None, weights="model-based",
           **predict_options) -> ROC:
    t0, t1 = _window(t_start, t_horizon, dt)
    ts, th = t0 + _EPS, t1 + _EPS
    weights = _check_weights(weights)
    ar = _AtRisk(fit, newdata, ts, th)
    risk = ar.risk(fit, ts, th, **predict_options)
    cr = ar.censored_risk(fit, th, **predict_options) if weights == "model-based" else None
    r = roc_from_risks(risk, ar.time, ar.event, th, weights, cr)
    table = pd.DataFrame({"threshold": THRESHOLDS, "sensitivity": r["TP"],
                          "specificity": 1.0 - r["FP"], "n_TP": r["nTP"], "n_FN": r["nFN"],
                          "n_FP": r["nFP"], "n_TN": r["nTN"]})
    return ROC(table, r["Youden"], r["F1score"], t0, t1, len(ar.ids), weights)


def tv_auc(fit, newdata, t_start, t_horizon=None, dt=None, weights="model-based",
           **predict_options) -> AUC:
    roc = tv_roc(fit, newdata, t_start, t_horizon, dt, weights, **predict_options)
    return AUC(roc.auc, roc.t_start, roc.t_horizon, roc.n_at_risk, roc.weights)


def tv_brier(fit, newdata, t_start, t_horizon=None, dt=None, integrated=False,
             weights="model-based", **predict_options) -> Brier:
    t0, t1 = _window(t_start, t_horizon, dt)
    ts, th = t0 + _EPS, t1 + _EPS
    weights = _check_weights(weights)
    predict_options.setdefault("n_samples", 400)            # as tvBrier() in JMbayes2
    ar = _AtRisk(fit, newdata, ts, th)

    def at(h):
        if not ((ar.time < h) & (ar.event == 1)).any():
            raise ValueError(f"there are no events in the interval [t_start, {h:g})")
        risk = ar.risk(fit, ts, h, **predict_options)
        cr = (ar.censored_risk(fit, h, **predict_options) if weights == "model-based"
              else None)
        return brier_from_risks(risk, ar.time, ar.event, h, weights, cr)

    last = at(th)
    value = last["Brier"]
    if integrated:                     # Simpson's rule, as JMbayes2 (Brier at t_start = 0)
        value = 2 * at(0.5 * (ts + th))["Brier"] / 3 + value / 6
    return Brier(value, integrated, t0, t1, last["nr"], last["nint"], last["ncens"], weights)


def calibration(fit, newdata, t_start, t_horizon=None, dt=None, df_ns=None,
                **predict_options) -> Calibration:
    t0, t1 = _window(t_start, t_horizon, dt)
    ts, th = t0 + _EPS, t1 + _EPS
    ar = _AtRisk(fit, newdata, ts, th)
    risk = ar.risk(fit, ts, th, **predict_options)
    c = calibration_from_risks(risk, ar.time, ar.event, th, df_ns)
    curve = pd.DataFrame({"predicted": c["predicted"], "observed": c["observed"]})
    patients = pd.DataFrame({ar.id: ar.ids, "predicted": risk, "observed": c["obs_pi_u_t"]})
    return Calibration(c["ICI"], c["E50"], c["E90"], curve, patients, t0, t1, c["df_ns"])
