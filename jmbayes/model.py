"""Model setup: the Python counterpart of jm() in R/jm.R and jm_fit() in R/jm_fit.R.

Scope:
  * longitudinal families: gaussian, binomial (logit; optionally with trials), poisson,
    negative binomial, Student's t (fixed df), beta, gamma, unit Lindley, censored normal,
    beta-binomial (see families.py)
  * functional forms: value(), slope(), area(), no transformations
  * survival rows as jm() sees them: right censoring, delayed entry and counting-process
    rows (Surv(start, stop, event): time-varying covariates, recurrent events with a
    frailty, gap or calendar timescale), left and interval censoring (Surv(time, time2,
    event, type = "interval"), survreg() starting values); strata (a baseline hazard per
    stratum, optionally with stratum-specific effects), competing risks (one stratum per
    cause, times the user's strata, as crisk_setup()); no weights or multi-state models
Anything outside this scope raises NotImplementedError rather than silently
fitting a different model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import re

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from jmbayes import assoc as asc
from jmbayes import design as dz
from jmbayes.families import FAMILIES, has_sigma
from jmbayes.fits import MixedFit, fit_coxph, fit_glmm, fit_lme, fit_weibull_interval

# Gauss-Kronrod 15-point rule, same order as gaussKronrod() in help_functions.R
GK_SK = np.array([-0.949107912342758, -0.741531185599394, -0.405845151377397,
                  0, 0.405845151377397, 0.741531185599394, 0.949107912342758,
                  -0.991455371120813, -0.864864423359769, -0.586087235467691,
                  -0.207784955007898, 0.207784955007898, 0.586087235467691,
                  0.864864423359769, 0.991455371120813])
GK_WK = np.array([0.0630920926299786, 0.140653259715526, 0.190350578064785,
                  0.209482141084728, 0.190350578064785, 0.140653259715526,
                  0.0630920926299786, 0.0229353220105292, 0.10479001032225,
                  0.169004726639268, 0.204432940075299, 0.204432940075299,
                  0.169004726639268, 0.10479001032225, 0.0229353220105292])
# gaussKronrod(7): the first 7 nodes with Gauss weights (JMbayes2's default with delayed
# entry, i.e. Surv(start, stop, event))
GK7_WK = np.array([0.12948496616887, 0.279705391489277, 0.381830050505119,
                   0.417959183673469, 0.381830050505119, 0.279705391489277,
                   0.12948496616887])


def gauss_kronrod(k):
    if k == 15:
        return GK_SK, GK_WK
    if k == 7:
        return GK_SK[:7], GK7_WK
    raise NotImplementedError("GK_k must be 7 or 15")


@dataclass
class Outcome:
    name: str
    family: str
    y: np.ndarray
    X: np.ndarray
    Z: np.ndarray
    sid: np.ndarray                 # subject index (0..n-1) of each observation
    x_names: list
    z_names: list
    # association terms, in the order of the functional-form formula; each is a dict
    # form ('value' | 'slope' | 'area') and its designs X_H, Z_H (quadrature points)
    # and X_h, Z_h (event/censoring times)
    assoc: list
    re_cols: np.ndarray             # columns of the full random-effects matrix
    fe_cols: np.ndarray             # positions in the stacked beta vector
    has_sigma: bool
    # fixed effects outside the hierarchical centering (updated by Metropolis-Hastings);
    # these columns are mean-centred, Xbar holds the means (zero elsewhere)
    response: dz.Response | None = None
    fixed: dz.Terms | None = None   # fixed-effects terms
    random: dz.Terms | None = None  # random-effects terms
    levels: dict = field(default_factory=dict)   # factor levels and spline knots
    nhc: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    Xbar: np.ndarray = field(default_factory=lambda: np.zeros(0))
    df: float = 0.0                 # Student's t degrees of freedom (fixed)
    aux: np.ndarray | None = None   # trials (binomial families) or censoring codes
    aux_col: object = None          # the column (or constant) aux came from
    # functional forms (0.8): designs per form ("slots"); each association column in
    # `assoc` is u * prod(transform(eta_slot)) (factors = [(slot, transform), ...])
    slots: list = field(default_factory=list)
    by_levels: dict = field(default_factory=dict)


@dataclass
class ModelData:
    n: int
    q: int
    outcomes: list
    # survival
    time: np.ndarray
    event: np.ndarray
    log_Pwk: np.ndarray
    W0_H: np.ndarray
    W0_h: np.ndarray
    W_H: np.ndarray
    W_h: np.ndarray
    W_bar: np.ndarray
    W_sds: np.ndarray
    w_names: list
    Wlong_bar: np.ndarray
    Wlong_sds: np.ndarray
    assoc_names: list               # e.g. ['value(log(serBilir))', 'slope(log(serBilir))']
    assoc_outcome: np.ndarray       # outcome index of each association column
    knots: np.ndarray
    GK_k: int
    # hierarchical centering of fixed effects
    ind_FE_HC: np.ndarray
    X_dot: np.ndarray               # (n, q, p_HC)
    patterns: list                  # list of (subject idx, RE idx, FE-HC idx)
    upper_part: np.ndarray          # (row, col) pairs of the free Cholesky entries
    ind_zero_D: np.ndarray          # (row, col) pairs of correlations fixed at zero
    priors: dict
    inits: dict
    beta_names: list
    sid_names: list = field(default_factory=list)
    # how the data were set up, to rebuild designs for new patients
    meta: dict = field(default_factory=dict)
    # competing risks: survival rows are (patient, cause) pairs, as after crisk_setup();
    # idT maps each row to its patient and strata to its cause. None: one row per patient
    idT: np.ndarray | None = None
    strata: np.ndarray | None = None
    causes: list | None = None
    # strata of the baseline hazard (the causes for competing risks), with each
    # stratum's spline knots; entry times per survival row (zeros without delayed entry)
    strata_levels: list | None = None
    knots_strata: list | None = None
    trunc: np.ndarray | None = None
    # survival rows (added in 0.7): delta per row (0 right censored, 1 event, 2 left
    # censored, 3 interval censored); quadrature points = R * GK_k points of the main
    # integrals followed by GK_k points per interval-censored row on [Time_left,
    # Time_right]; pt_row maps points to rows
    delta: np.ndarray | None = None
    pt_row: np.ndarray | None = None
    n_main_H: int | None = None
    interval_rows: np.ndarray | None = None
    left_rows: np.ndarray | None = None
    # groups with their own covariate effects and associations (causes, or strata with
    # by_strata=True)
    group: np.ndarray | None = None
    group_levels: list | None = None
    user_strata_levels: list | None = None
    recurrent: str | None = None    # "gap" / "calendar": stratum 0 = recurrent process
    surv_kind: str = "right"        # "right", "counting" or "interval"
    weights: np.ndarray | None = None   # case weights per survival row (coxph(weights=))

    @property
    def n_strata(self) -> int:
        return 1 if not self.strata_levels else len(self.strata_levels)

    @property
    def row_patient(self) -> np.ndarray:
        return np.arange(self.n) if self.idT is None else self.idT


def outcome_slots(o):
    """(slots, columns) of an outcome. Fits saved before 0.8 stored one design per
    column; they become one slot per column with the identity transform."""
    if getattr(o, "slots", None):
        return o.slots, o.assoc
    slots, cols = [], []
    for a in o.assoc:
        slots.append(dict(form=asc.Form(a["form"]), X_H=a["X_H"], Z_H=a["Z_H"],
                          X_h=a["X_h"], Z_h=a["Z_h"]))
        c = {k: v for k, v in a.items() if k not in ("X_H", "Z_H", "X_h", "Z_h")}
        c.update(factors=[(len(slots) - 1, "identity")], name=f"{a['form']}({o.name})")
        cols.append(c)
    return slots, cols


def _bspline_basis(x, knots, degree):
    nb = len(knots) - degree - 1
    out = np.zeros((len(x), nb))
    for j in range(nb):
        c = np.zeros(nb)
        c[j] = 1.0
        out[:, j] = BSpline(knots, c, degree, extrapolate=False)(x)
    out[np.isnan(out)] = 0.0
    # splineDesign includes the right boundary in the last basis function
    at_end = np.isclose(x, knots[-1])
    if at_end.any():
        out[at_end] = 0.0
        out[at_end, -1] = 1.0
    return out


def _knots(time_right, segments, degree, basis="bs"):
    # knots() in help_functions.R
    xl = np.sqrt(np.finfo(float).eps)
    xr = time_right.max() + 0.001
    kn = np.linspace(xl, xr, segments + 1)
    if basis != "bs":
        return kn
    return np.concatenate([np.repeat(xl, degree), kn, np.repeat(xr, degree)])


BASE_HAZARD = dict(basis="bs", degree=2, segments=9, timescale="identity")


def base_hazard_config(base_hazard=None, basis="bs", degree=2, segments=9,
                       timescale="identity"):
    """The log baseline hazard's basis, as jm(base_hazard = , control = list(basis,
    Bsplines_degree, base_hazard_segments, timescale_base_hazard)): get_hazard()
    reads "piecewise constant", "piecewise linear", "Weibull", "log time" and
    "natural cubic spline" / "ns" from base_hazard."""
    cfg = dict(basis=basis, degree=int(degree), segments=int(segments), timescale=timescale)
    if base_hazard:
        b = str(base_hazard)
        hit = False
        if re.search(r"[pP]iecewise.*[cC]onst", b):
            cfg["degree"], hit = 0, True
        if re.search(r"[pP]iecewise.*[lL]inear", b):
            cfg["degree"], hit = 1, True
        if re.search(r"[wW]eibull", b):
            cfg.update(timescale="log", basis="ns", segments=1, weibull=True)
            hit = True
        if re.search(r"[logLog].*time", b):
            cfg["timescale"], hit = "log", True
        if re.search(r"[nN][sS](?!ta)", b) or re.search(r"[nN]atural.*(cubic|spline)", b):
            cfg["basis"], hit = "ns", True
        if not hit:
            raise ValueError(f"could not discern a baseline hazard from {base_hazard!r}; "
                             "use e.g. 'piecewise constant', 'piecewise linear', "
                             "'Weibull', 'log time' or 'natural cubic spline'")
    if cfg["basis"] not in ("bs", "ns") or cfg["timescale"] not in ("identity", "log"):
        raise ValueError("basis must be 'bs' or 'ns', timescale 'identity' or 'log'")
    return cfg


def baseline_knots(x, cfg):
    kn = _knots(x, cfg["segments"], cfg["degree"], cfg["basis"])
    return np.log(kn) if cfg["timescale"] == "log" else kn


def baseline_basis(x, knots, cfg):
    """create_W0() for one stratum."""
    x = np.asarray(x, float)
    tt = np.log(x) if cfg["timescale"] == "log" else x
    if cfg["basis"] == "bs":
        return _bspline_basis(tt, knots, cfg["degree"])
    from jmbayes.splines import SplineSpec
    kn = np.asarray(knots, float)
    spec = SplineSpec("ns", len(kn) - 1, kn[1:-1], kn[[0, -1]], 3, intercept=True)
    return spec.basis(tt)


def _rows_at_times(times_obs, times_fill):
    """Row (per subject) whose covariates hold at each time: last visit <= t."""
    t = times_obs.copy()
    if t[0] == 0:
        t[0] = -1e4
    else:
        t = np.concatenate([[-1e4], t])
    idx = np.searchsorted(t, times_fill, side="right") - 1
    if times_obs[0] != 0:          # all.inside = TRUE in findInterval
        idx = np.clip(idx, 1, len(t) - 1) - 1
    return np.clip(idx, 0, len(times_obs) - 1)


class TimeDesigner:
    """Designs of the longitudinal sub-models at arbitrary times, as in JMbayes2's
    LongData_HazardModel() + design_matrices_functional_forms(): covariates are taken from
    the last visit at or before t (carried forward), the time variable is replaced by t."""
    EPS = 0.001                                           # slope(): central difference

    def __init__(self, data_filled: pd.DataFrame, time_var: str, sid_col: str = "_sid"):
        self.d, self.time_var = data_filled, time_var
        self.subj_rows = [(di[time_var].to_numpy(), di.index.to_numpy())
                          for _, di in data_filled.groupby(sid_col, sort=True)]
        self._frames = {}

    def data_at(self, T, who=None):
        """T: (r, m) times; row i belongs to subject who[i] (default: row i = subject i)
        -> data frame with r*m rows."""
        if who is None:
            who = range(len(self.subj_rows))
        idx = [self.subj_rows[w][1][_rows_at_times(self.subj_rows[w][0], T[i])]
               for i, w in enumerate(who)]
        out = self.d.loc[np.concatenate(idx)].reset_index(drop=True)
        out[self.time_var] = np.asarray(T, float).ravel()
        return out

    def at(self, T, who, key):
        """Data at times T (rows tied to subjects who), cached under key."""
        if key not in self._frames:
            self._frames[key] = self.data_at(np.asarray(T, float), who)
        return self._frames[key]

    def design(self, form, terms, levels, T, who=None, name=None, center=None,
               zero_ind=()):
        """Design matrix of one functional form at times T (cached per name). form: a
        form name or an assoc.Form with options. Positional calls design(form, terms,
        levels, T, name, center) of earlier versions still work."""
        if isinstance(who, str):                           # old signature
            who, name, center = None, who, name
        f = form if isinstance(form, asc.Form) else asc.Form(form)
        return form_matrix(self, f, terms, levels, np.asarray(T, float), who, name,
                           center, zero_ind)


def form_matrix(des, f, terms, levels, T, who, name, center=None, zero_ind=()):
    """design_matrices_functional_forms() in help_functions.R for one form."""
    tag = f"{name}|{f.ie_time}"

    def X(Tt, key):
        return terms.build(des.at(Tt, who, f"{tag}|{key}"), levels)[0]

    if f.form in ("value", "coefs"):
        M = X(T, "at")
        if center is not None:
            M = M - center
        if f.form == "coefs" and len(zero_ind):
            M = np.ones_like(M)
            M[:, np.asarray(zero_ind, int) - 1] = 0.0
        return M
    if f.form in ("slope", "velocity"):                    # Xbar cancels in differences
        if f.direction == "both":
            t1, t2 = T + f.eps, T - f.eps
        else:
            t1, t2 = T, T - f.eps
        e = (t1 - t2).ravel()
        return (X(t1, f"p{f.eps}{f.direction}") - X(t2, f"m{f.eps}{f.direction}")) / e[:, None]
    if f.form == "acceleration":
        t1, t2 = T + 0.001, T - 0.001
        e = (t1 - t2).ravel()[:, None]
        return (X(t1, "p0.001both") - 2.0 * X(T, "at") + X(t2, "m0.001both")) / (e * e * 0.25)
    if f.form == "area":                                   # average over [0, t] or window
        if f.time_window is None:
            nodes = (T / 2.0)[..., None] * (GK_SK + 1.0)
        else:
            L = np.minimum(T, f.time_window)
            P = 0.5 * L
            nodes = P[..., None] * GK_SK + (T - P)[..., None]
        M = X(nodes.reshape(T.shape[0], -1), f"area{f.time_window}")
        w = np.tile(0.5 * GK_WK, T.size)
        return (w[:, None] * M).reshape(T.size, len(GK_WK), -1).sum(1)
    if f.form == "Delta":
        t2 = (np.zeros_like(T) if f.time_window is None
              else np.maximum(T - f.time_window, 0.0))
        e = (T - t2).ravel()
        if not f.standardise:
            e[:] = 1.0
        else:
            e[e == 0] = 1.0
        return (X(T, "at") - X(t2, f"delta{f.time_window}")) / e[:, None]
    raise NotImplementedError(f"functional form {f.form}()")


def by_columns(by, df, levels, full=()):
    """Covariate columns of an interaction in a functional form (construct_Umat()):
    categorical variables in `full` get one column per level (R's coding when the
    interaction has no lower-order term), others treatment contrasts."""
    import itertools
    cols, names = [], []
    for term in sorted(by.terms, key=len):
        pieces = []
        for fct in term:
            if isinstance(fct, dz.Column) and fct.name in full and \
                    dz._is_categorical(df[fct.name]):
                lv = levels.setdefault(fct.name, dz._levels(df[fct.name]))
                pieces.append([(f"{fct.name}{l}", (df[fct.name] == l).astype(float)
                                .to_numpy()) for l in lv])
            else:
                pieces.append(dz._factor_columns(df, fct, levels))
        for combo in itertools.product(*pieces):
            v = np.ones(len(df))
            for _, x in combo:
                v = v * x
            cols.append(v)
            names.append(":".join(n for n, _ in combo))
    return names, (np.column_stack(cols) if cols else np.zeros((len(df), 0)))


def full_coding(a, alist):
    """Categorical variables of a.by coded with all levels: those whose interaction with
    a's forms has no lower-order term among the association terms (R's rule)."""
    if a.by is None:
        return ()
    out = set()
    for term in a.by.terms:
        for fct in term:
            if not isinstance(fct, dz.Column):
                continue
            rest = tuple(f for f in term if f != fct)
            lower = [b for b in alist if b.factors == a.factors and
                     ((not rest and b.by is None) or
                      (rest and b.by is not None and rest in b.by.terms))]
            if not lower:
                out.add(fct.name)
    return tuple(sorted(out))


def fill_within_subject(dataL: pd.DataFrame, sid_col: str = "_sid") -> pd.DataFrame:
    """Covariates carried forward then backward within subject (locf in JMbayes2)."""
    other = [c for c in dataL.columns if c != sid_col]
    out = dataL.copy()
    out[other] = dataL.groupby(sid_col)[other].ffill()
    out[other] = out.groupby(sid_col)[other].bfill()
    return out


def _create_hc(x, x_info, z_info, sid_obs, df_obs, fixed, levels, random=None):
    """create_HC_X() in help_functions.R, matching columns by their components."""
    def time_varying(col):
        s = pd.Series(col).groupby(sid_obs)
        return not s.apply(lambda v: np.all(np.abs(v - v.iloc[0]) < np.finfo(float).eps ** 0.5)).all()

    x_comps = [c.components for c in x_info]
    z_comps = [c.components for c in z_info]
    n_res, p = len(z_comps), len(x_comps)
    mat = np.zeros((n_res, p))
    for k, zc in enumerate(z_comps):
        if zc in x_comps:
            mat[k, x_comps.index(zc)] = 1
    X_HC = [None] * n_res
    first = ~pd.Series(sid_obs).duplicated().to_numpy()
    notin = [j for j, xc in enumerate(x_comps) if xc not in z_comps]
    base = [j for j in notin if not time_varying(x[:, j])]
    if base:
        X_HC[0] = x[first][:, base]
        mat[0, base] = 2
    for k in range(1, n_res):
        zc = set(z_comps[k])           # fixed-effect interactions containing this column
        xint = [j for j, xc in enumerate(x_comps) if zc < set(xc)]
        if not xint:
            continue
        tmp = df_obs.copy()
        for c in z_info[k].variables:
            tmp[c] = 1.0
        x_tmp = fixed.build(tmp, levels)[0]
        if random is not None:
            # JMbayes2 sets the data variable of the random-effect column to 1. For a
            # function of it (a spline, ns(year)1:sex) that gives ns(1) * sex instead of
            # the multiplier sex, so the centring no longer leaves the linear predictor
            # unchanged (NOTES.md). Divide by the random-effect column at the same data
            # (1 for year itself, so identical to JMbayes2 there).
            zk = random.build(tmp, levels)[0][:, k]
            ok = np.abs(zk) > 1e-12
            for j in xint:
                x_tmp[:, j] = np.where(ok, x_tmp[:, j] / np.where(ok, zk, 1.0), x_tmp[:, j])
        base_k = [j for j in xint if not time_varying(x_tmp[:, j])]
        if base_k:
            X_HC[k] = x_tmp[first][:, base_k]
            mat[k, base_k] = 3
    x_in_z_base = np.where((mat > 0).sum(0) == 1)[0]
    x_notin_z = np.where(mat.sum(0) == 0)[0]
    return mat, X_HC, x_in_z_base, x_notin_z


def _vcov_center(vcov, Xbar):
    """vcov_center() in help_functions.R: covariance of the centred-scale coefficients."""
    V = np.array(vcov, float, copy=True)
    if np.all(np.abs(Xbar) < np.sqrt(np.finfo(float).eps)):
        return V
    xb = Xbar.copy()
    xb[0] = 1.0
    var_temp = xb @ V @ xb
    v = V @ xb
    V[0, 1:] = v[1:]
    V[1:, 0] = v[1:]
    V[0, 0] = var_temp
    return V


def _penalty(p, k, what):
    """priors$penalty_gammas / penalty_alphas: "none", "ridge" or "horseshoe"; as jm(),
    a penalty on a single coefficient is dropped with a warning."""
    p = p or "none"
    if p not in ("none", "ridge", "horseshoe"):
        raise ValueError(f"penalty_{what} must be 'none', 'ridge' or 'horseshoe'")
    if p != "none" and k <= 1:
        import warnings
        warnings.warn(f"it is more meaningful to penalize the '{what}' coefficients if "
                      "there is more than one; no penalty used", UserWarning)
        return "none"
    return p


def _strata_levels(col: pd.Series) -> list:
    levels = (list(map(str, col.cat.categories)) if isinstance(col.dtype, pd.CategoricalDtype)
              else sorted(map(str, pd.unique(col))))
    present = set(col.astype(str))
    return [v for v in levels if v in present]


def _survival_rows(dataS: pd.DataFrame, surv: dict, n: int):
    """The rows of the survival model, as jm() sees them after crisk_setup() /
    rc_setup(): one row per patient, or several (counting-process data: time-varying
    covariates, recurrent events), times one row per cause for competing risks.

    Per row: patient, entry (truncation) time, Time_right, Time_left, delta (0 right
    censored, 1 event, 2 left censored, 3 interval censored), stratum, covariates and the
    group whose covariate / association effects apply (cause, or stratum with
    by_strata)."""
    dataS = dataS.sort_values(["_sid", surv["start"]] if surv.get("start") else ["_sid"],
                              kind="mergesort").reset_index(drop=True)
    pid0 = dataS["_sid"].to_numpy()
    R0 = len(dataS)
    tcol = dataS[surv["time"]].astype(float).to_numpy()
    if surv.get("time2"):                                  # Surv(time, time2, event, "interval")
        kind = "interval"
        code = dataS[surv["event"]].astype(float).to_numpy()
        t2 = dataS[surv["time2"]].astype(float).to_numpy()
        delta0 = code.astype(int)
        T_right0 = tcol.copy()
        T_right0[delta0 == 3] = t2[delta0 == 3]
        T_right0[delta0 == 2] = 0.0
        T_left0 = tcol.copy()
        T_left0[delta0 <= 1] = 0.0
        trunc0 = np.zeros(R0)
    else:
        kind = "counting" if surv.get("start") else "right"
        T_right0, T_left0 = tcol, np.zeros(R0)
        trunc0 = (dataS[surv["start"]].astype(float).to_numpy() if surv.get("start")
                  else np.zeros(R0))
        delta0 = None
    surv_levels = {}
    W_p, w_names_p, _ = surv["covariates"].without_intercept().build(dataS, surv_levels)
    wts0 = (dataS[surv["weights"]].astype(float).to_numpy() if surv.get("weights")
            else None)
    user_levels = _strata_levels(dataS[surv["strata"]]) if surv.get("strata") else None
    user0 = (np.array([user_levels.index(v) for v in dataS[surv["strata"]].astype(str)])
             if user_levels else np.zeros(R0, int))
    n_user = len(user_levels) if user_levels else 1
    causes = surv.get("causes")
    if causes is None:
        if delta0 is None:
            delta0 = dataS[surv["event"]].astype(float).to_numpy().astype(int)
        base, strata, delta = np.arange(R0), user0, delta0
        strata_levels = user_levels
        group, group_levels = ((user0, user_levels) if surv.get("by_strata") and user_levels
                               else (None, None))
    else:
        # crisk_setup(): every row once per cause; event = 1 on the row of the cause
        # that happened; stratum = cause (x the user's strata)
        Kc = len(causes)
        status = dataS[surv["event"]].astype(str).to_numpy()
        base = np.repeat(np.arange(R0), Kc)
        cause = np.tile(np.arange(Kc), R0)
        delta = (status[base] == np.asarray(causes, dtype=str)[cause]).astype(int)
        strata = cause * n_user + user0[base]
        strata_levels = ([str(c) for c in causes] if not user_levels else
                         [f"{c}, {u}" for c in causes for u in user_levels])
        group, group_levels = cause, [str(c) for c in causes]
    R = len(base)
    W0r = W_p[base]
    if group is not None:                    # covariate-major columns, like x:strata(CR)
        G, p0 = len(group_levels), W_p.shape[1]
        W_raw = np.zeros((R, p0 * G))
        for j in range(p0):
            for g in range(G):
                rows = group == g
                W_raw[rows, j * G + g] = W0r[rows, j]
        w_names = [f"{nm} [{c}]" for nm in w_names_p for c in group_levels]
    else:
        W_raw, w_names = W0r, list(w_names_p)
    return dict(kind=kind, base=base, pid0=pid0, R0=R0, row_pat=pid0[base],
                trunc=trunc0[base], T_right=T_right0[base], T_left=T_left0[base],
                delta=np.asarray(delta, int), strata=strata, strata_levels=strata_levels,
                user_levels=user_levels, n_user=n_user, group=group,
                group_levels=group_levels, W_raw=W_raw, w_names=w_names,
                surv_levels=surv_levels, trunc0=trunc0, T_right0=T_right0,
                weights=None if wts0 is None else wts0[base],
                T_left0=T_left0, delta0=delta0, dataS=dataS)


def jm_setup(dataL: pd.DataFrame, dataS: pd.DataFrame, long_models: list,
             surv: dict, id_var: str, time_var: str, *, GK_k: int | None = None,
             base_hazard_segments: int = 9, Bsplines_degree: int = 2,
             diff_order: int = 2, initial_fits: dict | None = None,
             which_independent=None, base_hazard=None, penalty_gammas="none",
             penalty_alphas="none") -> ModelData:
    """Build everything the sampler needs, like jm().

    surv: time, event, covariates, and optionally start (entry / counting-process start),
    time2 (interval censoring), strata, by_strata, causes (competing risks), recurrent
    ("gap" or "calendar"; the first stratum is the recurrent process, the others
    terminal events).

    initial_fits: optional results of the separate fits (e.g. exported from R by
    benchmark/export_initfits.R). When given, they replace the Python fits, so the
    starting values and the prior centres are exactly those JMbayes2 would use:
    {"gammas": [...], "outcomes": [{"betas", "vcov", "D", "b", "b_ids", "log_sigma"}, ...]}
    """
    recurrent = surv.get("recurrent")
    if recurrent not in (None, False, "gap", "calendar"):
        raise ValueError("recurrent must be 'gap' or 'calendar'")
    recurrent = recurrent or None
    if surv.get("time2") and (surv.get("start") or surv.get("causes") is not None
                              or recurrent):
        raise NotImplementedError("interval censoring together with entry times, "
                                  "competing risks or recurrent events")
    if recurrent and surv.get("causes") is not None:
        raise NotImplementedError("recurrent events together with competing risks "
                                  "(give the terminal events as strata)")
    if time_var in surv["covariates"].variables():
        raise NotImplementedError(
            f"the time variable '{time_var}' in the survival covariates; for an effect "
            "that changes over time use an interaction in the association, e.g. "
            f"value(y):ns({time_var}, ...)")
    if GK_k is None:                                 # as jm(): 7 points for Surv(start, ...)
        GK_k = 7 if surv.get("start") else 15
    gk_sk, gk_wk = gauss_kronrod(GK_k)
    # --- longitudinal data, ordered like jm(): by id (first appearance) then time
    ids_unique = list(dict.fromkeys(dataL[id_var]))
    id_rank = {g: i for i, g in enumerate(ids_unique)}
    dataL = dataL.assign(_sid=dataL[id_var].map(id_rank).astype("int64"))
    dataL = dataL.sort_values(["_sid", time_var], kind="mergesort").reset_index(drop=True)
    n = len(ids_unique)
    # covariates carried forward/backward within subject (as in design_matrices_functional_forms)
    dataL_filled = fill_within_subject(dataL)

    # --- survival data
    dataS = dataS.assign(_sid=dataS[id_var].map(id_rank))
    if dataS["_sid"].isna().any() or set(dataS["_sid"]) != set(range(n)):
        raise ValueError("survival data must have rows for exactly the subjects in the "
                         "longitudinal data")
    if not surv.get("start") and len(dataS) != n:
        raise ValueError("survival data must have one row per subject (several rows per "
                         "subject need start/stop times)")
    dataS = dataS.assign(_sid=dataS["_sid"].astype("int64"))
    sr = _survival_rows(dataS, surv, n)
    R, base, row_pat = len(sr["base"]), sr["base"], sr["row_pat"]
    trunc, T_right, T_left, delta = sr["trunc"], sr["T_right"], sr["T_left"], sr["delta"]
    strata, strata_levels = sr["strata"], sr["strata_levels"]
    n_strata = 1 if strata_levels is None else len(strata_levels)
    if strata_levels is None:
        strata = np.zeros(R, int)
    W_raw, w_names = sr["W_raw"], sr["w_names"]
    event = (delta == 1).astype(float)
    irows = np.where(delta == 3)[0]
    left = np.where(delta == 2)[0]
    if initial_fits is not None:
        gammas_cox = np.atleast_1d(np.asarray(initial_fits["gammas"], float))
        if W_raw.shape[1] == 0:
            gammas_cox = np.zeros(0)
    elif W_raw.shape[1] == 0:
        gammas_cox = np.zeros(0)
    elif sr["kind"] == "interval":                  # survreg(): -coef / scale
        if surv.get("weights"):
            raise NotImplementedError("case weights with interval censoring")
        gammas_cox = fit_weibull_interval(T_left, T_right, delta, W_raw)
    else:
        gammas_cox = fit_coxph(T_right, event, W_raw,
                               strata if strata_levels is not None else None,
                               trunc if surv.get("start") else None,
                               weights=sr["weights"])

    # --- quadrature: [entry, Time_integration] per row (Time_left for left- and
    # interval-censored rows), plus [Time_left, Time_right] for interval-censored rows
    T_int = T_right.copy()
    T_int[delta >= 2] = T_left[delta >= 2]
    P = (T_int - trunc) / 2.0
    st = P[:, None] * gk_sk[None, :] + ((T_int + trunc) / 2.0)[:, None]      # (R, GK_k)
    log_Pwk = (np.log(P)[:, None] + np.log(gk_wk)[None, :]).ravel()
    P2 = np.abs(T_right[irows] - T_int[irows]) / 2.0
    st2 = P2[:, None] * gk_sk[None, :] + ((T_right[irows] + T_int[irows]) / 2.0)[:, None]
    log_Pwk2 = (np.log(P2)[:, None] + np.log(gk_wk)[None, :]).ravel()
    n_main = R * GK_k
    pt_row = np.concatenate([np.repeat(np.arange(R), GK_k), np.repeat(irows, GK_k)])
    log_Pwk = np.concatenate([log_Pwk, log_Pwk2])
    st_all = np.concatenate([st.ravel(), st2.ravel()])
    pt_strata = strata[pt_row]
    gap = recurrent == "gap"
    x_H = st_all - (trunc[pt_row] if gap else 0.0)
    # (JMbayes2 evaluates the second interval integral on the calendar scale too)
    x_h = T_right - (trunc if gap else 0.0)

    # knots per stratum from range(Time_right) (gap: Time_right - entry)
    bh = (base_hazard if isinstance(base_hazard, dict) else
          base_hazard_config(base_hazard, degree=Bsplines_degree,
                             segments=base_hazard_segments))
    base_hazard_segments = bh["segments"]
    knots_strata = [baseline_knots(x_h[strata == k], bh) for k in range(n_strata)]
    knots = knots_strata[0]
    nb0 = baseline_basis(np.array([x_h.max()]), knots, bh).shape[1]
    W0_H = np.zeros((len(st_all), nb0 * n_strata))
    W0_h = np.zeros((R, nb0 * n_strata))
    for k in range(n_strata):
        rH, rh = pt_strata == k, strata == k
        W0_H[rH, k * nb0:(k + 1) * nb0] = baseline_basis(x_H[rH], knots_strata[k], bh)
        W0_h[rh, k * nb0:(k + 1) * nb0] = baseline_basis(x_h[rh], knots_strata[k], bh)

    W_H = W_raw[pt_row]
    W_h = W_raw.copy()
    W_bar = W_H[:n_main].mean(0)
    W_sds = W_H[:n_main].std(0, ddof=1)
    W_H = (W_H - W_bar) / W_sds
    W_h = (W_h - W_bar) / W_sds

    # --- designs at quadrature points (H) and event times (h), on the rows before the
    # competing-risk expansion, each with its patient's longitudinal data
    designer = TimeDesigner(dataL_filled, time_var)
    R0, pid0 = sr["R0"], sr["pid0"]
    tr0, Tint0 = sr["trunc0"], sr["T_right0"].copy()
    d0 = sr["delta0"]
    if d0 is not None:
        Tint0[d0 >= 2] = sr["T_left0"][d0 >= 2]
    st0 = (((Tint0 - tr0) / 2.0)[:, None] * gk_sk[None, :]
           + ((Tint0 + tr0) / 2.0)[:, None])
    time_sets = {"H": (st0, pid0), "h": (sr["T_right0"][:, None], pid0)}
    if len(irows):
        time_sets["H2"] = (st2, pid0[irows])
    expand = R != R0

    dS0 = sr["dataS"]

    def capped(f, where):
        """Times of a time set, capped at the form's intervention time (IE_time)."""
        T, who = time_sets[where]
        if f.ie_time is None:
            return T, who
        ie = dS0[f.ie_time].astype(float).to_numpy()
        ie = ie[irows] if where == "H2" else ie
        return np.minimum(T, ie[:, None]), who

    def form_design(f, terms, levels_, where, center=None, zero_ind=()):
        if where == "H":
            D = designer.design(f, terms, levels_, *capped(f, "H"), "H", center, zero_ind)
            if expand:
                D = D.reshape(R0, GK_k, -1)[base].reshape(R * GK_k, -1)
            if len(irows):
                D2 = designer.design(f, terms, levels_, *capped(f, "H2"), "H2", center,
                                     zero_ind)
                D = np.concatenate([D, D2])
            return D
        D = designer.design(f, terms, levels_, *capped(f, "h"), "h", center, zero_ind)
        return D[base] if expand else D

    def surv_frame(T, rows):
        """Survival rows at times T, with the time variable set to those times (as
        SurvData_HazardModel()), for covariate interactions in functional forms."""
        df = dS0.iloc[np.repeat(rows, T.shape[1])].reset_index(drop=True)
        df[time_var] = np.asarray(T, float).ravel()
        return df

    by_frames = {}

    def by_design(by, levels_, full):
        """Interaction columns at all quadrature points (U_H) and event times (U_h)."""
        if "H" not in by_frames:
            by_frames["H"] = surv_frame(st0, np.arange(R0))
            by_frames["h"] = surv_frame(sr["T_right0"][:, None], np.arange(R0))
            if len(irows):
                by_frames["H2"] = surv_frame(st2, irows)
        names, UH = by_columns(by, by_frames["H"], levels_, full)
        _, Uh = by_columns(by, by_frames["h"], levels_, full)
        if expand:
            UH = UH.reshape(R0, GK_k, -1)[base].reshape(R * GK_k, -1)
            Uh = Uh[base]
        if len(irows):
            UH = np.concatenate([UH, by_columns(by, by_frames["H2"], levels_, full)[1]])
        return names, UH, Uh

    # --- per-outcome data and separate fits
    outcomes, fits = [], []
    re_off, fe_off = 0, 0
    beta_names = []
    hc_parts = []
    group, group_levels = sr["group"], sr["group_levels"]
    for m in long_models:
        resp, fixed, random = m["response"], m["fixed"], m["random"]
        lhs = resp.name
        aux_col = m.get("aux")
        needed = sorted({resp.var, *fixed.variables(), *random.variables(),
                         *([aux_col] if isinstance(aux_col, str) else [])})
        missing = [c for c in needed if c not in dataL.columns]
        if missing:
            raise KeyError(f"columns {missing} not found in the longitudinal data")
        ok = dataL[needed].notna().all(axis=1).to_numpy()
        d = dataL.loc[ok].reset_index(drop=True)
        levels = {}
        y = resp.values(d)
        aux = (None if aux_col is None else
               d[aux_col].astype(float).to_numpy() if isinstance(aux_col, str)
               else np.full(len(d), float(aux_col)))
        X, x_names, x_info = fixed.build(d, levels)
        Z, z_names, z_info = random.build(d, levels)
        sid = d["_sid"].to_numpy()
        fam = m.get("family", "gaussian")
        if fam not in FAMILIES:
            raise NotImplementedError(f"family {fam}")
        df_t = float(m.get("df") or 0.0)
        if initial_fits is not None:
            f = initial_fits["outcomes"][len(fits)]
            row = {str(g): i for i, g in enumerate(f["b_ids"])}
            b_all = np.asarray(f["b"], float)
            present = np.unique(sid)
            fit = MixedFit(np.asarray(f["betas"], float), np.asarray(f["vcov"], float),
                           np.asarray(f["D"], float),
                           b_all[[row[str(ids_unique[i])] for i in present]],
                           float(f["log_sigma"]))
        elif fam == "gaussian":
            fit = fit_lme(y, X, Z, sid, diagonal=m.get("uncorrelated", False))
        else:
            fit = fit_glmm(y, X, Z, sid, fam, df_t, diagonal=m.get("uncorrelated", False),
                           aux=aux)

        mat, X_HC, x_in_z_base, x_notin_z = _create_hc(X, x_info, z_info, sid, d, fixed,
                                                     levels, random)
        # center_X(): only the columns outside the hierarchical centering are centred
        Xbar = np.zeros(X.shape[1])
        Xbar[x_notin_z] = X[:, x_notin_z].mean(0)
        X = X - Xbar
        # functional forms ("slots": one design per form and options) at quadrature
        # points and event times, and the association columns built from them
        alist = asc.as_assoc_list(m.get("forms"))
        slots, slot_of = [], {}

        def slot(f):
            if f not in slot_of:
                slot_of[f] = len(slots)
                slots.append(dict(form=f,
                                  X_H=form_design(f, fixed, levels, "H", Xbar, f.zero_ind_X),
                                  Z_H=form_design(f, random, levels, "H", None, f.zero_ind_Z),
                                  X_h=form_design(f, fixed, levels, "h", Xbar, f.zero_ind_X),
                                  Z_h=form_design(f, random, levels, "h", None, f.zero_ind_Z)))
            return slot_of[f]

        assoc = []
        by_levels = {}
        pt_group = None if group is None else group[pt_row]
        for a in alist:
            factors = [(slot(f), t) for f, t in a.factors]
            label = a.label(lhs)
            if a.by is None:
                parts = [(label, None, None, None)]
            else:
                full = full_coding(a, alist)
                bn, UH, Uh = by_design(a.by, by_levels, full)
                parts = [(f"{label}:{nm}", UH[:, c], Uh[:, c], (a.by, full, c))
                         for c, nm in enumerate(bn)]
            for name, uH, uh, byinfo in parts:
                col = dict(form=label, name=name, factors=factors, by=byinfo,
                           u_H=uH, u_h=uh)
                if group is None:
                    assoc.append({k: v for k, v in col.items() if v is not None})
                    continue
                for k, c in enumerate(group_levels):  # value(x):CR - one column per group
                    gH = (pt_group == k).astype(float)
                    gh = (group == k).astype(float)
                    assoc.append(dict({k_: v for k_, v in col.items() if v is not None},
                                      cause=c, u_H=gH if uH is None else gH * uH,
                                      u_h=gh if uh is None else gh * uh))
        qj, pj = Z.shape[1], X.shape[1]
        outcomes.append(Outcome(lhs, fam, y, X, Z, sid, x_names, z_names, assoc,
                                np.arange(re_off, re_off + qj),
                                np.arange(fe_off, fe_off + pj),
                                has_sigma(fam), response=resp, fixed=fixed, random=random,
                                levels=levels, nhc=np.asarray(x_notin_z, int), Xbar=Xbar,
                                df=df_t, aux=aux, aux_col=aux_col, slots=slots,
                                by_levels=by_levels))
        hc_parts.append((mat, X_HC, x_in_z_base, sid))
        fits.append(fit)
        beta_names += [f"{lhs}: {xn}" for xn in x_names]
        re_off += qj
        fe_off += pj
    q = re_off

    # --- hierarchical centering (create_X_dot) and missing-outcome patterns
    present = np.zeros((n, len(outcomes)), bool)
    for j, o in enumerate(outcomes):
        present[np.unique(o.sid), j] = True
    ind_FE_HC, hc_off = [], 0
    hc_cols_per_outcome = []
    for o, (mat, _, xb, _) in zip(outcomes, hc_parts):
        ind_FE_HC += list(o.fe_cols[xb])
        hc_cols_per_outcome.append(np.arange(hc_off, hc_off + len(xb)))
        hc_off += len(xb)
    p_HC = hc_off
    X_dot = np.zeros((n, q, p_HC))
    for o, (mat, X_HC, xb, sid), hc_cols in zip(outcomes, hc_parts, hc_cols_per_outcome):
        ids = np.unique(sid)  # subjects present, in order (sid is sorted)
        for k in range(len(o.z_names)):
            r = o.re_cols[k]
            for c_local, c in enumerate(xb):
                if mat[k, c] == 1:
                    X_dot[ids, r, hc_cols[c_local]] = 1.0
            if X_HC[k] is not None:
                base_cols = [c_local for c_local, c in enumerate(xb) if mat[k, c] > 1]
                X_dot[ids[:, None], r, hc_cols[base_cols][None, :]] = X_HC[k]
    patterns = []
    for patt in np.unique(present, axis=0):
        subj = np.where((present == patt).all(1))[0]
        re_idx = np.concatenate([o.re_cols for o, pr in zip(outcomes, patt) if pr])
        fe_idx = np.concatenate([c for c, pr in zip(hc_cols_per_outcome, patt) if pr])
        patterns.append((subj, re_idx, fe_idx))

    # --- initial values (jm.R lines 665-700)
    betas = [f.betas for f in fits]
    log_sigmas = np.array([f.log_sigma for f in fits])
    # bdiag2(): block diagonal with 1e-5 between outcomes, 0 for independent outcomes
    D = np.full((q, q), 1e-5)
    for o, f in zip(outcomes, fits):
        D[np.ix_(o.re_cols, o.re_cols)] = f.D
    if which_independent is not None:
        pairs = ([(a, b) for a in range(len(outcomes)) for b in range(a + 1, len(outcomes))]
                 if isinstance(which_independent, str) and which_independent == "all"
                 else [(a - 1, b - 1) for a, b in which_independent])
        for a, b in pairs:
            if a == b or not (0 <= a < len(outcomes) and 0 <= b < len(outcomes)):
                raise ValueError(f"which_independent: invalid pair {(a + 1, b + 1)}")
            D[np.ix_(outcomes[a].re_cols, outcomes[b].re_cols)] = 0.0
            D[np.ix_(outcomes[b].re_cols, outcomes[a].re_cols)] = 0.0
    if np.linalg.eigvalsh(D).min() <= 0:
        raise NotImplementedError("initial D not positive definite (nearPD not ported)")
    # correlations fixed at zero: upper triangle, column-major order (as which(arr.ind))
    zero_tol = np.sqrt(np.finfo(float).eps)
    ind_zero = [(r, c) for c in range(q) for r in range(c) if abs(D[r, c]) < zero_tol]
    b = np.zeros((n, q))
    for o, f in zip(outcomes, fits):
        b[np.ix_(np.unique(o.sid), o.re_cols)] = f.b
    nb = nb0                                           # spline coefficients per stratum
    n_alphaF = max(n_strata - 1, 1)
    penalized_bs = not bh.get("weibull", False)     # JMbayes2: Weibull not penalised
    inits = dict(betas=np.concatenate(betas), log_sigmas=log_sigmas, D=D, b=b,
                 bs_gammas=np.full(nb * n_strata, -0.1), gammas=gammas_cox.copy(),
                 alphas=np.zeros(sum(len(o.assoc) for o in outcomes)),
                 tau_bs_gammas=np.full(n_strata,
                                       (20.0 if base_hazard_segments > 1 else 0.001)
                                       if penalized_bs else 1.0))
    if recurrent:
        inits.update(frailty=np.zeros(n), alphaF=np.zeros(n_alphaF), sigmaF=np.array([0.1]))

    # --- Wlong scaling, from the (unjittered) initial values (create_Wlong_mats), over the
    # quadrature points of the main integral
    sid_H = row_pat[pt_row]
    def wcol(o, bj, a):
        out = a.get("u_H", 1.0)
        for k, t in a["factors"]:
            sl = o.slots[k]
            eta = sl["X_H"] @ bj + (sl["Z_H"] * b[sid_H][:, o.re_cols]).sum(1)
            out = out * asc.np_transform(t, eta)
        return out

    Wl = np.column_stack([wcol(o, bj, a) for o, bj in zip(outcomes, betas)
                          for a in o.assoc])[:n_main]
    assoc_names = [a["name"] + (f" [{a['cause']}]" if "cause" in a else "")
                   for o in outcomes for a in o.assoc]
    assoc_outcome = np.array([j for j, o in enumerate(outcomes) for _ in o.assoc])
    n_assoc = len(assoc_names)
    Wlong_bar = Wl.mean(0)
    Wlong_sds = Wl.std(0, ddof=1)

    # --- priors (jm.R lines 710-760)
    Dd = np.diff(np.eye(nb), n=diff_order if nb > 3 else 1, axis=0)
    Tau_bs1 = Dd.T @ Dd
    rank_bs = float(np.linalg.matrix_rank(Tau_bs1))
    if not penalized_bs:
        Tau_bs1 = 0.1 * np.eye(nb)
    Tau_bs = np.kron(np.eye(n_strata), Tau_bs1)       # one penalty block per stratum
    # prior means: the fitted fixed effects, intercept moved to the centred scale;
    # prior precisions: weak_informative_Tau() on the centred covariance (vcov_center)
    p_all = sum(len(bj) for bj in betas)
    mean_all = np.zeros(p_all)
    Tau_all = np.zeros((p_all, p_all))
    mean_nhc, Tau_nhc = [], []
    for o, f in zip(outcomes, fits):
        mb = f.betas.copy()
        mb[0] += o.Xbar @ f.betas
        V = _vcov_center(f.vcov, o.Xbar)
        T = np.diag(1.0 / np.minimum(14400.0 * np.diag(V), 1000.0))
        mean_all[o.fe_cols] = mb
        Tau_all[np.ix_(o.fe_cols, o.fe_cols)] = T
        mean_nhc.append(mb[o.nhc])
        Tau_nhc.append(T[np.ix_(o.nhc, o.nhc)])
    ind_FE_HC = np.array(ind_FE_HC)
    priors = dict(
        mean_betas_HC=mean_all[ind_FE_HC],
        Tau_betas_HC=Tau_all[np.ix_(ind_FE_HC, ind_FE_HC)],
        mean_betas_nHC=mean_nhc, Tau_betas_nHC=Tau_nhc,
        mean_bs_gammas=np.zeros(nb * n_strata), Tau_bs_gammas=Tau_bs,
        A_tau_bs_gammas=5.0, B_tau_bs_gammas=0.5,
        rank_Tau_bs_gammas=rank_bs,                                    # per stratum
        penalized_bs_gammas=penalized_bs,
        penalty_gammas=_penalty(penalty_gammas, len(gammas_cox), "gammas"),
        penalty_alphas=_penalty(penalty_alphas, n_assoc, "alphas"),
        A_lambda_gammas=1.0 if penalty_gammas == "ridge" else 0.5, B_lambda_gammas=1.0,
        A_tau_gammas=0.5, B_tau_gammas=1.0, A_nu_gammas=0.5, B_nu_gammas=1.0,
        A_xi_gammas=0.5, B_xi_gammas=1.0,
        A_lambda_alphas=0.5, B_lambda_alphas=1.0, A_tau_alphas=0.5, B_tau_alphas=1.0,
        A_nu_alphas=0.5, B_nu_alphas=1.0, A_xi_alphas=0.5, B_xi_alphas=1.0,
        mean_gammas=gammas_cox.copy(), Tau_gammas=np.eye(len(gammas_cox)) * 0.25,
        mean_alphas=np.zeros(n_assoc), Tau_alphas=np.eye(n_assoc) * 0.25,
        D_sds_shape=5.0, D_sds_mean=np.sqrt(np.diag(D)), D_L_etaLKJ=3.0,
        sigmas_shape=5.0, sigmas_mean=np.exp(log_sigmas),
    )
    if recurrent:
        priors.update(mean_alphaF=np.zeros(n_alphaF), Tau_alphaF=np.eye(n_alphaF) * 0.25,
                      sigmaF_shape=0.25 / 0.4, sigmaF_mean=0.25)
    iu = [(r, c) for c in range(q) for r in range(c)   # trimatu_ind(.., 1), column-major
          if (r, c) not in set(ind_zero)]
    identity_rows = R == n and np.array_equal(row_pat, np.arange(n))
    cr = surv.get("causes")
    return ModelData(n=n, q=q, outcomes=outcomes, time=T_right, event=event,
                     log_Pwk=log_Pwk, W0_H=W0_H, W0_h=W0_h, W_H=W_H, W_h=W_h,
                     W_bar=W_bar, W_sds=W_sds, w_names=w_names,
                     Wlong_bar=Wlong_bar, Wlong_sds=Wlong_sds, assoc_names=assoc_names,
                     assoc_outcome=assoc_outcome, knots=knots, GK_k=GK_k,
                     ind_FE_HC=ind_FE_HC, X_dot=X_dot, patterns=patterns,
                     upper_part=np.array(iu, dtype=int).reshape(-1, 2),
                     ind_zero_D=np.array(ind_zero, dtype=int).reshape(-1, 2),
                     priors=priors, inits=inits,
                     beta_names=beta_names, sid_names=ids_unique,
                     meta=dict(id_var=id_var, time_var=time_var, surv=dict(surv),
                               surv_levels=sr["surv_levels"],
                               Bsplines_degree=bh["degree"], base_hazard=bh),
                     idT=None if identity_rows else row_pat,
                     strata=strata if strata_levels is not None else None,
                     causes=None if cr is None else list(cr),
                     strata_levels=strata_levels, knots_strata=knots_strata, trunc=trunc,
                     delta=delta, pt_row=pt_row, n_main_H=n_main, interval_rows=irows,
                     left_rows=left, group=group, group_levels=group_levels,
                     user_strata_levels=sr["user_levels"], recurrent=recurrent,
                     surv_kind=sr["kind"], weights=sr["weights"])
