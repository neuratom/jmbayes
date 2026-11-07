"""Model setup: the Python counterpart of jm() in R/jm.R and jm_fit() in R/jm_fit.R.

Scope of this first version (matches the README pbc2 model):
  * longitudinal families: gaussian (identity link), binomial (logit link)
  * functional form: value() of each outcome, no transformations
  * one stratum, right-censored event times, no recurrent events / weights
Anything outside this scope raises NotImplementedError rather than silently
fitting a different model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from jmbayes2py import design as dz
from jmbayes2py.fits import MixedFit, fit_coxph, fit_glmm_binomial, fit_lme

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


def _knots(time_right, segments, degree):
    # knots() in help_functions.R (basis = "bs")
    xl = np.sqrt(np.finfo(float).eps)
    xr = time_right.max() + 0.001
    kn = np.linspace(xl, xr, segments + 1)
    return np.concatenate([np.repeat(xl, degree), kn, np.repeat(xr, degree)])


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

    def data_at(self, T):
        """T: (n, m) times per subject -> data frame with n*m rows."""
        idx = [ridx[_rows_at_times(tt, T[i])] for i, (tt, ridx) in enumerate(self.subj_rows)]
        out = self.d.loc[np.concatenate(idx)].reset_index(drop=True)
        out[self.time_var] = np.asarray(T, float).ravel()
        return out

    def frame(self, key, name, T):
        if (key, name) not in self._frames:
            if key == "at":
                f = self.data_at(T)
            elif key == "plus":
                f = self.data_at(T + self.EPS)
            elif key == "minus":
                f = self.data_at(T - self.EPS)
            elif key == "area":                            # 15 GK nodes on [0, t] per time
                nodes = (T / 2.0)[..., None] * (GK_SK + 1.0)
                f = self.data_at(nodes.reshape(T.shape[0], -1))
            self._frames[(key, name)] = f
        return self._frames[(key, name)]

    def design(self, form, terms, levels, T, name, center=None):
        """Design matrix of one association form at times T (cached per name)."""
        if form == "value":
            M = terms.build(self.frame("at", name, T), levels)[0]
            return M - center if center is not None else M
        if form == "slope":                                # Xbar cancels in the difference
            M1 = terms.build(self.frame("plus", name, T), levels)[0]
            M2 = terms.build(self.frame("minus", name, T), levels)[0]
            e = ((T + self.EPS) - (T - self.EPS)).ravel()
            return (M1 - M2) / e[:, None]
        if form == "area":                                 # average of the trajectory on [0, t]
            M = terms.build(self.frame("area", name, T), levels)[0]
            w = np.tile(0.5 * GK_WK, T.size)
            return (w[:, None] * M).reshape(T.size, len(GK_WK), -1).sum(1)
        raise NotImplementedError(f"functional form {form}()")


def fill_within_subject(dataL: pd.DataFrame, sid_col: str = "_sid") -> pd.DataFrame:
    """Covariates carried forward then backward within subject (locf in JMbayes2)."""
    other = [c for c in dataL.columns if c != sid_col]
    out = dataL.copy()
    out[other] = dataL.groupby(sid_col)[other].ffill()
    out[other] = out.groupby(sid_col)[other].bfill()
    return out


def _create_hc(x, x_info, z_info, sid_obs, df_obs, fixed, levels):
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


def jm_setup(dataL: pd.DataFrame, dataS: pd.DataFrame, long_models: list,
             surv: dict, id_var: str, time_var: str, *, GK_k: int = 15,
             base_hazard_segments: int = 9, Bsplines_degree: int = 2,
             diff_order: int = 2, initial_fits: dict | None = None,
             which_independent=None) -> ModelData:
    """Build everything the sampler needs, like jm().

    initial_fits: optional results of the separate fits (e.g. exported from R by
    benchmark/export_initfits.R). When given, they replace the Python fits, so the
    starting values and the prior centres are exactly those JMbayes2 would use:
    {"gammas": [...], "outcomes": [{"betas", "vcov", "D", "b", "b_ids", "log_sigma"}, ...]}
    """
    if GK_k != 15:
        raise NotImplementedError("only GK_k = 15")
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
    if dataS["_sid"].isna().any() or len(dataS) != n:
        raise ValueError("survival data must have one row per subject in the longitudinal data")
    dataS = dataS.assign(_sid=dataS["_sid"].astype("int64"))
    dataS = dataS.sort_values("_sid").reset_index(drop=True)
    time = dataS[surv["time"]].astype(float).to_numpy()
    event = dataS[surv["event"]].astype(float).to_numpy()
    if not np.all(np.isin(event, [0, 1])):
        raise NotImplementedError("only right-censored data")
    surv_levels = {}
    W_raw, w_names, _ = surv["covariates"].without_intercept().build(dataS, surv_levels)
    if initial_fits is None:
        gammas_cox = fit_coxph(time, event, W_raw)
    else:
        gammas_cox = np.atleast_1d(np.asarray(initial_fits["gammas"], float))

    # --- quadrature
    P = time / 2.0
    st = P[:, None] * GK_SK[None, :] + P[:, None]                # (n, 15)
    log_Pwk = (np.log(P)[:, None] + np.log(GK_WK)[None, :]).ravel()
    st_flat = st.ravel()
    sid_H = np.repeat(np.arange(n), GK_k)

    knots = _knots(time, base_hazard_segments, Bsplines_degree)
    W0_H = _bspline_basis(st_flat, knots, Bsplines_degree)
    W0_h = _bspline_basis(time, knots, Bsplines_degree)

    W_H = W_raw[sid_H]
    W_h = W_raw.copy()
    W_bar = W_H.mean(0)
    W_sds = W_H.std(0, ddof=1)
    W_H = (W_H - W_bar) / W_sds
    W_h = (W_h - W_bar) / W_sds

    # --- designs at quadrature points (H) and event times (h)
    designer = TimeDesigner(dataL_filled, time_var)
    time_sets = {"H": st, "h": time[:, None]}

    def form_design(form, terms, levels_, where, center=None):
        return designer.design(form, terms, levels_, time_sets[where], where, center)

    # --- per-outcome data and separate fits
    outcomes, fits = [], []
    re_off, fe_off = 0, 0
    beta_names = []
    hc_parts = []
    for m in long_models:
        resp, fixed, random = m["response"], m["fixed"], m["random"]
        lhs = resp.name
        needed = sorted({resp.var, *fixed.variables(), *random.variables()})
        missing = [c for c in needed if c not in dataL.columns]
        if missing:
            raise KeyError(f"columns {missing} not found in the longitudinal data")
        ok = dataL[needed].notna().all(axis=1).to_numpy()
        d = dataL.loc[ok].reset_index(drop=True)
        levels = {}
        y = resp.values(d)
        X, x_names, x_info = fixed.build(d, levels)
        Z, z_names, z_info = random.build(d, levels)
        sid = d["_sid"].to_numpy()
        fam = m.get("family", "gaussian")
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
        elif fam == "binomial":
            fit = fit_glmm_binomial(y, X, Z, sid, diagonal=m.get("uncorrelated", False))
        else:
            raise NotImplementedError(f"family {fam}")

        mat, X_HC, x_in_z_base, x_notin_z = _create_hc(X, x_info, z_info, sid, d, fixed,
                                                     levels)
        # center_X(): only the columns outside the hierarchical centering are centred
        Xbar = np.zeros(X.shape[1])
        Xbar[x_notin_z] = X[:, x_notin_z].mean(0)
        X = X - Xbar
        # association designs at quadrature points and at event times
        assoc = []
        for form in m.get("forms", ["value"]):
            assoc.append(dict(
                form=form,
                X_H=form_design(form, fixed, levels, "H", Xbar),
                Z_H=form_design(form, random, levels, "H"),
                X_h=form_design(form, fixed, levels, "h", Xbar),
                Z_h=form_design(form, random, levels, "h")))
        qj, pj = Z.shape[1], X.shape[1]
        outcomes.append(Outcome(lhs, fam, y, X, Z, sid, x_names, z_names, assoc,
                                np.arange(re_off, re_off + qj),
                                np.arange(fe_off, fe_off + pj),
                                fam == "gaussian", response=resp, fixed=fixed, random=random,
                                levels=levels, nhc=np.asarray(x_notin_z, int), Xbar=Xbar))
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
    nres = [len(o.re_cols) for o in outcomes]
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
    nb = W0_H.shape[1]
    inits = dict(betas=np.concatenate(betas), log_sigmas=log_sigmas, D=D, b=b,
                 bs_gammas=np.full(nb, -0.1), gammas=gammas_cox.copy(),
                 alphas=np.zeros(sum(len(o.assoc) for o in outcomes)),
                 tau_bs_gammas=np.array([20.0 if base_hazard_segments > 1 else 0.001]))

    # --- Wlong scaling, from the (unjittered) initial values (create_Wlong_mats)
    Wl = np.column_stack([a["X_H"] @ bj + (a["Z_H"] * b[sid_H][:, o.re_cols]).sum(1)
                          for o, bj in zip(outcomes, betas) for a in o.assoc])
    assoc_names = [f"{a['form']}({o.name})" for o in outcomes for a in o.assoc]
    assoc_outcome = np.array([j for j, o in enumerate(outcomes) for _ in o.assoc])
    n_assoc = len(assoc_names)
    Wlong_bar = Wl.mean(0)
    Wlong_sds = Wl.std(0, ddof=1)

    # --- priors (jm.R lines 710-760)
    Dd = np.diff(np.eye(nb), n=diff_order if nb > 3 else 1, axis=0)
    Tau_bs = Dd.T @ Dd
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
        mean_bs_gammas=np.zeros(nb), Tau_bs_gammas=Tau_bs,
        A_tau_bs_gammas=5.0, B_tau_bs_gammas=0.5,
        rank_Tau_bs_gammas=float(np.linalg.matrix_rank(Tau_bs)),
        mean_gammas=gammas_cox.copy(), Tau_gammas=np.eye(len(gammas_cox)) * 0.25,
        mean_alphas=np.zeros(n_assoc), Tau_alphas=np.eye(n_assoc) * 0.25,
        D_sds_shape=5.0, D_sds_mean=np.sqrt(np.diag(D)), D_L_etaLKJ=3.0,
        sigmas_shape=5.0, sigmas_mean=np.exp(log_sigmas),
    )
    iu = [(r, c) for c in range(q) for r in range(c)   # trimatu_ind(.., 1), column-major
          if (r, c) not in set(ind_zero)]
    return ModelData(n=n, q=q, outcomes=outcomes, time=time, event=event,
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
                               surv_levels=surv_levels, Bsplines_degree=Bsplines_degree))
