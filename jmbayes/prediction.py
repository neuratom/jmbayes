"""Dynamic predictions for new patients, as predict.jm() in JMbayes2.

For a patient with measurements up to time t who is event-free at t:
  1. take posterior draws of the parameters (the first n_samples draws, as JMbayes2);
  2. for each draw, sample the patient's random effects from p(b | y, T > t, theta) with
     a random-walk Metropolis-Hastings run of n_mcmc iterations, keeping the last value;
  3. compute the cumulative risk P(T <= u | T > t) = 1 - S(u) / S(t) at future times u,
     or the expected marker values;
  4. report the mean over draws and the (1 - level) / 2, (1 + level) / 2 quantiles.

Difference in layout from JMbayes2: R runs the draws one after another, each starting
from the previous draw's random effects; here all draws run in parallel (one tensor
operation for all draws and patients), each starting from zero. Agreement with R is
checked in tests/test_predict.py.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from jmbayes import families as fam
from jmbayes.model import (GK_SK, GK_WK, TimeDesigner, _bspline_basis,
                              fill_within_subject)

LOG2PI = math.log(2 * math.pi)
# gaussKronrod(7) in help_functions.R: the 7-point Gauss rule (competing-risk CIFs)
GK7_SK = GK_SK[:7]
GK7_WK = np.array([0.12948496616887, 0.279705391489277, 0.381830050505119,
                   0.417959183673469, 0.381830050505119, 0.279705391489277,
                   0.12948496616887])


# ------------------------------------------------------------------ parameters
def _posterior_draws(fit, n_samples, draws="spread"):
    """n_samples posterior draws, converted back to the scale the sampler works on.

    draws="spread": evenly spaced over all chains (default; lower Monte Carlo error).
    draws="first":  the first n_samples draws with chains stacked in order, exactly as
                    JMbayes2 does (docall_rbind + 1:n_samples)."""
    md = fit.md

    def stacked(block):
        d = fit.blocks[block][0]                          # (C, S, k)
        return d.reshape(-1, d.shape[-1])

    M = stacked("bs_gammas").shape[0]
    if n_samples > M:
        n_samples = M
    if draws == "first":
        sl = np.arange(n_samples)
    elif draws == "spread":
        sl = np.unique(np.linspace(0, M - 1, n_samples).round().astype(int))
        n_samples = len(sl)
    else:
        raise ValueError("draws must be 'spread' or 'first'")
    p = sum(len(o.x_names) for o in md.outcomes)
    betas = np.zeros((n_samples, p))
    for j, o in enumerate(md.outcomes):
        bj = stacked(f"betas{j + 1}")[sl].copy()
        bj[:, 0] += bj @ o.Xbar                           # intercept to the centred scale
        betas[:, o.fe_cols] = bj
    sig_all = np.ones((n_samples, len(md.outcomes)))
    has = [o.has_sigma for o in md.outcomes]
    if any(has):
        sig_all[:, has] = stacked("sigmas")[sl]
    D_low = stacked("D")[sl]
    q = md.q
    rows, cols = np.tril_indices(q)
    order = np.lexsort((rows, cols))
    D = np.zeros((n_samples, q, q))
    D[:, rows[order], cols[order]] = D_low
    D = D + np.transpose(np.tril(D, -1), (0, 2, 1))
    sds = np.sqrt(np.einsum("sii->si", D))
    R = D / (sds[:, :, None] * sds[:, None, :])
    R[:, np.arange(q), np.arange(q)] = 1.0
    L = np.transpose(np.linalg.cholesky(R), (0, 2, 1))   # upper, R = L'L
    return dict(betas=betas, sigmas=sig_all, bs=stacked("bs_gammas")[sl],
                gammas=stacked("gammas")[sl] * md.W_sds,
                alphas=stacked("alphas")[sl] * md.Wlong_sds, L=L, sds=sds), n_samples


# ------------------------------------------------------------------ new-patient data
def _time_grid(md, t_last, times):
    """Prediction times per patient: as predict_Event() in R (21 points by default)."""
    T_right = md.time
    t_max = np.quantile(T_right, 0.9)
    grids = []
    for t in t_last:
        if times is None:
            hi = T_right.max() if t > t_max else t_max
            grids.append(np.linspace(t, hi, 21))
        else:
            tt = np.sort(np.asarray(times, float))
            grids.append(np.concatenate([[t], tt[(tt > t) & (tt <= T_right.max())]]))
    return grids


class _NewData:
    """Design matrices for new patients, built with the fitted model's formulas, factor
    levels, spline knots and centring constants."""

    def __init__(self, md, newdata: pd.DataFrame, grids):
        meta = md.meta
        id_var, time_var = meta["id_var"], meta["time_var"]
        sv = meta["surv"]
        ids = list(dict.fromkeys(newdata[id_var]))
        rank = {g: i for i, g in enumerate(ids)}
        d = newdata.assign(_sid=newdata[id_var].map(rank).astype("int64"))
        d = d.sort_values(["_sid", time_var], kind="mergesort").reset_index(drop=True)
        self.ids, self.n = ids, len(ids)
        last = d.groupby("_sid", sort=True).tail(1).reset_index(drop=True)
        self.t_last = last[sv["time"]].astype(float).to_numpy()
        if md.causes is not None:
            if (last[sv["event"]].astype(str) != str(sv["censored"])).any():
                raise ValueError(f"set '{sv['event']}' to '{sv['censored']}' in newdata: "
                                 "predictions are for patients event-free up to their "
                                 "last time")
        elif (last[sv["event"]].astype(float) != 0).any():
            raise ValueError(f"set '{sv['event']}' to 0 in newdata: predictions are for "
                             "patients event-free up to their last time")
        # longitudinal measurements
        self.long = []
        for o in md.outcomes:
            vars_ = sorted({o.response.var, *o.fixed.variables(), *o.random.variables()})
            ok = d[vars_].notna().all(axis=1).to_numpy()
            dd = d.loc[ok].reset_index(drop=True)
            X = o.fixed.build(dd, o.levels)[0]
            Z = o.random.build(dd, o.levels)[0]
            self.long.append(dict(y=o.response.values(dd), X=X - o.Xbar, Z=Z,
                                  sid=dd["_sid"].to_numpy()))
        # survival: designs at the 15 Gauss-Kronrod nodes on [0, u] for every grid time u
        self.m = max(len(g) for g in grids)
        G = np.stack([np.pad(g, (0, self.m - len(g)), mode="edge") for g in grids])
        self.grid, self.grid_len = G, [len(g) for g in grids]
        if sv.get("start"):
            # delayed entry: integrate from each patient's entry time, so that sampling
            # the random effects conditions on being event-free from entry to t
            s0 = last[sv["start"]].astype(float).to_numpy()[:, None]
            if (s0[:, 0] >= self.t_last).any():
                raise ValueError(f"newdata: '{sv['start']}' must be before '{sv['time']}'")
            P = (G - s0) / 2.0
            nodes = P[..., None] * GK_SK + ((G + s0) / 2.0)[..., None]
        else:
            P = G / 2.0
            nodes = P[..., None] * (GK_SK + 1.0)                 # (n, m, 15)
        self.log_Pwk = np.log(P)[..., None] + np.log(GK_WK)
        deg = meta["Bsplines_degree"]
        if md.strata_levels is not None and md.causes is None:
            # strata: each patient's spline block, with that stratum's knots
            lev = [str(v) for v in md.strata_levels]
            vals = last[sv["strata"]].astype(str)
            bad = sorted(set(vals) - set(lev))
            if bad:
                raise ValueError(f"newdata: unknown '{sv['strata']}' values {bad} "
                                 f"(the model has {lev})")
            k_of = np.array([lev.index(v) for v in vals])
            nb0 = len(md.knots_strata[0]) - deg - 1
            self.W0 = np.zeros((self.n, self.m, 15, nb0 * len(lev)))
            for i, k in enumerate(k_of):
                self.W0[i, ..., k * nb0:(k + 1) * nb0] = _bspline_basis(
                    nodes[i].ravel(), md.knots_strata[k], deg).reshape(self.m, 15, -1)
        else:
            self.W0 = _bspline_basis(nodes.ravel(), md.knots, deg).reshape(self.n, self.m,
                                                                            15, -1)
        W_raw = sv["covariates"].without_intercept().build(last, dict(meta["surv_levels"]))[0]
        designer = TimeDesigner(fill_within_subject(d), time_var)
        self.designer = designer
        self.d = d
        self.cr = md.causes is not None
        if not self.cr:
            self.W = (W_raw - md.W_bar) / md.W_sds               # (n, P)
            Tn = nodes.reshape(self.n, -1)
            self.assoc = []
            for j, o in enumerate(md.outcomes):
                for a in o.assoc:
                    Xa = designer.design(a["form"], o.fixed, o.levels, Tn, "nodes", o.Xbar)
                    Za = designer.design(a["form"], o.random, o.levels, Tn, "nodes")
                    self.assoc.append(dict(j=j, X=Xa, Z=Za))
            return
        # competing risks: covariates of each cause's row, standardised as in the fit
        causes = md.causes
        Kc, p0 = len(causes), W_raw.shape[1]
        self.Wc = []
        for k in range(Kc):
            Wk = np.zeros((self.n, p0 * Kc))
            Wk[:, np.arange(p0) * Kc + k] = W_raw
            self.Wc.append((Wk - md.W_bar) / md.W_sds)
        # points: H(u) nodes on [0, u] (sampling), the 7-point rule on [t, u] for the CIF
        # integral, and 15 nodes on [0, s] for H(s) at each of those points
        t0 = G[:, :1]
        P7 = (G - t0) / 2.0
        s7 = P7[..., None] * GK7_SK + ((G + t0) / 2.0)[..., None]       # (n, m, 7)
        with np.errstate(divide="ignore"):
            self.log_w7 = np.log(P7)[..., None] + np.log(GK7_WK)        # -inf when u = t
        nodes_s = (s7 / 2.0)[..., None] * (GK_SK + 1.0)                 # (n, m, 7, 15)
        with np.errstate(divide="ignore"):
            self.log_Pwk_s = np.log(s7 / 2.0)[..., None] + np.log(GK_WK)
        self.sets = {}
        for key, pts in (("H", nodes), ("h7", s7), ("H7", nodes_s)):
            Tn = pts.reshape(self.n, -1)
            B = _bspline_basis(Tn.ravel(), md.knots, deg).reshape(*pts.shape, -1)
            des = {}
            for j, o in enumerate(md.outcomes):
                for form in dict.fromkeys(a["form"] for a in o.assoc):
                    des[(j, form)] = (
                        designer.design(form, o.fixed, o.levels, Tn, f"cr_{key}", o.Xbar),
                        designer.design(form, o.random, o.levels, Tn, f"cr_{key}"))
            self.sets[key] = dict(B=B, shape=pts.shape[1:], des=des)
        self.assoc = [dict(j=j, form=a["form"], cause=causes.index(a["cause"]))
                      for j, o in enumerate(md.outcomes) for a in o.assoc]


# ------------------------------------------------------------------ the engine
class _Engine:
    def __init__(self, fit, nd: _NewData, draws, device):
        md = fit.md
        self.md, self.nd = md, nd
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float64,  # noqa
                                      device=self.dev)
        self.t = t
        self.S = draws["betas"].shape[0]
        self.th = {k: t(v) for k, v in draws.items()}
        self.long = [dict(y=t(L["y"]), X=t(L["X"]), Z=t(L["Z"]),
                          sid=torch.as_tensor(np.array(L["sid"]), device=self.dev))
                     for L in nd.long]
        self.re = [torch.as_tensor(o.re_cols, device=self.dev) for o in md.outcomes]
        self.fe = [torch.as_tensor(o.fe_cols, device=self.dev) for o in md.outcomes]
        self.Wbar, self.Wisd = t(md.Wlong_bar), 1.0 / t(md.Wlong_sds)
        self.families = [o.family for o in md.outcomes]
        self.dfs = [o.df for o in md.outcomes]
        self.cr = nd.cr
        if self.cr:
            self.Kc = len(md.causes)
            self.nb = self.th["bs"].shape[1] // self.Kc
            self.Wg = [(t(W) @ self.th["gammas"].T).T for W in nd.Wc]    # (S, n) per cause
            self.sets = {k: dict(B=t(v["B"]), shape=v["shape"],
                                 des={jf: (t(X), t(Z)) for jf, (X, Z) in v["des"].items()})
                         for k, v in nd.sets.items()}
            self.log_Pwk = t(nd.log_Pwk)
            return
        self.assoc = [dict(j=a["j"], X=t(a["X"]), Z=t(a["Z"])) for a in nd.assoc]
        n, m = nd.n, nd.m
        # baseline part of the log hazard at every node: log P w + W0 bs + W gamma
        base = (t(nd.log_Pwk)[None] + torch.einsum("nmkb,sb->snmk", t(nd.W0), self.th["bs"])
                + (t(nd.W) @ self.th["gammas"].T).T[:, :, None, None])
        self.base = base                                         # (S, n, m, 15)

    def ll_long(self, b):
        S, n = self.S, self.nd.n
        out = torch.zeros(S, n, dtype=torch.float64, device=self.dev)
        for j, L in enumerate(self.long):
            bj = b[:, :, self.re[j]][:, L["sid"], :]
            eta = self.th["betas"][:, self.fe[j]] @ L["X"].T + (L["Z"] * bj).sum(-1)
            ll = fam.log_dens(self.families[j], L["y"], eta, self.th["sigmas"][:, j:j + 1],
                              self.dfs[j])
            out.index_add_(1, L["sid"], ll)
        return out

    def cumhaz(self, b, cols=None, r_sampling=False):
        """H(u) for all grid times u (or only the first, cols=slice(0, 1)).

        r_sampling=True reproduces how JMbayes2's simulate_REs() writes the survival
        part when it samples a new patient's random effects: reported association
        coefficients times standardised marker terms (each association divided by the
        marker's SD) and uncentred survival covariates. Used only to check agreement
        with R; see NOTES.md."""
        if self.cr:
            if r_sampling:
                raise NotImplementedError("r_sampling is not available for competing risks")
            return self.cumhaz_cr(b, cols)
        S, n, m = self.S, self.nd.n, self.nd.m
        cols = slice(None) if cols is None else cols
        lam = self.base[:, :, cols]
        if r_sampling:
            shift = (self.th["gammas"] * self.t(self.md.W_bar / self.md.W_sds)).sum(-1)
            lam = lam + shift[:, None, None, None]
        for c, a in enumerate(self.assoc):
            j = a["j"]
            X = a["X"].view(n, m, 15, -1)[:, cols]
            Z = a["Z"].view(n, m, 15, -1)[:, cols]
            eta = (torch.einsum("nmkp,sp->snmk", X, self.th["betas"][:, self.fe[j]])
                   + torch.einsum("nmkr,snr->snmk", Z, b[:, :, self.re[j]]))
            w = (eta - self.Wbar[c]) * self.Wisd[c]
            a_c = self.th["alphas"][:, c] * (self.Wisd[c] if r_sampling else 1.0)
            lam = lam + a_c[:, None, None, None] * w
        return torch.exp(lam).sum(-1)                            # (S, n, m')

    # ---- competing risks: cause-specific log hazards at a set of time points
    def _etas(self, b, key, sl=None):
        """Linear predictor of every (outcome, form) used in an association, at the
        points of set `key` (optionally only the grid columns sl), (S, n, *shape)."""
        st, n = self.sets[key], self.nd.n
        out = {}
        for (j, form), (X, Z) in st["des"].items():
            Xv = X.view(n, *st["shape"], -1)
            Zv = Z.view(n, *st["shape"], -1)
            if sl is not None:
                Xv, Zv = Xv[:, sl], Zv[:, sl]
            out[(j, form)] = (torch.einsum("n...p,sp->sn...", Xv, self.th["betas"][:, self.fe[j]])
                              + torch.einsum("n...r,snr->sn...", Zv, b[:, :, self.re[j]]))
        return out

    def _loghaz(self, etas, key, k, sl=None):
        """log hazard of cause k (standardised parametrisation, as in the sampler)."""
        st = self.sets[key]
        B = st["B"] if sl is None else st["B"][:, sl]
        bs_k = self.th["bs"][:, k * self.nb:(k + 1) * self.nb]
        lam = torch.einsum("n...b,sb->sn...", B, bs_k)
        lam = lam + self.Wg[k].reshape(*self.Wg[k].shape, *([1] * (lam.dim() - 2)))
        for c, a in enumerate(self.nd.assoc):
            eta = etas[(a["j"], a["form"])]
            w = ((eta if a["cause"] == k else 0.0 * eta) - self.Wbar[c]) * self.Wisd[c]
            lam = lam + self.th["alphas"][:, c].reshape(-1, *([1] * (w.dim() - 1))) * w
        return lam

    def cumhaz_cr(self, b, cols=None):
        """Total cumulative hazard over all causes at the grid times, (S, n, m')."""
        sl = slice(None) if cols is None else cols
        etas = self._etas(b, "H", sl)
        lw = self.log_Pwk[:, sl][None]
        return sum(torch.exp(lw + self._loghaz(etas, "H", k, sl)).sum(-1)
                   for k in range(self.Kc))

    def cif_cr(self, b):
        """CIF of each cause from the last time t to every grid time u:
        sum over the 7-point rule on [t, u] of w h_k(s) S(s), divided by S(t)
        (the competing-risk branch of predict_Event())."""
        H_t = self.cumhaz_cr(b, slice(0, 1))[..., 0]                    # (S, n)
        etas_s = self._etas(b, "H7")
        lw = self.t(self.nd.log_Pwk_s)[None]
        H_s = sum(torch.exp(lw + self._loghaz(etas_s, "H7", k)).sum(-1)
                  for k in range(self.Kc))                              # (S, n, m, 7)
        etas_h = self._etas(b, "h7")
        log_w = self.t(self.nd.log_w7)[None]
        out = []
        for k in range(self.Kc):
            f = log_w + self._loghaz(etas_h, "h7", k) - H_s + H_t[:, :, None, None]
            out.append(torch.exp(f).sum(-1))                            # (S, n, m)
        return out

    def ll_re(self, b):
        L, sds = self.th["L"], self.th["sds"]
        Linv = torch.linalg.solve_triangular(L, torch.eye(L.shape[-1], dtype=L.dtype,
                                                          device=self.dev).expand_as(L),
                                             upper=True)
        z = (b / sds[:, None, :]) @ Linv
        const = (-L.shape[-1] / 2 * LOG2PI
                 - torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
                 - torch.log(sds).sum(-1))
        return const[:, None] - 0.5 * (z * z).sum(-1)

    def sample_b(self, n_mcmc, gen, r_sampling=False):
        S, n, q = self.S, self.nd.n, self.md.q
        b = torch.zeros(S, n, q, dtype=torch.float64, device=self.dev)
        scale = torch.full_like(b, 0.2)
        first = slice(0, 1)
        den = self.ll_long(b) - self.cumhaz(b, first, r_sampling)[..., 0] + self.ll_re(b)
        for i in range(n_mcmc):
            for j in range(q):
                prop = b.clone()
                prop[:, :, j] += scale[:, :, j] * torch.randn(S, n, dtype=b.dtype,
                                                              device=self.dev, generator=gen)
                num = (self.ll_long(prop) - self.cumhaz(prop, first, r_sampling)[..., 0]
                       + self.ll_re(prop))
                u = torch.rand(S, n, dtype=b.dtype, device=self.dev, generator=gen)
                lr = num - den
                acc = torch.isfinite(lr) & (torch.exp(lr) > u)
                b = torch.where(acc[..., None], prop, b)
                den = torch.where(acc, num, den)
                if i > 9:                                        # robbins_monro(.., i)
                    sc = scale[:, :, j]
                    up = sc * (1.0 + 1.0 / (0.45 * i))
                    down = sc * (1.0 - 1.0 / (0.55 * i))
                    scale[:, :, j] = torch.where(acc, up, down)
        return b


def _normalize_cif(x):
    """normalize_CIF() in predict_Event(): make each curve non-decreasing."""
    x = x.copy()
    while True:
        dec = np.diff(x) < 0
        if not dec.any():
            return x
        ii = np.argmax(dec)
        x[ii + 1] = x[ii] + 1e-06


def _summaries(arr, level):
    lo, hi = (1 - level) / 2, (1 + level) / 2
    return (arr.mean(0), np.quantile(arr, lo, axis=0, method="linear"),
            np.quantile(arr, hi, axis=0, method="linear"))


@torch.no_grad()
def predict(fit, newdata: pd.DataFrame, process: str = "event", times=None,
            n_samples: int = 200, n_mcmc: int = 55, level: float = 0.95, seed: int = 123,
            device: str | None = None, type: str = "subject_specific",
            draws: str = "spread", r_sampling: bool = False) -> pd.DataFrame:
    md = fit.md
    sv = md.meta["surv"]
    from jmbayes import checks as ck
    extra = [sv[k] for k in ("start", "strata") if sv.get(k)]
    ck._missing_report(newdata, [md.meta["id_var"], md.meta["time_var"], sv["time"],
                                 sv["event"], *extra, *sv["covariates"].variables()],
                       "newdata")
    ck.check_visits(newdata, md.meta["id_var"], md.meta["time_var"],
                    [(o.response, [*o.fixed.variables(), *o.random.variables()], o.family)
                     for o in md.outcomes], "newdata", require_measured=False)
    draws, S = _posterior_draws(fit, n_samples, draws)
    last_times = (newdata.groupby(md.meta["id_var"], sort=False)[sv["time"]].last()
                  .astype(float).to_numpy())
    grids = _time_grid(md, last_times, times if process == "event" else None)
    nd = _NewData(md, newdata, grids)
    eng = _Engine(fit, nd, draws, device)
    gen = torch.Generator(device=eng.dev).manual_seed(seed)
    b = eng.sample_b(n_mcmc, gen, r_sampling)
    if process == "event" and md.causes is not None:
        cifs = [c.cpu().numpy() for c in eng.cif_cr(b)]
        rows = []
        for i in range(nd.n):
            k = nd.grid_len[i]
            for ci, cause in enumerate(md.causes):
                curves = np.stack([_normalize_cif(cifs[ci][s, i, :k]) for s in range(S)])
                pred, low, upp = _summaries(curves, level)
                for u, p_, l_, h_ in zip(nd.grid[i, :k], pred, low, upp):
                    rows.append({md.meta["id_var"]: nd.ids[i], "cause": cause,
                                 sv["time"]: u, "pred_CIF": p_, "low_CIF": l_,
                                 "upp_CIF": h_})
        return pd.DataFrame(rows)
    if process == "event":
        H = eng.cumhaz(b).cpu().numpy()                          # (S, n, m)
        cif = 1.0 - np.exp(-(H - H[:, :, :1]))
        rows = []
        for i in range(nd.n):
            k = nd.grid_len[i]
            curves = np.stack([_normalize_cif(cif[s, i, :k]) for s in range(S)])
            pred, low, upp = _summaries(curves, level)
            for u, p_, l_, h_ in zip(nd.grid[i, :k], pred, low, upp):
                rows.append({md.meta["id_var"]: nd.ids[i], sv["time"]: u,
                             "pred_CIF": p_, "low_CIF": l_, "upp_CIF": h_})
        return pd.DataFrame(rows)
    if process == "longitudinal":
        return _predict_long(md, nd, eng, b, times, level, type)
    raise ValueError("process must be 'event' or 'longitudinal'")


def _predict_long(md, nd, eng, b, times, level, type_):
    """Marker predictions at the rows of newdata, or at `times` per patient."""
    time_var, id_var = md.meta["time_var"], md.meta["id_var"]
    if times is None:
        T = None
        base = nd.d
    else:
        tt = np.asarray(times, float)
        T = np.tile(tt, (nd.n, 1))
        base = nd.designer.data_at(T)
    out = base[[id_var, time_var]].copy() if times is None else pd.DataFrame(
        {id_var: np.repeat(nd.ids, T.shape[1]), time_var: T.ravel()})
    sid = (base["_sid"].to_numpy() if times is None else np.repeat(np.arange(nd.n), T.shape[1]))
    for j, o in enumerate(md.outcomes):
        X = o.fixed.build(base, o.levels)[0]
        Z = o.random.build(base, o.levels)[0]
        X = eng.t(X - o.Xbar)
        Z = eng.t(Z)
        eta = eng.th["betas"][:, eng.fe[j]] @ X.T
        if type_ == "subject_specific":
            bj = b[:, :, eng.re[j]][:, torch.as_tensor(sid, device=eng.dev), :]
            eta = eta + (Z * bj).sum(-1)
        mu = fam.mean(o.family, eta)
        pred, low, upp = _summaries(mu.cpu().numpy(), level)
        out[f"pred_{o.name}"], out[f"low_{o.name}"], out[f"upp_{o.name}"] = pred, low, upp
    return out.reset_index(drop=True)
