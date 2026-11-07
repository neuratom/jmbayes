"""MCMC sampler: a PyTorch port of mcmc_cpp() in src/mcmc_fit.cpp.

Same algorithm as JMbayes2, step for step:
  per iteration -> bs_gammas (one-at-a-time MH), tau_bs_gammas (Gibbs),
  gammas (MH), alphas (MH), and from iteration 100 on: D (sds: log-normal MH,
  Cholesky entries: uniform MH), random effects (per-subject MH, one column at
  a time), fixed effects (hierarchical-centering Gibbs), sigmas (log-normal MH).
  Step sizes adapt with the same Robbins-Monro rule and start iterations.

What differs is only how the work is laid out:
  * all chains run together as a batch dimension (one tensor op advances every chain),
  * every update is a tensor operation over all subjects / quadrature points,
  * random-effect updates change trajectories incrementally instead of rebuilding them,
  * per-subject log-likelihoods are not stored per iteration; the fit statistics
    are accumulated on the fly (this is the RAM fix),
  * on a GPU, one whole iteration is recorded once as a CUDA graph and then
    replayed, so thousands of tiny kernel launches become one.

`replicate_quirks=True` (default) keeps two behaviours of the C++ code so the
results match JMbayes2: the stale log-density of the random effects after the
fixed-effects step, and the stale survival denominator after the tau update.
See NOTES.md.
"""
from __future__ import annotations

import math
import time as _time

import numpy as np
import torch

from jmbayes2py.model import ModelData

LOG2PI = math.log(2 * math.pi)
CONST_UNIF = 0.5 * math.sqrt(12.0)


def robbins_monro(scale, acc, itv, target=0.45):
    up = scale * (1.0 + 1.0 / (target * itv))
    down = scale * (1.0 - 1.0 / ((1.0 - target) * itv))
    return torch.where(acc, up, down)


def r_jitter(x: np.ndarray, rng: np.random.Generator, factor=2.0) -> np.ndarray:
    """R's jitter(x, factor) (used by jm_fit() to start each chain)."""
    x = np.asarray(x, float)
    flat = x.ravel()
    z = np.ptp(flat)
    if z == 0:
        z = abs(flat.min())
    if z == 0:
        z = 1.0
    digits = int(3 - np.floor(np.log10(z)))
    xx = np.unique(np.round(flat, digits))
    d = np.diff(xx)
    if len(d):
        d = d.min()
    else:
        d = xx[0] / 10 if xx[0] != 0 else z / 10
    amount = factor / 5 * abs(d)
    return (flat + rng.uniform(-amount, amount, flat.size)).reshape(x.shape)


class JointModelSampler:
    def __init__(self, md: ModelData, n_chains=3, device=None, seed=123,
                 replicate_quirks=True, dtype=torch.float64, cuda_graphs=True,
                 cpu_threads=1):
        self.md = md
        self.C = n_chains
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.dev.type == "cpu" and cpu_threads:
            torch.set_num_threads(cpu_threads)   # tiny ops: threading costs more than it saves
        self.dt = dtype
        self.seed = seed
        self.quirks = replicate_quirks
        self.use_graphs = cuda_graphs and self.dev.type == "cuda"
        t = lambda a: torch.tensor(np.asarray(a), dtype=dtype, device=self.dev)  # noqa: E731
        ti = lambda a: torch.tensor(np.asarray(a), dtype=torch.long, device=self.dev)  # noqa: E731
        self.t, self.ti = t, ti
        self.n, self.q, self.K = md.n, md.q, md.GK_k
        self.W0_H, self.W0_h = t(md.W0_H), t(md.W0_h)
        self.W_H, self.W_h = t(md.W_H), t(md.W_h)
        self.log_Pwk = t(md.log_Pwk)
        self.event = t(md.event)
        self.sid_H = ti(np.repeat(np.arange(md.n), md.GK_k))
        self.Wlong_bar = t(md.Wlong_bar)
        self.Wlong_isd = 1.0 / t(md.Wlong_sds)
        self.outs = []
        for o in md.outcomes:
            self.outs.append(dict(
                y=t(o.y), X=t(o.X), Z=t(o.Z), sid=ti(o.sid), fam=o.family,
                re=ti(o.re_cols), fe=ti(o.fe_cols), re_py=[int(c) for c in o.re_cols],
                has_sigma=o.has_sigma,
                # fixed effects outside the hierarchical centering (Metropolis-Hastings)
                nhc=[int(c) for c in o.nhc], nhc_glob=[int(o.fe_cols[c]) for c in o.nhc],
                nhc_idx=ti(np.asarray(o.fe_cols)[o.nhc])))
        self.n_out = len(self.outs)
        # association columns (one per value()/slope()/area() term), each tied to an outcome
        self.assoc = []
        for j, o in enumerate(md.outcomes):
            for a in o.assoc:
                self.assoc.append(dict(j=j, form=a["form"], X_H=t(a["X_H"]), Z_H=t(a["Z_H"]),
                                       X_h=t(a["X_h"]), Z_h=t(a["Z_h"])))
        self.A = len(self.assoc)
        self.cols_of = [[c for c, a in enumerate(self.assoc) if a["j"] == j]
                        for j in range(self.n_out)]
        self.X_dot = t(md.X_dot)
        self.ind_FE_HC = ti(md.ind_FE_HC)
        self.patterns = [(ti(s), ti(r), ti(f)) for s, r, f in md.patterns]
        self.up = [(int(r), int(c)) for r, c in md.upper_part]
        self.zero = [(int(r), int(c)) for r, c in md.ind_zero_D]
        free = np.zeros((md.q, md.q))
        for r, c in self.up:
            free[r, c] = 1.0
        self.free_mask = t(free)
        self.eye_q = torch.eye(self.q, dtype=dtype, device=self.dev)
        pr = md.priors
        self.pr = {k: (t(v) if isinstance(v, np.ndarray) else v) for k, v in pr.items()}
        self.m_nhc = [t(m) for m in pr.get("mean_betas_nHC", [np.zeros(0)] * self.n_out)]
        self.T_nhc = [t(T) for T in pr.get("Tau_betas_nHC", [np.zeros((0, 0))] * self.n_out)]
        self.Tau_mean_HC = self.pr["Tau_betas_HC"] @ self.pr["mean_betas_HC"]
        i = np.arange(1, self.q)
        self.lkj_coef = t(self.q - i - 1.0 + 2.0 * pr["D_L_etaLKJ"] - 2.0)

    # ------------------------------------------------------------------ pieces
    def _rand(self, *shape):
        return torch.randn(*shape, dtype=self.dt, device=self.dev)

    def _unif(self, *shape):
        return torch.rand(*shape, dtype=self.dt, device=self.dev)

    def eta_long(self, j, betas, b):
        o = self.outs[j]
        bj = b[:, :, o["re"]][:, o["sid"], :]
        return betas[:, o["fe"]] @ o["X"].T + (o["Z"] * bj).sum(-1)

    def ll_long_obs(self, j, eta, sigma):
        o = self.outs[j]
        y = o["y"]
        if o["fam"] == "gaussian":
            s = sigma[:, None]
            return -torch.log(s) - 0.5 * LOG2PI - (y - eta) ** 2 / (2 * s * s)
        mu = 1.0 / (1.0 + torch.exp(-eta))                         # binomial, logit link
        return torch.log(torch.where(y > 0, mu, 1.0 - mu))

    def ll_long_subj(self, j, eta, sigma):
        o = self.outs[j]
        out = torch.zeros(eta.shape[0], self.n, dtype=self.dt, device=self.dev)
        return out.index_add_(1, o["sid"], self.ll_long_obs(j, eta, sigma))

    def eta_surv(self, c, betas, b, where):
        """Linear predictor of association column c at quadrature points (H) or event
        times (h)."""
        a = self.assoc[c]
        o = self.outs[a["j"]]
        bj = b[:, :, o["re"]]
        if where == "H":
            X, Z = a["X_H"], a["Z_H"]
            bj = bj[:, self.sid_H, :]
        else:
            X, Z = a["X_h"], a["Z_h"]
        return betas[:, o["fe"]] @ X.T + (Z * bj).sum(-1)

    def wlong_col(self, c, eta):
        return (eta - self.Wlong_bar[c]) * self.Wlong_isd[c]

    def refresh_assoc(self, s, betas, b):
        for c in range(self.A):
            s[f"etaH{c}"] = self.eta_surv(c, betas, b, "H").contiguous()
            s[f"etah{c}"] = self.eta_surv(c, betas, b, "h").contiguous()
        s["WlH"] = torch.stack([self.wlong_col(c, s[f"etaH{c}"]) for c in range(self.A)], -1)
        s["Wlh"] = torch.stack([self.wlong_col(c, s[f"etah{c}"]) for c in range(self.A)], -1)
        s["WlH_a"] = (s["WlH"] * s["alphas"][:, None, :]).sum(-1)
        s["Wlh_a"] = (s["Wlh"] * s["alphas"][:, None, :]).sum(-1)

    def log_surv(self, W0H_bs, WH_g, WlH_a, W0h_bs, Wh_g, Wlh_a):
        lam = torch.exp(self.log_Pwk + W0H_bs + WH_g + WlH_a)
        H = lam.view(lam.shape[0], self.n, self.K).sum(-1)
        return -H + self.event * (W0h_bs + Wh_g + Wlh_a)

    def re_const(self, L, sds):
        return -self.q / 2 * LOG2PI - torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1) \
            - torch.log(sds).sum(-1)

    def tri_inv(self, L):
        return torch.linalg.solve_triangular(L, self.eye_q.expand_as(L), upper=True)

    def log_re(self, b, L, sds):
        z = (b / sds[:, None, :]) @ self.tri_inv(L)               # rows: x L^{-1}
        return self.re_const(L, sds)[:, None] - 0.5 * (z * z).sum(-1)

    def quad(self, x, mean, Tau):
        z = x - mean
        return ((z @ Tau) * z).sum(-1)

    def log_prior_surv(self, bs, gammas, alphas, tau_bs):
        pr = self.pr
        return (-0.5 * tau_bs * self.quad(bs, pr["mean_bs_gammas"], pr["Tau_bs_gammas"])
                - 0.5 * self.quad(gammas, pr["mean_gammas"], pr["Tau_gammas"])
                - 0.5 * self.quad(alphas, pr["mean_alphas"], pr["Tau_alphas"]))

    def lkj(self, L):
        d = torch.diagonal(L, dim1=-2, dim2=-1)[:, 1:]
        return (self.lkj_coef * torch.log(d)).sum(-1)

    @staticmethod
    def log_dgamma(x, shape, scale):
        return -math.lgamma(shape) - shape * torch.log(scale) + (shape - 1) * torch.log(x) - x / scale

    @staticmethod
    def log_dlnorm(x, mu, sigma):
        lx = torch.log(x)
        return -torch.log(sigma) - 0.5 * LOG2PI - lx - (lx - mu) ** 2 / (2 * sigma * sigma)

    def accept(self, log_ratio):
        u = self._unif(*log_ratio.shape)
        return torch.isfinite(log_ratio) & (torch.log(u) < log_ratio)

    def surv(self, s, **over):
        g = lambda k: over.get(k, s[k])  # noqa: E731
        return self.log_surv(g("W0H_bs"), g("WH_g"), g("WlH_a"), g("W0h_bs"), g("Wh_g"),
                             g("Wlh_a"))

    # ------------------------------------------------------------- state
    def init_state(self, n_iter):
        md, C, dt, dev = self.md, self.C, self.dt, self.dev
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        ini = md.inits
        jit = lambda x: np.stack([r_jitter(x, rng) for _ in range(C)])  # noqa: E731
        s = dict(betas=self.t(jit(ini["betas"])), b=self.t(jit(ini["b"])),
                 bs=self.t(jit(ini["bs_gammas"])), gammas=self.t(jit(ini["gammas"])),
                 alphas=self.t(jit(ini["alphas"])),
                 tau=self.t(jit(ini["tau_bs_gammas"]))[:, 0].contiguous())
        s["sigmas"] = torch.exp(self.t(jit(ini["log_sigmas"])))
        D = ini["D"]
        sds = np.sqrt(np.diag(D))
        L = np.linalg.cholesky(D / np.outer(sds, sds)).T             # upper, R = L'L
        s["sds"] = self.t(np.tile(sds, (C, 1)))
        s["L"] = self.t(np.tile(L, (C, 1, 1)))
        s["W0H_bs"] = s["bs"] @ self.W0_H.T
        s["W0h_bs"] = s["bs"] @ self.W0_h.T
        s["WH_g"] = s["gammas"] @ self.W_H.T
        s["Wh_g"] = s["gammas"] @ self.W_h.T
        # association design from the unjittered initial values, like JMbayes2
        b0 = self.t(ini["b"])[None].expand(C, -1, -1)
        be0 = self.t(ini["betas"])[None].expand(C, -1)
        self.refresh_assoc(s, be0, b0)
        s["ll_surv"] = self.surv(s)
        s["den_surv"] = s["ll_surv"].sum(-1) + self.log_prior_surv(s["bs"], s["gammas"],
                                                                   s["alphas"], s["tau"])
        s["ll_re"] = self.log_re(s["b"], s["L"], s["sds"])
        for j in range(self.n_out):
            s[f"eta{j}"] = self.eta_long(j, s["betas"], s["b"])
        self.refresh_long(s)
        f = lambda k: torch.full((C, k), 0.1, dtype=dt, device=dev)  # noqa: E731
        s["sc_bs"], s["sc_g"], s["sc_a"] = f(s["bs"].shape[1]), f(s["gammas"].shape[1]), \
            f(self.A)
        s["sc_sds"], s["sc_L"], s["sc_sig"] = f(self.q), f(len(self.up)), f(self.n_out)
        s["sc_b"] = torch.full((C, self.n, self.q), 0.1, dtype=dt, device=dev)
        for j, o in enumerate(self.outs):
            if o["nhc"]:
                s[f"sc_beta{j}"] = f(len(o["nhc"]))
        # outputs (small: iterations x chains x parameters) and running summaries
        z = lambda *sh: torch.zeros(*sh, dtype=dt, device=dev)  # noqa: E731
        p = s["betas"].shape[1]
        for k, w in dict(r_bs=s["bs"].shape[1], r_tau=1, r_gammas=s["gammas"].shape[1],
                         r_alphas=self.A, r_sds=self.q, r_L=len(self.up), r_betas=p,
                         r_sigmas=self.n_out).items():
            s[k] = z(n_iter, C, w)
        s["acc_b"], s["sum_b"] = z(C, self.n, self.q), z(C, self.n, self.q)
        s["n_keep"], s["sum_tot"] = z(()), z(C)
        s["st_mean"], s["st_m2"] = z(C, self.n), z(C, self.n)
        s["lse_pos"] = torch.full((C, self.n), -math.inf, dtype=dt, device=dev)
        s["lse_neg"] = torch.full((C, self.n), -math.inf, dtype=dt, device=dev)
        s["it"] = torch.zeros(1, dtype=torch.long, device=dev)
        return s

    def refresh_long(self, s):
        tot = 0
        for j in range(self.n_out):
            s[f"llp{j}"] = self.ll_long_subj(j, s[f"eta{j}"], s["sigmas"][:, j])
            tot = tot + s[f"llp{j}"]
        s["ll_long"] = tot

    def record(self, s, key, value):
        s[key].index_copy_(0, s["it"], value[None])

    def assoc_shift(self, s, j, shifts):
        """Proposed association columns of outcome j when its linear predictors move by
        shifts(a) -> (shift at quadrature points, shift at event times)."""
        prop = dict(WlH_a=s["WlH_a"], Wlh_a=s["Wlh_a"], cols={})
        for c in self.cols_of[j]:
            dH, dh = shifts(self.assoc[c])
            eH, eh = s[f"etaH{c}"] + dH, s[f"etah{c}"] + dh
            wH, wh = self.wlong_col(c, eH), self.wlong_col(c, eh)
            ac = s["alphas"][:, c:c + 1]
            prop["WlH_a"] = prop["WlH_a"] + ac * (wH - s["WlH"][..., c])
            prop["Wlh_a"] = prop["Wlh_a"] + ac * (wh - s["Wlh"][..., c])
            prop["cols"][c] = (eH, eh, wH, wh)
        return prop

    def assoc_accept(self, s, prop, accH, acch):
        """Keep the proposed association columns where accepted."""
        for c, (eH, eh, wH, wh) in prop["cols"].items():
            s[f"etaH{c}"] = torch.where(accH, eH, s[f"etaH{c}"])
            s[f"etah{c}"] = torch.where(acch, eh, s[f"etah{c}"])
            s["WlH"][..., c] = torch.where(accH, wH, s["WlH"][..., c])
            s["Wlh"][..., c] = torch.where(acch, wh, s["Wlh"][..., c])
        s["WlH_a"] = torch.where(accH, prop["WlH_a"], s["WlH_a"])
        s["Wlh_a"] = torch.where(acch, prop["Wlh_a"], s["Wlh_a"])

    # ------------------------------------------------------------- updates
    def update_vector(self, s, name, sc, DH, Dh, keyH, keyh, itf):
        adapt = itf > 19
        for i in range(s[name].shape[1]):
            diff = s[sc][:, i] * self._rand(self.C)
            prop = s[name].clone()
            prop[:, i] += diff
            pH = s[keyH] + DH[:, i] * diff[:, None]
            ph = s[keyh] + Dh[:, i] * diff[:, None]
            ll = self.surv(s, **{keyH: pH, keyh: ph})
            pri = dict(bs=s["bs"], gammas=s["gammas"], alphas=s["alphas"])
            pri[name] = prop
            num = ll.sum(-1) + self.log_prior_surv(pri["bs"], pri["gammas"], pri["alphas"],
                                                    s["tau"])
            acc = self.accept(num - s["den_surv"])
            a = acc[:, None]
            s[name] = torch.where(a, prop, s[name])
            s[keyH] = torch.where(a, pH, s[keyH])
            s[keyh] = torch.where(a, ph, s[keyh])
            s["ll_surv"] = torch.where(a, ll, s["ll_surv"])
            s["den_surv"] = torch.where(acc, num, s["den_surv"])
            s[sc][:, i] = torch.where(adapt, robbins_monro(s[sc][:, i], acc, itf), s[sc][:, i])

    def update_alphas(self, s, itf):
        adapt = itf > 19
        for i in range(self.A):
            diff = s["sc_a"][:, i] * self._rand(self.C)
            prop = s["alphas"].clone()
            prop[:, i] += diff
            pH = s["WlH_a"] + s["WlH"][..., i] * diff[:, None]
            ph = s["Wlh_a"] + s["Wlh"][..., i] * diff[:, None]
            ll = self.surv(s, WlH_a=pH, Wlh_a=ph)
            num = ll.sum(-1) + self.log_prior_surv(s["bs"], s["gammas"], prop, s["tau"])
            acc = self.accept(num - s["den_surv"])
            a = acc[:, None]
            s["alphas"] = torch.where(a, prop, s["alphas"])
            s["WlH_a"] = torch.where(a, pH, s["WlH_a"])
            s["Wlh_a"] = torch.where(a, ph, s["Wlh_a"])
            s["ll_surv"] = torch.where(a, ll, s["ll_surv"])
            s["den_surv"] = torch.where(acc, num, s["den_surv"])
            s["sc_a"][:, i] = torch.where(adapt, robbins_monro(s["sc_a"][:, i], acc, itf),
                                          s["sc_a"][:, i])

    def update_tau(self, s):
        pr = self.pr
        quad = self.quad(s["bs"], pr["mean_bs_gammas"] * 0, pr["Tau_bs_gammas"])
        shape = pr["A_tau_bs_gammas"] + 0.5 * pr["rank_Tau_bs_gammas"]
        rate = pr["B_tau_bs_gammas"] + 0.5 * quad
        s["tau"] = torch._standard_gamma(torch.full_like(rate, shape)) / rate
        if not self.quirks:
            s["den_surv"] = s["ll_surv"].sum(-1) + self.log_prior_surv(
                s["bs"], s["gammas"], s["alphas"], s["tau"])

    def complete_L(self, prop):
        """Fill the diagonal and the zero-correlation entries (propose_L in JMbayes2_D.h)."""
        prop = prop + torch.diag_embed(torch.sqrt(1.0 - (prop * prop).sum(-2)))
        for j0, j1 in self.zero:
            prop = prop.clone()
            prop[:, j0, j1] = -(prop[:, :, j0] * prop[:, :, j1]).sum(-1) / prop[:, j0, j0]
            prop[:, j1, j1] = torch.sqrt(1.0 - (prop[:, :j1, j1] ** 2).sum(-1))
        return prop

    def update_D(self, s, itf):
        pr = self.pr
        adapt = itf > 119
        it100 = itf - 100
        shape, mean = pr["D_sds_shape"], pr["D_sds_mean"]
        prior = lambda sd: self.log_dgamma(sd, shape, mean / shape).sum(-1)  # noqa: E731
        b = s["b"]
        # sds, one at a time: z = b diag(1/sds) L^{-1}; changing sds_i rescales row i
        Linv = self.tri_inv(s["L"])
        M = Linv / s["sds"][:, :, None]
        z = b @ M
        den = s["ll_re"].sum(-1) + prior(s["sds"])
        for i in range(self.q):
            sc = s["sc_sds"][:, i]
            cur = s["sds"][:, i]
            mu_cur = torch.log(cur) - 0.5 * sc * sc
            new = torch.exp(mu_cur + sc * self._rand(self.C))
            prop = s["sds"].clone()
            prop[:, i] = new
            dM = M[:, i, :] * (cur / new - 1.0)[:, None]
            zp = z + b[:, :, i:i + 1] * dM[:, None, :]
            llp = self.re_const(s["L"], prop)[:, None] - 0.5 * (zp * zp).sum(-1)
            num = llp.sum(-1) + prior(prop)
            mu_new = torch.log(new) - 0.5 * sc * sc
            lr = num - den + self.log_dlnorm(cur, mu_new, sc) - self.log_dlnorm(new, mu_cur, sc)
            acc = self.accept(lr)
            a = acc[:, None]
            s["sds"] = torch.where(a, prop, s["sds"])
            s["ll_re"] = torch.where(a, llp, s["ll_re"])
            M = M.clone()
            M[:, i, :] = torch.where(a, M[:, i, :] + dM, M[:, i, :])
            z = torch.where(acc[:, None, None], zp, z)
            den = torch.where(acc, num, den)
            s["sc_sds"][:, i] = torch.where(adapt, robbins_monro(sc, acc, it100), sc)
        # Cholesky factor of the correlation matrix, one free entry at a time
        den = s["ll_re"].sum(-1) + self.lkj(s["L"])
        x = b / s["sds"][:, None, :]
        const_sds = -self.q / 2 * LOG2PI - torch.log(s["sds"]).sum(-1)
        for i, (r, c) in enumerate(self.up):
            rr = CONST_UNIF * s["sc_L"][:, i]
            cur = s["L"][:, r, c]
            new = cur - rr + 2 * rr * self._unif(self.C)
            prop = s["L"] * self.free_mask              # free upper entries only
            prop[:, r, c] = new
            prop = self.complete_L(prop)
            finite = torch.isfinite(prop).flatten(1).all(-1)
            safe = torch.where(finite[:, None, None], prop, s["L"])
            zp = x @ self.tri_inv(safe)
            dg = torch.diagonal(safe, dim1=-2, dim2=-1)
            llp = (const_sds - torch.log(dg).sum(-1))[:, None] - 0.5 * (zp * zp).sum(-1)
            num = llp.sum(-1) + self.lkj(safe)
            lr = torch.where(finite, num - den, torch.full_like(num, math.nan))
            acc = self.accept(lr)
            s["L"] = torch.where(acc[:, None, None], prop, s["L"])
            s["ll_re"] = torch.where(acc[:, None], llp, s["ll_re"])
            den = torch.where(acc, num, den)
            s["sc_L"][:, i] = torch.where(adapt, robbins_monro(s["sc_L"][:, i], acc, it100),
                                          s["sc_L"][:, i])
        self.record(s, "r_sds", s["sds"])
        if self.up:
            self.record(s, "r_L", torch.stack([s["L"][:, r, c] for r, c in self.up], -1))

    def update_b(self, s, itf, keep):
        adapt = itf > 119
        it100 = itf - 100
        den = s["ll_long"] + s["ll_surv"] + s["ll_re"]
        M = self.tri_inv(s["L"]) / s["sds"][:, :, None]             # z = b M
        z = s["b"] @ M
        const = self.re_const(s["L"], s["sds"])[:, None]
        for j, o in enumerate(self.outs):
            for k, col in enumerate(o["re_py"]):
                delta = s["sc_b"][:, :, col] * self._rand(self.C, self.n)
                newcol = s["b"][:, :, col] + delta
                eta_p = s[f"eta{j}"] + o["Z"][:, k] * delta[:, o["sid"]]
                part_p = self.ll_long_subj(j, eta_p, s["sigmas"][:, j])
                ll_long_p = s["ll_long"] - s[f"llp{j}"] + part_p
                deltaH = delta[:, self.sid_H]
                prop = self.assoc_shift(s, j, lambda a, k=k, dH=deltaH, d=delta: (
                    a["Z_H"][:, k] * dH, a["Z_h"][:, k] * d))
                WlH_a_p, Wlh_a_p = prop["WlH_a"], prop["Wlh_a"]
                ll_surv_p = self.surv(s, WlH_a=WlH_a_p, Wlh_a=Wlh_a_p)
                zp = z + delta[:, :, None] * M[:, col, None, :]
                ll_re_p = const - 0.5 * (zp * zp).sum(-1)
                num = ll_long_p + ll_surv_p + ll_re_p
                acc = self.accept(num - den)                         # (C, n)
                accH = acc[:, self.sid_H]
                s["b"][:, :, col] = torch.where(acc, newcol, s["b"][:, :, col])
                z = torch.where(acc[:, :, None], zp, z)
                den = torch.where(acc, num, den)
                s[f"llp{j}"] = torch.where(acc, part_p, s[f"llp{j}"])
                s["ll_long"] = torch.where(acc, ll_long_p, s["ll_long"])
                s["ll_surv"] = torch.where(acc, ll_surv_p, s["ll_surv"])
                s["ll_re"] = torch.where(acc, ll_re_p, s["ll_re"])
                s[f"eta{j}"] = torch.where(acc[:, o["sid"]], eta_p, s[f"eta{j}"])
                self.assoc_accept(s, prop, accH, acc)
                s["acc_b"][:, :, col] += (acc & keep).to(self.dt)
                sc = s["sc_b"][:, :, col]
                s["sc_b"][:, :, col] = torch.where(adapt, robbins_monro(sc, acc, it100), sc)

    def update_betas(self, s, itf):
        C = self.C
        bHC = s["betas"][:, self.ind_FE_HC]
        u = s["b"] + torch.einsum("nrp,cp->cnr", self.X_dot, bHC)
        U = s["L"] * s["sds"][:, None, :]                          # D = U'U
        D = U.transpose(-1, -2) @ U
        p = len(self.ind_FE_HC)
        XtX = torch.zeros(C, p, p, dtype=self.dt, device=self.dev)
        Xtu = torch.zeros(C, p, dtype=self.dt, device=self.dev)
        for subj, re_idx, fe_idx in self.patterns:
            Dp = D[:, re_idx][:, :, re_idx]
            Rt = torch.linalg.cholesky_ex(Dp).L                     # lower = R'
            Rti = torch.linalg.solve_triangular(
                Rt, torch.eye(len(re_idx), dtype=self.dt, device=self.dev).expand_as(Rt),
                upper=False)
            Xd = self.X_dot[subj][:, re_idx][:, :, fe_idx]           # (m, k, p_i)
            ui = u[:, subj][:, :, re_idx]                            # (C, m, k)
            Xt = torch.einsum("cab,mbp->cmap", Rti, Xd)
            ut = torch.einsum("cab,cmb->cma", Rti, ui)
            Xtu[:, fe_idx] += torch.einsum("cmap,cma->cp", Xt, ut)
            XtX[:, fe_idx[:, None], fe_idx[None, :]] += torch.einsum("cmap,cmaq->cpq", Xt, Xt)
        Q = self.pr["Tau_betas_HC"] + XtX
        bv = self.Tau_mean_HC + Xtu
        Lp = torch.linalg.cholesky_ex(Q).L
        yy = torch.linalg.solve_triangular(Lp, bv[..., None], upper=False)[..., 0]
        yy = yy + self._rand(C, p)
        new = torch.linalg.solve_triangular(Lp.transpose(-1, -2), yy[..., None],
                                            upper=True)[..., 0]
        s["betas"][:, self.ind_FE_HC] = new
        s["b"] = u - torch.einsum("nrp,cp->cnr", self.X_dot, new)
        for j in range(self.n_out):
            s[f"eta{j}"] = self.eta_long(j, s["betas"], s["b"])
        self.refresh_assoc(s, s["betas"], s["b"])
        s["ll_surv"] = self.surv(s)
        self.refresh_long(s)
        if not self.quirks:
            s["ll_re"] = self.log_re(s["b"], s["L"], s["sds"])
        self.update_betas_nhc(s, itf)
        self.record(s, "r_betas", s["betas"])

    def update_betas_nhc(self, s, itf):
        """Fixed effects outside the hierarchical centering: one-at-a-time random-walk
        Metropolis-Hastings, as in the second half of update_betas() (JMbayes2_FE.h)."""
        if not any(o["nhc"] for o in self.outs):
            return
        adapt = itf > 119
        for j, o in enumerate(self.outs):
            if not o["nhc"]:
                continue
            sig = s["sigmas"][:, j]
            sc = f"sc_beta{j}"
            prior = lambda bv: -0.5 * self.quad(bv, self.m_nhc[j], self.T_nhc[j])  # noqa: E731
            den = (self.ll_long_obs(j, s[f"eta{j}"], sig).sum(-1) + s["ll_surv"].sum(-1)
                   + prior(s["betas"][:, o["nhc_idx"]]))
            for i, (loc, glob) in enumerate(zip(o["nhc"], o["nhc_glob"])):
                diff = s[sc][:, i] * self._rand(self.C)
                prop = s["betas"].clone()
                prop[:, glob] += diff
                d = diff[:, None]
                eta_p = s[f"eta{j}"] + d * o["X"][:, loc]
                prop_a = self.assoc_shift(s, j, lambda a, loc=loc, d=d: (
                    d * a["X_H"][:, loc], d * a["X_h"][:, loc]))
                WlH_a_p, Wlh_a_p = prop_a["WlH_a"], prop_a["Wlh_a"]
                ll_surv_p = self.surv(s, WlH_a=WlH_a_p, Wlh_a=Wlh_a_p)
                num = (self.ll_long_obs(j, eta_p, sig).sum(-1) + ll_surv_p.sum(-1)
                       + prior(prop[:, o["nhc_idx"]]))
                acc = self.accept(num - den)
                a = acc[:, None]
                s["betas"] = torch.where(a, prop, s["betas"])
                s[f"eta{j}"] = torch.where(a, eta_p, s[f"eta{j}"])
                self.assoc_accept(s, prop_a, a, a)
                s["ll_surv"] = torch.where(a, ll_surv_p, s["ll_surv"])
                den = torch.where(acc, num, den)
                cur = s[sc][:, i]
                s[sc][:, i] = torch.where(adapt, robbins_monro(cur, acc, itf - 100), cur)
        self.refresh_long(s)

    def update_sigmas(self, s, itf):
        pr = self.pr
        adapt = itf > 119
        for j, o in enumerate(self.outs):
            if not o["has_sigma"]:
                continue
            sc = s["sc_sig"][:, j]
            cur = s["sigmas"][:, j]
            scale_pr = pr["sigmas_mean"][j] / pr["sigmas_shape"]
            den = self.ll_long_obs(j, s[f"eta{j}"], cur).sum(-1) + \
                self.log_dgamma(cur, pr["sigmas_shape"], scale_pr)
            mu_cur = torch.log(cur) - 0.5 * sc * sc
            new = torch.exp(mu_cur + sc * self._rand(self.C))
            num = self.ll_long_obs(j, s[f"eta{j}"], new).sum(-1) + \
                self.log_dgamma(new, pr["sigmas_shape"], scale_pr)
            mu_new = torch.log(new) - 0.5 * sc * sc
            lr = num - den + self.log_dlnorm(cur, mu_new, sc) - self.log_dlnorm(new, mu_cur, sc)
            acc = self.accept(lr)
            s["sigmas"][:, j] = torch.where(acc, new, cur)
            s["sc_sig"][:, j] = torch.where(adapt, robbins_monro(sc, acc, itf - 100), sc)
        self.record(s, "r_sigmas", s["sigmas"])

    def accumulate(self, s, keep):
        """Running fit statistics and random-effect means (after burn-in)."""
        k = keep.to(self.dt)
        ll = s["ll_long"] + s["ll_surv"] + s["ll_re"]
        s["n_keep"] = s["n_keep"] + k
        s["sum_tot"] = s["sum_tot"] + k * ll.sum(-1)
        d = ll - s["st_mean"]
        s["st_mean"] = s["st_mean"] + k * d / torch.clamp(s["n_keep"], min=1.0)
        s["st_m2"] = s["st_m2"] + k * d * (ll - s["st_mean"])
        s["lse_pos"] = torch.where(keep, torch.logaddexp(s["lse_pos"], ll), s["lse_pos"])
        s["lse_neg"] = torch.where(keep, torch.logaddexp(s["lse_neg"], -ll), s["lse_neg"])

    # ------------------------------------------------------------- one iteration
    def step(self, s, full, n_burnin):
        itf = s["it"][0].to(self.dt)
        keep = itf > (n_burnin - 1)
        self.update_vector(s, "bs", "sc_bs", self.W0_H, self.W0_h, "W0H_bs", "W0h_bs", itf)
        self.record(s, "r_bs", s["bs"])
        self.update_tau(s)
        self.record(s, "r_tau", s["tau"][:, None])
        if s["gammas"].shape[1]:
            self.update_vector(s, "gammas", "sc_g", self.W_H, self.W_h, "WH_g", "Wh_g", itf)
            self.record(s, "r_gammas", s["gammas"])
        self.update_alphas(s, itf)
        self.record(s, "r_alphas", s["alphas"])
        if full:
            self.update_D(s, itf)
            self.update_b(s, itf, keep)
            self.update_betas(s, itf)
            s["den_surv"] = s["ll_surv"].sum(-1) + self.log_prior_surv(
                s["bs"], s["gammas"], s["alphas"], s["tau"])
            self.update_sigmas(s, itf)
            self.refresh_long(s)
            s["sum_b"] = s["sum_b"] + keep.to(self.dt) * s["b"]
        self.accumulate(s, keep)

    # ------------------------------------------------------------- CUDA graphs
    def _capture(self, s, n_burnin):
        """Record one iteration of each phase as a CUDA graph (state stays in place)."""
        keys = list(s.keys())
        snap = {k: v.clone() for k, v in s.items()}
        rng = torch.cuda.get_rng_state()

        def body(full):
            work = dict(s)
            self.step(work, full, n_burnin)
            for k in keys:
                if work[k] is not s[k]:
                    s[k].copy_(work[k])

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):                       # warm-up (cuBLAS/cuSOLVER handles)
                body(False)
                body(True)
        torch.cuda.current_stream().wait_stream(side)
        graphs = {}
        for full in (False, True):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                body(full)
            graphs[full] = g
        for k in keys:                               # undo the warm-up
            s[k].copy_(snap[k])
        torch.cuda.set_rng_state(rng)
        return graphs

    # ------------------------------------------------------------- run
    @torch.no_grad()
    def run(self, n_iter=3500, n_burnin=500, progress=True):
        s = self.init_state(n_iter)
        t0 = _time.perf_counter()
        graphs = self._capture(s, n_burnin) if self.use_graphs else None
        t_capture = _time.perf_counter() - t0
        t0 = _time.perf_counter()
        for it in range(n_iter):
            s["it"].fill_(it)
            if graphs is not None:
                graphs[it > 99].replay()
            else:
                self.step(s, it > 99, n_burnin)
            if progress and (it + 1) % max(1, n_iter // 10) == 0:
                if self.dev.type == "cuda":
                    torch.cuda.synchronize()
                print(f"  iteration {it + 1}/{n_iter}  ({_time.perf_counter() - t0:.1f}s)",
                      flush=True)
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed = _time.perf_counter() - t0
        sl = slice(n_burnin, None)
        out = dict(bs_gammas=s["r_bs"][sl], tau_bs_gammas=s["r_tau"][sl],
                   gammas=s["r_gammas"][sl], alphas=s["r_alphas"][sl], sds=s["r_sds"][sl],
                   L=s["r_L"][sl], betas=s["r_betas"][sl], sigmas=s["r_sigmas"][sl])
        out = {k: v.cpu().numpy() for k, v in out.items()}
        K = n_iter - n_burnin
        out["acc_b"] = (s["acc_b"] / K).cpu().numpy()
        out["mean_b"] = (s["sum_b"] / K).cpu().numpy()
        out["fitstats_acc"] = dict(sum_tot=s["sum_tot"].cpu().numpy(),
                                   mean=s["st_mean"].cpu().numpy(),
                                   m2=s["st_m2"].cpu().numpy(),
                                   lse_pos=s["lse_pos"].cpu().numpy(),
                                   lse_neg=s["lse_neg"].cpu().numpy())
        out["n_keep"] = int(s["n_keep"].item())
        out["elapsed"] = elapsed
        out["capture_time"] = t_capture
        out["device"] = str(self.dev)
        out["gpu_peak_mb"] = (torch.cuda.max_memory_allocated(self.dev) / 1e6
                              if self.dev.type == "cuda" else 0.0)
        return out
