"""Posterior summaries in the same form as summary.jm() / jm_fit() post-processing."""
from __future__ import annotations

import math

import numpy as np
import torch

from jmbayes.model import ModelData


# ---- coda-style diagnostics ---------------------------------------------------

def _ar_yw_spec0(x):
    """spectrum0.ar(): AR fit by Yule-Walker with AIC order choice (R's ar())."""
    n = len(x)
    if np.std(x) == 0:
        return 0.0
    order_max = int(min(n - 1, math.floor(10 * math.log10(n))))
    xc = x - x.mean()
    acf = np.array([np.dot(xc[:n - k], xc[k:]) / n for k in range(order_max + 1)])
    # Levinson-Durbin
    vars_ = [acf[0]]
    phi = np.zeros(0)
    phis = [phi]
    v = acf[0]
    for k in range(1, order_max + 1):
        num = acf[k] - (np.dot(phi, acf[1:k][::-1]) if k > 1 else 0.0)
        pac = num / v
        phi = np.concatenate([phi - pac * phi[::-1], [pac]])
        v = v * (1 - pac ** 2)
        vars_.append(v)
        phis.append(phi)
    vars_ = np.array(vars_)
    aic = n * np.log(vars_) + 2 * np.arange(order_max + 1) + 2
    order = int(np.argmin(aic))
    var_pred = vars_[order] * n / (n - (order + 1))
    return var_pred / (1 - phis[order].sum()) ** 2


def effective_size(x):
    """JMbayes2's effective_size(): chains stacked, AR spectral density at 0."""
    x = np.asarray(x)
    spec = _ar_yw_spec0(x)
    return 0.0 if spec == 0 else len(x) * x.var(ddof=1) / spec


def gelman_psrf(chains):
    """coda::gelman.diag point estimate (autoburnin = TRUE, transform = FALSE)."""
    chains = np.asarray(chains)                     # (n_chains, n_iter)
    m, n0 = chains.shape
    if m < 2:
        return np.nan
    x = chains[:, n0 - n0 // 2:] if n0 >= 50 else chains
    nn = x.shape[1]
    s2 = x.var(1, ddof=1)
    xbar = x.mean(1)
    w = s2.mean()
    b = nn * xbar.var(ddof=1)
    if w <= 0:
        return np.nan
    muhat = xbar.mean()
    var_w = s2.var(ddof=1) / m
    var_b = 2 * b ** 2 / (m - 1)
    cov_wb = nn / m * (np.cov(s2, xbar ** 2)[0, 1] - 2 * muhat * np.cov(s2, xbar)[0, 1])
    V = (nn - 1) * w / nn + (1 + 1 / m) * b / nn
    var_V = ((nn - 1) ** 2 * var_w + (1 + 1 / m) ** 2 * var_b
             + 2 * (nn - 1) * (1 + 1 / m) * cov_wb) / nn ** 2
    df_V = 2 * V ** 2 / var_V
    df_adj = (df_V + 3) / (df_V + 1)
    R2 = (nn - 1) / nn + (1 + 1 / m) * (1 / nn) * (b / w)
    return math.sqrt(df_adj * R2)


def ptail(x):
    return 2 * min(np.mean(x >= 0), np.mean(x < 0))


# ---- turning raw draws into named parameter tables ------------------------------

def build_D(sds, Lvals, upper, q, zero=()):
    """D = diag(sds) L'L diag(sds), returned as the lower triangle (R's order).
    reconstr_D() in help_functions.R, including correlations fixed at zero."""
    S = sds.shape[0]
    L = np.zeros((S, q, q))
    if len(upper):
        L[:, upper[:, 0], upper[:, 1]] = Lvals
    diag = np.sqrt(1 - (L ** 2).sum(1))
    L[:, np.arange(q), np.arange(q)] = diag
    for j0, j1 in zero:
        L[:, j0, j1] = -(L[:, :, j0] * L[:, :, j1]).sum(-1) / L[:, j0, j0]
        L[:, j1, j1] = np.sqrt(1 - (L[:, :j1, j1] ** 2).sum(-1))
    R = np.einsum("sij,sik->sjk", L, L)
    D = R * sds[:, :, None] * sds[:, None, :]
    rows, cols = np.tril_indices(q)
    order = np.lexsort((rows, cols))                   # column-major lower triangle
    return D[:, rows[order], cols[order]], [f"D[{r + 1}, {c + 1}]"
                                            for r, c in zip(rows[order], cols[order])]


def collect(md: ModelData, out: dict, n_thin=1):
    """Draws per parameter block as (n_chains, n_kept, k) arrays plus names."""
    keep = slice(None, None, n_thin)
    sw = lambda a: np.swapaxes(a[keep], 0, 1)          # noqa: E731 -> (C, S, k)
    blocks = {}
    for j, o in enumerate(md.outcomes):
        bj = sw(out["betas"])[:, :, o.fe_cols].copy()
        if o.Xbar.size and np.any(o.Xbar != 0):
            bj[:, :, 0] -= bj @ o.Xbar          # intercept back on the uncentred scale
        blocks[f"betas{j + 1}"] = (bj, [f"{n}" for n in o.x_names])
    blocks["sigmas"] = (sw(out["sigmas"])[:, :, [o.has_sigma for o in md.outcomes]],
                        [f"sigma_{j + 1}" for j, o in enumerate(md.outcomes) if o.has_sigma])
    blocks["gammas"] = (sw(out["gammas"]) / md.W_sds, md.w_names)
    blocks["alphas"] = (sw(out["alphas"]) / md.Wlong_sds,
                        list(md.assoc_names))
    blocks["bs_gammas"] = (sw(out["bs_gammas"]),
                           [f"bs_gammas_{k + 1}" for k in range(out["bs_gammas"].shape[2])])
    n_tau = out["tau_bs_gammas"].shape[2]
    if md.priors.get("penalized_bs_gammas", True):     # (fixed for a Weibull baseline)
        blocks["tau_bs_gammas"] = (sw(out["tau_bs_gammas"]),
                                   ["tau_bs_gammas"] if n_tau == 1
                                   else [f"tau_bs_gammas_{k + 1}" for k in range(n_tau)])
    if "sigmaF" in out:
        blocks["sigmaF"] = (sw(out["sigmaF"]), ["sigma_frailty"])
        lev = md.strata_levels or []
        blocks["alphaF"] = (sw(out["alphaF"]), [f"frailty:{v}" for v in lev[1:]]
                            or ["frailty"])
    C, S = blocks["gammas"][0].shape[:2]
    Dv, Dn = build_D(sw(out["sds"]).reshape(C * S, -1), sw(out["L"]).reshape(C * S, -1),
                     md.upper_part, md.q, [tuple(z) for z in md.ind_zero_D])
    blocks["D"] = (Dv.reshape(C, S, -1), Dn)
    return blocks


def table(draws, names):
    C, S, k = draws.shape
    rows = []
    for i in range(k):
        x = draws[:, :, i]
        flat = x.reshape(-1)
        ess = effective_size(flat)
        sd = flat.std(ddof=1)
        rows.append(dict(name=names[i], Mean=flat.mean(), StDev=sd,
                         **{"2.5%": np.quantile(flat, 0.025),
                            "97.5%": np.quantile(flat, 0.975)},
                         P=ptail(flat), Rhat=gelman_psrf(x),
                         ESS=ess, SE=sd / math.sqrt(ess) if ess > 0 else np.nan))
    return rows


# ---- fit statistics from the streamed accumulators ------------------------------

def _lower_to_full(Dl, q):
    rows, cols = np.tril_indices(q)
    order = np.lexsort((rows, cols))
    D = np.zeros((*Dl.shape[:-1], q, q))
    D[..., rows[order], cols[order]] = Dl
    return D + np.swapaxes(np.tril(D, -1), -1, -2)


def _sds_L(D):
    """sds and upper Cholesky factor of the correlation matrix (R = L'L), batched."""
    sds = np.sqrt(np.einsum("...ii->...i", D))
    R = D / (sds[..., :, None] * sds[..., None, :])
    return sds, np.swapaxes(np.linalg.cholesky(R), -1, -2)


def loglik_params(sampler, P: dict, b):
    """logLik_jm_stripped(): per-patient log-likelihood (longitudinal + survival +
    random effects, + frailty with recurrent events) for a batch of parameter values
    on the sampler's scale. P: betas (B, p), sigmas (B, J), bs, gammas, alphas, sds,
    L, and with recurrent events frailty (B, n), sigmaF (B,), alphaF (B, k); b: (B or 1,
    n, q). Returns (B, n) float64."""
    s_ = sampler
    t = s_.t
    T = {k: (v if torch.is_tensor(v) else t(v)) for k, v in P.items()}
    B = T["betas"].shape[0]
    bb = (b if torch.is_tensor(b) else t(b)).expand(B, -1, -1)
    C0 = s_.C
    s_.C = B
    try:
        ll_long = sum(s_.ll_long_subj(j, s_.eta_long(j, T["betas"], bb), T["sigmas"][:, j])
                      for j in range(s_.n_out))
        etaH = {k: s_.eta_surv(k, T["betas"], bb, "H") for k in range(s_.nS)}
        etah = {k: s_.eta_surv(k, T["betas"], bb, "h") for k in range(s_.nS)}
        z = torch.zeros(B, 1, 0, dtype=s_.dt, device=s_.dev)
        WlH = (torch.stack([s_.wlong_col(c, etaH, "H") for c in range(s_.A)], -1)
               if s_.A else z)
        Wlh = (torch.stack([s_.wlong_col(c, etah, "h") for c in range(s_.A)], -1)
               if s_.A else z)
        a = T["alphas"][:, None, :]
        FH = Fh = None
        if s_.rec:
            FH, Fh = s_.frail_terms(T["frailty"], T["sigmaF"], T["alphaF"])
        ll_surv = s_.log_surv(T["bs"] @ s_.W0_H.T, T["gammas"] @ s_.W_H.T,
                              (WlH * a).sum(-1), T["bs"] @ s_.W0_h.T,
                              T["gammas"] @ s_.W_h.T, (Wlh * a).sum(-1), FH, Fh)
        ll_re = s_.log_re(bb, T["L"], T["sds"])
        out = ll_long + ll_surv + ll_re
        if s_.rec:
            out = out - 0.5 * math.log(2 * math.pi) - 0.5 * T["frailty"] ** 2
    finally:
        s_.C = C0
    return out.to(torch.float64)


def _mean_params(md: ModelData, blocks, out):
    """Posterior means on the sampler's scale (as thetas <- statistics$Mean in jm_fit)."""
    def mean(k):
        a = blocks[k][0]
        return a.reshape(a.shape[0] * a.shape[1], -1).mean(0)
    betas = np.zeros(out["betas"].shape[2])
    for j, o in enumerate(md.outcomes):
        bj = mean(f"betas{j + 1}")
        if o.Xbar.size:
            bj = bj.copy()
            bj[0] += o.Xbar @ bj                 # back to the centred scale of the sampler
        betas[o.fe_cols] = bj
    sds, L = _sds_L(_lower_to_full(mean("D"), md.q))
    P = dict(betas=betas, sigmas=out["sigmas"].reshape(-1, out["sigmas"].shape[2]).mean(0),
             bs=mean("bs_gammas"), gammas=mean("gammas") * md.W_sds,
             alphas=mean("alphas") * md.Wlong_sds, sds=sds, L=L)
    if "sigmaF" in out:
        P.update(sigmaF=np.array(out["sigmaF"].mean()),
                 alphaF=out["alphaF"].reshape(-1, out["alphaF"].shape[2]).mean(0),
                 frailty=out["mean_frailty"].mean(0))
    return {k: np.asarray(v)[None] for k, v in P.items()}


def loglik_at_means(sampler, md: ModelData, blocks, out):
    """logLik_jm(): per-subject log-likelihood at the posterior means."""
    P = _mean_params(md, blocks, out)
    b = out["mean_b"].mean(0)[None]
    return loglik_params(sampler, P, b)[0].cpu().numpy()


def posterior_b(out):
    """Posterior means and covariances of each patient's random effects over all
    chains (postmeans_b, postvars_b in jm_fit())."""
    K = out["n_keep"] * out["sum_b"].shape[0]
    m = out["sum_b"].sum(0) / K
    V = (out["sum_bb"].sum(0) / K - m[:, :, None] * m[:, None, :]) * K / (K - 1)
    return m, V


def marginal_fit_stats(sampler, md: ModelData, blocks, out, ll_mean_cond, batch=None):
    """Marginal DIC, WAIC and LPML, as mlogLik_jm() in jm_fit(): for every kept draw the
    log-likelihood at the posterior means of the random effects plus the Laplace term
    0.5 (q log 2 pi + log det postvar_i), streamed (no draws x patients table kept)."""
    mb, V = posterior_b(out)
    q = md.q
    sign, logdet = np.linalg.slogdet(V)
    const = 0.5 * (q * math.log(2 * math.pi) + logdet)
    const = np.where(sign > 0, const, np.nan)
    S, C = out["betas"].shape[:2]
    flat = lambda a: a.reshape(S * C, *a.shape[2:])  # noqa: E731 (draw-major, chain-minor)
    Dv, _ = build_D(flat(out["sds"]), flat(out["L"]), md.upper_part, q,
                    [tuple(z) for z in md.ind_zero_D])
    sds_all, L_all = _sds_L(_lower_to_full(Dv, q))
    P_all = dict(betas=flat(out["betas"]), sigmas=flat(out["sigmas"]),
                 bs=flat(out["bs_gammas"]), gammas=flat(out["gammas"]),
                 alphas=flat(out["alphas"]), sds=sds_all, L=L_all)
    if "sigmaF" in out:
        P_all.update(sigmaF=flat(out["sigmaF"])[:, 0], alphaF=flat(out["alphaF"]),
                     frailty=flat(out["frailty"]))
    N = S * C
    n = md.n
    if batch is None:
        pts = max(1, md.W0_H.shape[0] + sum(len(o.y) for o in md.outcomes))
        batch = int(max(1, min(512, 4e7 // pts)))
    b_t = sampler.t(mb)[None]
    c_t = torch.as_tensor(const, dtype=torch.float64, device=sampler.dev)
    tot = 0.0
    mean = torch.zeros(n, dtype=torch.float64, device=sampler.dev)
    m2 = torch.zeros_like(mean)
    lse_p = torch.full_like(mean, -math.inf)
    lse_n = torch.full_like(mean, -math.inf)
    k = 0
    with torch.no_grad():
        for a in range(0, N, batch):
            P = {key: v[a:a + batch] for key, v in P_all.items()}
            ll = loglik_params(sampler, P, b_t) + c_t                     # (B, n)
            tot += float(ll.sum())
            nb = ll.shape[0]                                  # merge batch moments (Chan)
            bm = ll.mean(0)
            bM2 = ((ll - bm) ** 2).sum(0)
            d = bm - mean
            mean = mean + d * nb / (k + nb)
            m2 = m2 + bM2 + d * d * k * nb / (k + nb)
            k += nb
            lse_p = torch.logaddexp(lse_p, torch.logsumexp(ll, 0))
            lse_n = torch.logaddexp(lse_n, torch.logsumexp(-ll, 0))
    acc = dict(sum_tot=np.array([tot]), mean=mean.cpu().numpy()[None],
               m2=m2.cpu().numpy()[None], lse_pos=lse_p.cpu().numpy()[None],
               lse_neg=lse_n.cpu().numpy()[None])
    return fit_stats(acc, N, ll_mean_cond + const)


def fit_stats(acc, n_keep, ll_mean_parms):
    """fit_stats() in help_functions.R, from per-chain running sums."""
    C = acc["mean"].shape[0]
    N = C * n_keep
    D_bar = -2.0 * acc["sum_tot"].sum() / N
    D_hat = -2.0 * ll_mean_parms.sum()
    pD = D_bar - D_hat
    lse = lambda a: np.logaddexp.reduce(a, axis=0)     # noqa: E731
    lse_pos, lse_neg = lse(acc["lse_pos"]), lse(acc["lse_neg"])
    CC = math.log(N)
    LPML = np.sum(-lse_neg + CC)
    LPPD = -2.0 * np.sum(lse_pos - CC)
    # combine per-chain Welford moments
    gmean = acc["mean"].mean(0)
    m2 = acc["m2"].sum(0) + (n_keep * (acc["mean"] - gmean) ** 2).sum(0)
    pWAIC2 = 2.0 * np.sum(m2 / (N - 1))
    return dict(DIC=pD + D_bar, pD=pD, LPML=LPML, WAIC=LPPD + pWAIC2)


def print_summary(md, blocks, fit, timing, n_chains, n_iter, n_burnin, n_thin):
    def show(title, rows, cols=("Mean", "StDev", "2.5%", "97.5%", "P", "Rhat")):
        print(f"\n{title}")
        w = max(len(r["name"]) for r in rows) + 2
        print(" " * w + "".join(f"{c:>10}" for c in cols))
        for r in rows:
            print(f"{r['name']:<{w}}" + "".join(f"{r[c]:>10.4f}" for c in cols))

    print(f"\nJoint model (Python/PyTorch port of JMbayes2)  "
          f"subjects: {md.n}  events: {int(md.event.sum())}")
    print(f"MCMC: {n_chains} chains, {n_iter} iterations, burn-in {n_burnin}, thin {n_thin}, "
          f"{timing:.1f}s")
    print("\nConditional fit statistics:  " +
          "  ".join(f"{k}={v:.3f}" for k, v in fit.items()))
    show("Survival outcome", table(*blocks["gammas"]) + table(*blocks["alphas"]))
    for j, o in enumerate(md.outcomes):
        rows = table(*blocks[f"betas{j + 1}"])
        if o.has_sigma:
            k = [i for i, oo in enumerate(md.outcomes) if oo.has_sigma].index(j)
            s, n = blocks["sigmas"]
            rows += table(s[:, :, [k]], ["sigma"])
        show(f"Longitudinal outcome: {o.name} ({o.family})", rows)
    show("Random-effects covariance D", table(*blocks["D"]),
         cols=("Mean", "StDev", "2.5%", "97.5%", "Rhat"))
