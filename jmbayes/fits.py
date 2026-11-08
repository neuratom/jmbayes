"""Python versions of the separate model fits that jm() starts from.

JMbayes2 takes fitted lme(), mixed_model() and coxph() objects and uses them
for starting values and to centre some priors. These are the same estimators:

* lme (REML linear mixed model)        -> statsmodels MixedLM, reml=True
* mixed_model: binomial, Poisson, negative binomial, Student's t (ML with
  adaptive Gauss-Hermite, 11 points, like GLMMadaptive's default) -> fit_glmm() below
* coxph (Efron ties, R's default)       -> fit_coxph() below
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from jmbayes.families import has_sigma, np_loglik, np_score_weight


@dataclass
class MixedFit:
    betas: np.ndarray        # fixed effects
    vcov: np.ndarray         # covariance of fixed effects
    D: np.ndarray            # random-effects covariance
    b: np.ndarray            # per-subject random effects (n_subjects_in_outcome x q)
    log_sigma: float         # -20 when the family has no dispersion (R convention)


def fit_lme(y, X, Z, groups, diagonal=False) -> MixedFit:
    """REML linear mixed model; diagonal=True gives uncorrelated random effects."""
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLMParams
    model = sm.MixedLM(y, X, groups=groups, exog_re=Z)
    free = (MixedLMParams.from_components(np.ones(X.shape[1]), cov_re=np.eye(Z.shape[1]))
            if diagonal else None)
    res = None
    for method in ("lbfgs", "bfgs", "powell"):
        try:
            res = model.fit(reml=True, method=method, free=free)
            if res.converged:
                break
        except Exception:  # noqa: BLE001 - try the next optimiser
            continue
    if res is None:
        raise RuntimeError("lme fit failed")
    p = X.shape[1]
    vcov = np.asarray(res.cov_params())[:p, :p]
    D = np.asarray(res.cov_re)
    uniq = list(dict.fromkeys(groups))
    b = np.vstack([np.asarray(res.random_effects[g]) for g in uniq])
    return MixedFit(np.asarray(res.fe_params), vcov, D, b,
                    float(np.log(np.sqrt(res.scale))))


# ---------------------------------------------------------------------------
# GLMMs (binomial, Poisson, negative binomial, Student's t): maximum likelihood with
# adaptive Gauss-Hermite quadrature, as GLMMadaptive::mixed_model()
# ---------------------------------------------------------------------------

def _gh_grid(q: int, k: int):
    x, w = np.polynomial.hermite.hermgauss(k)          # physicists' weights
    grid = np.array(list(np.ndindex(*([k] * q))))
    nodes = x[grid]                                    # (k^q, q)
    log_w = np.log(w[grid]).sum(1)
    return nodes, log_w


class _GLMM:
    def __init__(self, y, X, Z, sid, n_agq=11, diagonal=False, family="binomial", df=0.0):
        self.y, self.X, self.Z, self.sid = y, X, Z, sid
        self.family, self.df = family, df
        self.n_phi = int(has_sigma(family))                # log size / log scale
        self.n = sid.max() + 1
        self.p, self.q = X.shape[1], Z.shape[1]
        self.nodes, self.log_w = _gh_grid(self.q, n_agq)
        self.modes = np.zeros((self.n, self.q))
        self.tril = (np.diag_indices(self.q) if diagonal else np.tril_indices(self.q))

    def unpack(self, theta):
        beta = theta[:self.p]
        phi = theta[self.p] if self.n_phi else 0.0
        Lc = np.zeros((self.q, self.q))
        Lc[self.tril] = theta[self.p + self.n_phi:]
        Lc[np.diag_indices(self.q)] = np.exp(np.diag(Lc))
        return beta, phi, Lc @ Lc.T

    def _curv(self, w):
        H = np.zeros((self.n, self.q, self.q))
        np.add.at(H, self.sid, w[:, None, None] * self.Z[:, :, None] * self.Z[:, None, :])
        return H

    def _modes(self, beta, phi, Dinv):
        b = self.modes.copy()
        xb = self.X @ beta
        for _ in range(100):
            eta = xb + np.einsum("ij,ij->i", self.Z, b[self.sid])
            sc, w = np_score_weight(self.family, self.y, eta, phi, self.df)
            g = np.zeros((self.n, self.q))
            np.add.at(g, self.sid, -self.Z * sc[:, None])
            g += b @ Dinv
            H = self._curv(w) + Dinv
            step = np.linalg.solve(H, g[..., None])[..., 0]
            b = b - step
            if np.abs(step).max() < 1e-10:
                break
        # curvature at the final mode
        eta = xb + np.einsum("ij,ij->i", self.Z, b[self.sid])
        _, w = np_score_weight(self.family, self.y, eta, phi, self.df)
        return b, self._curv(w) + Dinv

    def negloglik(self, theta, update_modes=False):
        """Minus the log-likelihood; 1e300 at parameter values where it cannot be
        computed (so the optimiser's line search steps back)."""
        try:
            with np.errstate(all="ignore"):
                return self._negloglik(theta, update_modes)
        except np.linalg.LinAlgError:
            return 1e300

    def _negloglik(self, theta, update_modes):
        beta, phi, D = self.unpack(theta)
        Dinv = np.linalg.inv(D)
        b, H = self._modes(beta, phi, Dinv)
        if update_modes:
            self.modes = b
        A = np.linalg.cholesky(np.linalg.inv(H))                 # (n, q, q)
        bk = b[:, None, :] + np.sqrt(2) * np.einsum("nij,kj->nki", A, self.nodes)
        eta = (self.X @ beta)[:, None] + np.einsum("ij,ikj->ik", self.Z, bk[self.sid])
        ll_obs = np_loglik(self.family, self.y, eta, phi, self.df)
        ll = np.zeros((self.n, len(self.log_w)))
        np.add.at(ll, self.sid, ll_obs)
        quad = np.einsum("nki,ij,nkj->nk", bk, Dinv, bk)
        h = -ll + 0.5 * quad                                     # -log integrand (no consts)
        log_int = self.log_w + (self.nodes ** 2).sum(1) - h
        m = log_int.max(1, keepdims=True)
        lse = (m[:, 0] + np.log(np.exp(log_int - m).sum(1)))
        logdetA = np.log(np.diagonal(A, axis1=1, axis2=2)).sum(1)
        _, logdetD = np.linalg.slogdet(D)
        li = 0.5 * self.q * np.log(2) + logdetA + lse \
            - 0.5 * self.q * np.log(2 * np.pi) - 0.5 * logdetD
        out = -li.sum()
        return out if np.isfinite(out) else 1e300


def _glm_start(family, y, X):
    """Fixed effects and log extra parameter of the model without random effects."""
    import statsmodels.api as sm
    if family == "binomial":
        return sm.GLM(y, X, family=sm.families.Binomial()).fit().params, []
    if family in ("poisson", "negative_binomial"):
        beta = sm.GLM(y, X, family=sm.families.Poisson()).fit().params
        if family == "poisson":
            return beta, []
        mu = np.exp(X @ beta)
        excess = max(np.mean((y - mu) ** 2 - mu), 1e-3 * np.mean(mu))
        return beta, [np.log(np.clip(np.mean(mu) ** 2 / excess, 1e-3, 1e3))]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]                  # student_t
    return beta, [np.log(np.std(y - X @ beta))]


def fit_glmm(y, X, Z, groups, family="binomial", df=0.0, n_agq=11,
             diagonal=False) -> MixedFit:
    """GLMM by maximum likelihood (adaptive Gauss-Hermite, 11 points per random effect).
    log_sigma is GLMMadaptive's 'phis': the log size (negative binomial) or log scale
    (Student's t); -20 when the family has no extra parameter (R convention)."""
    uniq = list(dict.fromkeys(groups))
    index = {g: i for i, g in enumerate(uniq)}
    sid = np.array([index[g] for g in groups])
    m = _GLMM(y.astype(float), X, Z, sid, n_agq, diagonal, family, df)
    beta0, phi0 = _glm_start(family, y.astype(float), X)
    theta0 = np.concatenate([beta0, phi0, np.zeros(len(m.tril[0]))])
    m.negloglik(theta0, update_modes=True)
    best = None
    for _ in range(5):                    # refresh warm-start modes between rounds
        res = optimize.minimize(m.negloglik, theta0, method="BFGS",
                                options={"gtol": 1e-6, "maxiter": 500})
        m.negloglik(res.x, update_modes=True)
        if best is not None and abs(best.fun - res.fun) < 1e-8:
            best = res
            break
        best, theta0 = res, res.x
    theta = best.x
    Hs = _num_hessian(m.negloglik, theta)
    vcov = np.linalg.inv(Hs)[:m.p, :m.p]
    beta, phi, D = m.unpack(theta)
    m.negloglik(theta, update_modes=True)
    return MixedFit(beta, vcov, D, m.modes.copy(), float(phi) if m.n_phi else -20.0)


def fit_glmm_binomial(y, X, Z, groups, n_agq=11, diagonal=False) -> MixedFit:
    return fit_glmm(y, X, Z, groups, "binomial", n_agq=n_agq, diagonal=diagonal)


def _num_hessian(f, x, h=1e-4):
    k = len(x)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            e_i = np.eye(k)[i] * h
            e_j = np.eye(k)[j] * h
            H[i, j] = H[j, i] = (f(x + e_i + e_j) - f(x + e_i - e_j)
                                 - f(x - e_i + e_j) + f(x - e_i - e_j)) / (4 * h * h)
    return H


# ---------------------------------------------------------------------------
# Cox model, Efron ties
# ---------------------------------------------------------------------------

def fit_coxph(time, event, W, strata=None, start=None) -> np.ndarray:
    """Cox model by Newton-Raphson with Efron ties. strata: optional stratum of each row;
    risk sets are then formed within strata, as with coxph(... + strata()). start:
    optional entry times (delayed entry, Surv(start, stop, event)): a patient is at risk
    at t only when start < t <= time."""
    beta = np.zeros(W.shape[1])
    if strata is None:
        strata = np.zeros(len(time), int)
    if start is None:
        start = np.full(len(time), -np.inf)
    groups = [(k, np.unique(time[(event == 1) & (strata == k)])) for k in np.unique(strata)]
    for _ in range(50):
        grad = np.zeros_like(beta)
        hess = np.zeros((len(beta), len(beta)))
        r = np.exp(W @ beta)
        for k, t in ((k, t) for k, ts in groups for t in ts):
            risk = (time >= t) & (start < t) & (strata == k)
            dead = (time == t) & (event == 1) & (strata == k)
            d = dead.sum()
            s0r, s1r = r[risk].sum(), (r[risk, None] * W[risk]).sum(0)
            s2r = (r[risk, None, None] * W[risk, :, None] * W[risk, None, :]).sum(0)
            s0d, s1d = r[dead].sum(), (r[dead, None] * W[dead]).sum(0)
            s2d = (r[dead, None, None] * W[dead, :, None] * W[dead, None, :]).sum(0)
            grad += W[dead].sum(0)
            for l in range(d):
                frac = l / d
                s0 = s0r - frac * s0d
                s1 = s1r - frac * s1d
                s2 = s2r - frac * s2d
                grad -= s1 / s0
                hess -= s2 / s0 - np.outer(s1, s1) / s0 ** 2
        step = np.linalg.solve(hess, grad)
        beta = beta - step
        if np.abs(step).max() < 1e-10:
            break
    return beta
