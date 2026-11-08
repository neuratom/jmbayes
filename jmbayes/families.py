"""Distributions of the longitudinal markers, as log_long_i() in JMbayes2's
src/JMbayes2_LogDens.h and the GLMMadaptive families JMbayes2 takes them from.

family              link      extra parameter (JMbayes2's "sigma")     auxiliary column
gaussian            identity  residual SD                              -
binomial            logit     -                                        trials (optional)
poisson             log       -                                        -
negative_binomial   log       size (variance = mu + mu^2 / size)       -
student_t           identity  scale; df fixed (given by the user)      -
beta                logit     precision phi (shapes mu phi, (1-mu) phi) -
gamma               log       shape (scale = mu / shape)               -
unit_lindley        logit     -                                        -
censored_normal     identity  residual SD                              censoring code
beta_binomial       logit     precision phi                            trials (optional)

Censoring codes (censored_normal): 0 observed, 1 left-censored (the true value is at
most y, e.g. below a detection limit), 2 right-censored (at least y), as JMbayes2.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.special import digamma, expit, gammaln, log_expit, log_ndtr, polygamma

FAMILIES = {
    "gaussian": dict(link="identity", sigma=True),
    "binomial": dict(link="logit", sigma=False),
    "poisson": dict(link="log", sigma=False),
    "negative_binomial": dict(link="log", sigma=True),
    "student_t": dict(link="identity", sigma=True),
    "beta": dict(link="logit", sigma=True),
    "gamma": dict(link="log", sigma=True),
    "unit_lindley": dict(link="logit", sigma=False),
    "censored_normal": dict(link="identity", sigma=True),
    "beta_binomial": dict(link="logit", sigma=True),
}
TRIALS = ("binomial", "beta_binomial")          # may have a number of trials per row
LOG2PI = math.log(2 * math.pi)


def has_sigma(family: str) -> bool:
    return FAMILIES[family]["sigma"]


# ------------------------------------------------------------------ torch (sampler)
def precompute(family, y, aux=None):
    """Terms of the log density that do not depend on the parameters (tensors)."""
    pre = {}
    if family in ("poisson", "negative_binomial"):
        pre["lgy1"] = torch.lgamma(y + 1.0)
    elif family in TRIALS and aux is not None:
        pre["n"] = aux
        pre["lchoose"] = torch.lgamma(aux + 1.0) - torch.lgamma(y + 1.0) - torch.lgamma(
            aux - y + 1.0)
    elif family == "beta_binomial":
        pre["n"] = torch.ones_like(y)
        pre["lchoose"] = torch.zeros_like(y)
    elif family in ("beta", "gamma"):
        pre["ly"] = torch.log(y)
        if family == "beta":
            pre["l1y"] = torch.log1p(-y)
    elif family == "unit_lindley":
        pre["c"] = -3.0 * torch.log1p(-y)
        pre["r"] = y / (1.0 - y)
    elif family == "censored_normal":
        pre["obs"], pre["left"] = aux == 0, aux == 1
    return pre


# Families whose density is a difference of large lgamma terms when the dispersion is
# large (e.g. lgamma(y + A) - lgamma(A) with A ~ 1e4). In float32 the rounding error then
# exceeds the signal and the sampler drifts to huge dispersions, so these are computed
# in float64.
WIDE = ("negative_binomial", "beta", "gamma", "beta_binomial")


def log_dens(family, y, eta, sigma, df=0.0, pre=None):
    """Log density of each observation. eta: (..., N); sigma broadcastable to eta.
    pre: precompute(family, y, aux)."""
    if family in WIDE and eta.dtype != torch.float64:
        if pre is None:
            pre = precompute(family, y)
        d = torch.float64
        pre = {k: v.to(d) if v.is_floating_point() else v for k, v in pre.items()}
        out = _log_dens(family, y.to(d), eta.to(d), torch.as_tensor(sigma).to(eta.device, d),
                        df, pre)
        return out.to(eta.dtype)
    return _log_dens(family, y, eta, sigma, df, pre)


def _log_dens(family, y, eta, sigma, df=0.0, pre=None):
    if family == "gaussian":
        return -torch.log(sigma) - 0.5 * LOG2PI - (y - eta) ** 2 / (2 * sigma * sigma)
    if pre is None:
        pre = precompute(family, y)
    if family == "binomial":
        if "n" not in pre:
            mu = 1.0 / (1.0 + torch.exp(-eta))
            return torch.log(torch.where(y > 0, mu, 1.0 - mu))
        ls = torch.nn.functional.logsigmoid
        return pre["lchoose"] + y * ls(eta) + (pre["n"] - y) * ls(-eta)
    if family == "poisson":
        return y * eta - torch.exp(eta) - pre["lgy1"]
    if family == "negative_binomial":
        k = sigma
        log_mu_k = torch.log(torch.exp(eta) + k)
        return (torch.lgamma(y + k) - torch.lgamma(k) - pre["lgy1"] + k * torch.log(k)
                - k * log_mu_k + y * (eta - log_mu_k))
    if family == "student_t":
        const = math.lgamma(0.5 * (df + 1.0)) - math.lgamma(0.5 * df) - 0.5 * math.log(
            df * math.pi)
        r = (y - eta) / sigma
        return const - torch.log(sigma) - 0.5 * (df + 1.0) * torch.log1p(r * r / df)
    if family == "beta":
        mu = torch.sigmoid(eta)
        a, b = mu * sigma, (1.0 - mu) * sigma
        return ((a - 1.0) * pre["ly"] + (b - 1.0) * pre["l1y"] - torch.lgamma(a)
                - torch.lgamma(b) + torch.lgamma(a + b))
    if family == "gamma":
        k = sigma
        return (-torch.lgamma(k) - k * (eta - torch.log(k)) + (k - 1.0) * pre["ly"]
                - y * k * torch.exp(-eta))
    if family == "unit_lindley":
        th = torch.exp(-eta)                                   # 1 / mu - 1
        return -2.0 * eta - torch.log1p(th) + pre["c"] - th * pre["r"]
    if family == "censored_normal":
        z = (y - eta) / sigma
        dens = -torch.log(sigma) - 0.5 * LOG2PI - 0.5 * z * z
        return torch.where(pre["obs"], dens, torch.special.log_ndtr(
            torch.where(pre["left"], z, -z)))
    if family == "beta_binomial":
        mu = torch.sigmoid(eta)
        A, B, n = sigma * mu, sigma * (1.0 - mu), pre["n"]
        return (pre["lchoose"] + torch.lgamma(y + A) + torch.lgamma(n - y + B)
                - torch.lgamma(n + sigma) + torch.lgamma(sigma) - torch.lgamma(A)
                - torch.lgamma(B))
    raise NotImplementedError(f"family {family}")


def mean(family, eta):
    """Expected marker value given the linear predictor (the inverse link); for the
    binomial families with trials, the probability per trial."""
    link = FAMILIES[family]["link"]
    if link == "identity":
        return eta
    if link == "logit":
        return 1.0 / (1.0 + torch.exp(-eta))
    return torch.exp(eta)


# ------------------------------------------------------------------ numpy (initial fits)
def np_loglik(family, y, eta, phi, df=0.0, aux=None):
    """Log density per observation; eta (N,) or (N, K); phi = log extra parameter."""
    if eta.ndim == 2:
        y = y[:, None]
        aux = None if aux is None else aux[:, None]
    if family == "gaussian":
        s = np.exp(phi)
        return -phi - 0.5 * LOG2PI - (y - eta) ** 2 / (2 * s * s)
    if family == "binomial":
        if aux is None:
            return y * log_expit(eta) + (1 - y) * log_expit(-eta)
        return (_lchoose(aux, y) + y * log_expit(eta) + (aux - y) * log_expit(-eta))
    if family == "poisson":
        return y * eta - np.exp(eta) - gammaln(y + 1.0)
    if family == "negative_binomial":
        k = np.exp(phi)
        log_mu_k = np.logaddexp(eta, phi)
        return (gammaln(y + k) - gammaln(k) - gammaln(y + 1.0) + k * phi - k * log_mu_k
                + y * (eta - log_mu_k))
    if family == "student_t":
        s = np.exp(phi)
        r = (y - eta) / s
        return (gammaln(0.5 * (df + 1.0)) - gammaln(0.5 * df) - 0.5 * np.log(df * np.pi)
                - phi - 0.5 * (df + 1.0) * np.log1p(r * r / df))
    if family == "beta":
        p = np.exp(phi)
        mu = expit(eta)
        a, b = mu * p, (1 - mu) * p
        return ((a - 1) * np.log(y) + (b - 1) * np.log1p(-y) - gammaln(a) - gammaln(b)
                + gammaln(p))
    if family == "gamma":
        k = np.exp(phi)
        return -gammaln(k) - k * (eta - phi) + (k - 1) * np.log(y) - y * k * np.exp(-eta)
    if family == "unit_lindley":
        th = np.exp(-eta)
        return -2 * eta - np.log1p(th) - 3 * np.log1p(-y) - th * y / (1 - y)
    if family == "censored_normal":
        s = np.exp(phi)
        z = (y - eta) / s
        dens = -phi - 0.5 * LOG2PI - 0.5 * z * z
        return np.where(aux == 0, dens, log_ndtr(np.where(aux == 1, z, -z)))
    if family == "beta_binomial":
        p = np.exp(phi)
        n = np.ones_like(y) if aux is None else aux
        mu = expit(eta)
        A, B = p * mu, p * (1 - mu)
        return (_lchoose(n, y) + gammaln(y + A) + gammaln(n - y + B) - gammaln(n + p)
                + gammaln(p) - gammaln(A) - gammaln(B))
    raise NotImplementedError(f"family {family}")


def _lchoose(n, k):
    return gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)


def np_score_weight(family, y, eta, phi, df=0.0, aux=None):
    """d loglik / d eta and a positive curvature weight (Fisher information or a bound
    on the curvature), for the Newton steps that find each patient's random-effects
    mode. Only the score needs to be exact: the weight sets the step size and the scale
    of the adaptive quadrature."""
    if family == "gaussian":
        s2 = np.exp(2 * phi)
        return (y - eta) / s2, np.full_like(eta, 1.0 / s2)
    if family == "binomial":
        pr = expit(eta)
        n = 1.0 if aux is None else aux
        return y - n * pr, n * pr * (1 - pr)
    if family == "poisson":
        mu = np.exp(eta)
        return y - mu, mu
    if family == "negative_binomial":
        k = np.exp(phi)
        mu = np.exp(eta)
        f = k / (k + mu)
        return (y - mu) * f, mu * f
    if family == "student_t":
        s2 = np.exp(2 * phi)
        r = y - eta
        return (df + 1.0) * r / (df * s2 + r * r), np.full_like(eta, (df + 1.0) / ((df + 3.0) * s2))
    if family == "beta":
        p = np.exp(phi)
        mu = expit(eta)
        a, b = mu * p, (1 - mu) * p
        d = p * mu * (1 - mu)
        sc = d * (np.log(y) - np.log1p(-y) - digamma(a) + digamma(b))
        return sc, d * d * (polygamma(1, a) + polygamma(1, b))
    if family == "gamma":
        k = np.exp(phi)
        return k * (y * np.exp(-eta) - 1.0), np.full_like(eta, k)
    if family == "unit_lindley":
        mu = expit(eta)
        tr = np.exp(-eta) * y / (1 - y)
        return -1.0 - mu + tr, mu * (1 - mu) + tr
    if family == "censored_normal":
        s = np.exp(phi)
        z = (y - eta) / s
        lphi = -0.5 * LOG2PI - 0.5 * z * z
        sc = np.where(aux == 0, z / s,
                      np.where(aux == 1, -np.exp(lphi - log_ndtr(z)) / s,
                               np.exp(lphi - log_ndtr(-z)) / s))
        return sc, np.full_like(eta, 1.0 / (s * s))
    if family == "beta_binomial":
        p = np.exp(phi)
        n = np.ones_like(y) if aux is None else aux
        mu = expit(eta)
        A, B = p * mu, p * (1 - mu)
        d = p * mu * (1 - mu)
        sc = d * (digamma(y + A) - digamma(n - y + B) - digamma(A) + digamma(B))
        return sc, n * mu * (1 - mu) * (p + 1) / (p + n)
    raise NotImplementedError(f"family {family}")
