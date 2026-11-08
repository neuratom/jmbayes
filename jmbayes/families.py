"""Distributions of the longitudinal markers, as log_long_i() in JMbayes2's
src/JMbayes2_LogDens.h and the GLMMadaptive families JMbayes2 takes them from.

family              link      extra parameter (JMbayes2's "sigma")
gaussian            identity  residual SD
binomial            logit     -
poisson             log       -
negative_binomial   log       size (dispersion; variance = mu + mu^2 / size)
student_t           identity  scale; the degrees of freedom are fixed (given by the user)
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.special import expit, gammaln, log_expit

FAMILIES = {
    "gaussian": dict(link="identity", sigma=True),
    "binomial": dict(link="logit", sigma=False),
    "poisson": dict(link="log", sigma=False),
    "negative_binomial": dict(link="log", sigma=True),
    "student_t": dict(link="identity", sigma=True),
}
LOG2PI = math.log(2 * math.pi)


def has_sigma(family: str) -> bool:
    return FAMILIES[family]["sigma"]


# ------------------------------------------------------------------ torch (sampler)
def log_dens(family, y, eta, sigma, df=0.0, lgy1=None):
    """Log density of each observation. eta: (..., N); sigma broadcastable to eta.
    lgy1: lgamma(y + 1), precomputed for the count families."""
    if family == "gaussian":
        return -torch.log(sigma) - 0.5 * LOG2PI - (y - eta) ** 2 / (2 * sigma * sigma)
    if family == "binomial":
        mu = 1.0 / (1.0 + torch.exp(-eta))
        return torch.log(torch.where(y > 0, mu, 1.0 - mu))
    if lgy1 is None:
        lgy1 = torch.lgamma(y + 1.0)
    if family == "poisson":
        return y * eta - torch.exp(eta) - lgy1
    if family == "negative_binomial":
        k = sigma
        log_mu_k = torch.log(torch.exp(eta) + k)
        return (torch.lgamma(y + k) - torch.lgamma(k) - lgy1 + k * torch.log(k)
                - k * log_mu_k + y * (eta - log_mu_k))
    if family == "student_t":
        const = math.lgamma(0.5 * (df + 1.0)) - math.lgamma(0.5 * df) - 0.5 * math.log(
            df * math.pi)
        r = (y - eta) / sigma
        return const - torch.log(sigma) - 0.5 * (df + 1.0) * torch.log1p(r * r / df)
    raise NotImplementedError(f"family {family}")


def mean(family, eta):
    """Expected marker value given the linear predictor (the inverse link)."""
    link = FAMILIES[family]["link"]
    if link == "identity":
        return eta
    if link == "logit":
        return 1.0 / (1.0 + torch.exp(-eta))
    return torch.exp(eta)


# ------------------------------------------------------------------ numpy (initial fits)
def np_loglik(family, y, eta, phi, df=0.0):
    """Log density per observation; eta (N,) or (N, K); phi = log extra parameter."""
    if eta.ndim == 2:
        y = y[:, None]
    if family == "binomial":
        return y * log_expit(eta) + (1 - y) * log_expit(-eta)
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
    raise NotImplementedError(f"family {family}")


def np_score_weight(family, y, eta, phi, df=0.0):
    """d loglik / d eta and a positive curvature weight (Fisher information), for the
    Newton steps that find each patient's random-effects mode."""
    if family == "binomial":
        pr = expit(eta)
        return y - pr, pr * (1 - pr)
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
    raise NotImplementedError(f"family {family}")
