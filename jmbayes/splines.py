"""R-identical spline bases: splines::ns() and splines::bs().

The natural-spline basis is not unique (any rotation of it spans the same space), and
R's choice comes from a LINPACK Householder QR. To get the same coefficients and the
same priors as JMbayes2, that QR is reproduced exactly here (dqrdc2 / dqrsl).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline


def spline_design(knots, x, order, derivs=0):
    """splines::splineDesign(knots, x, ord, derivs) for x inside the knot range."""
    knots = np.asarray(knots, float)
    x = np.atleast_1d(np.asarray(x, float))
    derivs = np.broadcast_to(np.asarray(derivs), x.shape)
    k = order - 1
    nb = len(knots) - order
    out = np.zeros((len(x), nb))
    for j in range(nb):
        c = np.zeros(nb)
        c[j] = 1.0
        spl = BSpline(knots, c, k, extrapolate=True)
        for d in np.unique(derivs):
            rows = derivs == d
            f = spl.derivative(int(d)) if d > 0 else spl
            out[rows, j] = f(x[rows])
    return out


def _quantile7(x, probs):
    """R's default quantile (type 7)."""
    return np.quantile(np.asarray(x, float), probs, method="linear")


def _linpack_qr(a):
    """LINPACK dqrdc2 without pivoting (R's qr() default). Returns (qr, qraux)."""
    x = np.array(a, float, copy=True)
    n, p = x.shape
    qraux = np.zeros(p)
    for j in range(min(n, p)):
        nrmxl = np.linalg.norm(x[j:, j])
        if nrmxl == 0.0:
            continue
        if x[j, j] != 0.0:
            nrmxl = np.copysign(nrmxl, x[j, j])
        x[j:, j] /= nrmxl
        x[j, j] += 1.0
        for l in range(j + 1, p):
            t = -np.dot(x[j:, l], x[j:, j]) / x[j, j]
            x[j:, l] += t * x[j:, j]
        qraux[j] = x[j, j]
        x[j, j] = -nrmxl
    return x, qraux


def _qr_qty(qr, qraux, y):
    """qr.qty(): Q' y using the LINPACK Householder vectors (dqrsl)."""
    x = qr.copy()
    y = np.array(y, float, copy=True)
    n, p = x.shape
    for j in range(min(n - 1, p)):
        if qraux[j] == 0.0:
            continue
        temp = x[j, j]
        x[j, j] = qraux[j]
        t = -(x[j:, j] @ y[j:]) / x[j, j]
        y[j:] += np.outer(x[j:, j], t) if y.ndim == 2 else t * x[j:, j]
        x[j, j] = temp
    return y


class SplineSpec:
    """Knots fixed on the fitting data and reused for new data (like makepredictcall)."""

    def __init__(self, kind, df, knots, boundary, degree=3):
        self.kind, self.df, self.knots, self.boundary, self.degree = (
            kind, df, np.asarray(knots, float), np.asarray(boundary, float), degree)

    @classmethod
    def fit(cls, kind, x, df, degree=3):
        x = np.asarray(x, float)
        boundary = np.array([x.min(), x.max()])
        if kind == "ns":
            n_inner = df - 1                          # intercept = FALSE
        else:
            n_inner = df - degree                     # bs(), intercept = FALSE
        if n_inner > 0:
            probs = np.linspace(0, 1, n_inner + 2)[1:-1]
            knots = _quantile7(x, probs)
        else:
            knots = np.zeros(0)
        return cls(kind, df, knots, boundary, degree)

    def basis(self, x):
        x = np.asarray(x, float)
        return self._ns(x) if self.kind == "ns" else self._bs(x)

    # -- natural cubic spline (splines::ns) -------------------------------------
    def _ns(self, x):
        bk = self.boundary
        aknots = np.sort(np.concatenate([np.repeat(bk, 4), self.knots]))
        nb = len(aknots) - 4
        basis = np.zeros((len(x), nb))
        ol, orr = x < bk[0], x > bk[1]
        inside = ~(ol | orr)
        for mask, piv in ((ol, bk[0]), (orr, bk[1])):
            if mask.any():
                tt = spline_design(aknots, [piv, piv], 4, [0, 1])
                xl = np.column_stack([np.ones(mask.sum()), x[mask] - piv])
                basis[mask] = xl @ tt
        if inside.any():
            basis[inside] = spline_design(aknots, x[inside], 4)
        const = spline_design(aknots, bk, 4, [2, 2])
        const, basis = const[:, 1:], basis[:, 1:]     # intercept = FALSE
        qr, qraux = _linpack_qr(const.T)
        return _qr_qty(qr, qraux, basis.T).T[:, 2:]

    # -- B-spline (splines::bs) ---------------------------------------------------
    def _bs(self, x):
        bk, deg = self.boundary, self.degree
        order = deg + 1
        aknots = np.sort(np.concatenate([np.repeat(bk, order), self.knots]))
        nb = len(aknots) - order
        basis = np.zeros((len(x), nb))
        ol, orr = x < bk[0], x > bk[1]
        inside = ~(ol | orr)
        if (ol | orr).any():
            derivs = np.arange(deg + 1)
            scalef = np.array([np.prod(np.arange(1, d + 1)) for d in derivs], float)
            e = 0.25
            for mask, piv in ((ol, (1 - e) * bk[0] + e * aknots[order]),
                              (orr, (1 - e) * bk[1] + e * aknots[len(aknots) - order - 1])):
                if mask.any():
                    xl = np.column_stack([np.ones(mask.sum())] +
                                         [(x[mask] - piv) ** d for d in range(1, deg + 1)])
                    tt = spline_design(aknots, np.repeat(piv, order), order, derivs)
                    basis[mask] = xl @ (tt / scalef[:, None])
        if inside.any():
            basis[inside] = spline_design(aknots, x[inside], order)
        return basis[:, 1:]                            # intercept = FALSE
