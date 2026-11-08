"""How a marker enters the hazard: JMbayes2's functional forms.

A column of the association design ("Wlong") is

    U(t) * f1(eta_form1(t)) * f2(eta_form2(t)) * ...

where each eta_form is a functional form of the marker's linear predictor (value, slope,
velocity, acceleration, area, Delta, coefs), f is a transformation (expit, exp, log, ...)
and U(t) is an optional interaction with covariates of the survival data, evaluated at
time t (so an interaction with a function of time, e.g. ns(year, ...), gives a
time-varying association). This is create_Wlong() in JMbayes2.

Python:

    from jmbayes import assoc as A
    jmb.Marker(..., association=[A.value(), A.slope(eps=0.01, direction="backward"),
                                 A.area(time_window=2), A.vexpit(A.value()),
                                 A.value() & var("sex"),         # value(y):sex
                                 A.value() * ns(year, knots=[3, 6], boundary=[0, 15]),
                                 A.Dexp(A.slope())])             # exp(value) * slope

R strings (jm(functional_forms = ~ ...)):  parse("~ value(y) + vexpit(value(y)) +
    value(y):sex + slope(y, eps = 0.01) + Dexp(slope(y))", responses=["y"])
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from jmbayes.design import Terms, as_terms

FORMS = ("value", "slope", "velocity", "acceleration", "area", "Delta", "coefs")
TRANSFORMS = ("identity", "abs", "expit", "exp", "dexpit", "log", "log2", "log10", "sqrt",
              "poly2", "poly3", "poly4", "poly2_expit", "poly3_expit", "poly4_expit")


# ------------------------------------------------------------------ specification
@dataclass(frozen=True)
class Form:
    """One functional form of a marker, with its options (as value(), slope(), ... in
    JMbayes2). ie_time: column of the survival data with an intervention time after
    which the form is frozen at its value at that time (IE_time in JMbayes2)."""
    form: str = "value"
    eps: float = 0.001
    direction: str = "both"
    time_window: float | None = None
    standardise: bool = True
    zero_ind_X: tuple = ()
    zero_ind_Z: tuple = ()
    ie_time: str | None = None

    def __post_init__(self):
        if self.form not in FORMS:
            raise NotImplementedError(f"functional form {self.form}() (supported: {FORMS})")
        if self.direction not in ("both", "backward"):
            raise ValueError("direction must be 'both' or 'backward'")

    def label(self, resp):
        args = [resp]
        if self.form in ("slope", "velocity") and self.eps != 0.001:
            args.append(f"eps = {self.eps:g}")
        if self.form in ("slope", "velocity") and self.direction != "both":
            args.append(f'direction = "{self.direction}"')
        if self.time_window is not None:
            args.append(f"time_window = {self.time_window:g}")
        return f"{self.form}({', '.join(args)})"


@dataclass(frozen=True)
class Assoc:
    """A column (or, with a categorical interaction, columns) of the association design:
    a product of transformed forms times optional covariate terms."""
    factors: tuple = ((Form(), "identity"),)          # ((Form, transform), ...)
    by: Terms | None = None                           # interaction with covariates

    # -- composition -------------------------------------------------------------
    def __and__(self, other):
        """Interaction only (R's ':'): with another form (product) or with covariates."""
        if isinstance(other, Assoc):
            by = _join_terms(self.by, other.by)
            return Assoc(self.factors + other.factors, by)
        return Assoc(self.factors, _join_terms(self.by, as_terms(other).without_intercept()))

    def __mul__(self, other):
        """Crossing (R's '*'): the main term(s) and the interaction, as a list."""
        out = [self]
        if isinstance(other, Assoc):
            out.append(other)
        out.append(self & other)
        return AssocList(out)

    def label(self, resp):
        parts = [_tlabel(f.label(resp), t) for f, t in self.factors]
        return ":".join(parts)


class AssocList(list):
    """Several association terms (from '*')."""
    def __mul__(self, other):
        return AssocList([x for a in self for x in (a * other)])

    def __and__(self, other):
        return AssocList([a & other for a in self])


def _join_terms(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return Terms(tuple(t1 + t2 for t1 in a.terms for t2 in b.terms), False)


_TLAB = {"expit": "vexpit", "exp": "vexp", "abs": "vabs", "log": "vlog", "log2": "vlog2",
         "log10": "vlog10", "sqrt": "vsqrt", "dexpit": "Dexpit", "poly2": "poly2",
         "poly3": "poly3", "poly4": "poly4", "poly2_expit": "poly2(vexpit",
         "poly3_expit": "poly3(vexpit", "poly4_expit": "poly4(vexpit"}


def _tlabel(lab, t):
    if t == "identity":
        return lab
    s = _TLAB[t]
    return f"{s}({lab}))" if "(" in s else f"{s}({lab})"


def _form(name):
    def make(eps=0.001, direction="both", time_window=None, standardise=True,
             zero_ind_X=(), zero_ind_Z=(), ie_time=None):
        return Assoc(((Form(name, eps, direction, time_window, standardise,
                            tuple(zero_ind_X), tuple(zero_ind_Z), ie_time), "identity"),))
    make.__name__ = name
    make.__doc__ = f"{name}() functional form (options as in JMbayes2)."
    return make


value, slope, velocity, acceleration, area, Delta, coefs = (_form(f) for f in FORMS)


def _transform(t):
    def apply(a: Assoc) -> Assoc:
        if len(a.factors) != 1 or a.factors[0][1] != "identity":
            raise ValueError("transformations apply to a single untransformed form")
        return Assoc(((a.factors[0][0], t),), a.by)
    return apply


vexpit, vexp, vabs, vlog, vlog2, vlog10, vsqrt, poly2, poly3, poly4 = (
    _transform(t) for t in ("expit", "exp", "abs", "log", "log2", "log10", "sqrt", "poly2",
                            "poly3", "poly4"))


def Dexp(a: Assoc) -> Assoc:
    """d/dt exp(eta(t)) = exp(value) * slope (expand_Dexps() in JMbayes2)."""
    f = a.factors[0][0]
    return Assoc(((replace(f, form="value"), "exp"), (_as_slope(f), "identity")), a.by)


def Dexpit(a: Assoc) -> Assoc:
    """d/dt expit(eta(t)) = expit(value) (1 - expit(value)) * slope. (JMbayes2 0.6-5 maps
    the name to expit(value) * slope, see NOTES.md; jmbayes uses the derivative.)"""
    f = a.factors[0][0]
    return Assoc(((replace(f, form="value"), "dexpit"), (_as_slope(f), "identity")), a.by)


def _as_slope(f):
    return f if f.form in ("slope", "velocity") else replace(f, form="slope")


def as_assoc_list(spec) -> list:
    """Normalise Marker(association=...) to a list of Assoc."""
    if spec is None:
        return [value()]
    if isinstance(spec, (str, Assoc)):
        spec = [spec]
    out = []
    for a in spec:
        if isinstance(a, str):
            if a not in FORMS:
                raise NotImplementedError(f"association {a!r} not supported (use one of "
                                          f"{FORMS}, or jmbayes.assoc for options)")
            out.append(_form(a)())
        elif isinstance(a, Assoc):
            out.append(a)
        elif isinstance(a, (list, tuple)):
            out += as_assoc_list(list(a))
        else:
            raise TypeError(f"association: expected a form name or an Assoc, got {a!r}")
    if not out:
        raise ValueError("give at least one association")
    return list(dict.fromkeys(out))


# ------------------------------------------------------------------ transformations
def np_transform(t, x):
    if t == "identity":
        return x
    if t == "abs":
        return np.abs(x)
    if t == "expit":
        return 1.0 / (1.0 + np.exp(-x))
    if t == "exp":
        return np.exp(x)
    if t == "dexpit":
        p = 1.0 / (1.0 + np.exp(-x))
        return p * (1 - p)
    if t == "log":
        return np.log(x)
    if t == "log2":
        return np.log2(x)
    if t == "log10":
        return np.log10(x)
    if t == "sqrt":
        return np.sqrt(x)
    if t.startswith("poly"):
        k = int(t[4])
        base = 1.0 / (1.0 + np.exp(-x)) if t.endswith("expit") else x
        return base ** k
    raise NotImplementedError(t)


def torch_transform(t, x):
    if t == "identity":
        return x
    if t == "abs":
        return torch.abs(x)
    if t == "expit":
        return torch.sigmoid(x)
    if t == "exp":
        return torch.exp(x)
    if t == "dexpit":
        p = torch.sigmoid(x)
        return p * (1 - p)
    if t == "log":
        return torch.log(x)
    if t == "log2":
        return torch.log2(x)
    if t == "log10":
        return torch.log10(x)
    if t == "sqrt":
        return torch.sqrt(x)
    if t.startswith("poly"):
        k = int(t[4])
        base = torch.sigmoid(x) if t.endswith("expit") else x
        return base ** k
    raise NotImplementedError(t)


# ------------------------------------------------------------------ R strings
def parse(text: str, responses) -> dict:
    """jm(functional_forms = ~ ...) -> {response: [Assoc, ...]}; responses not mentioned
    are left out (the caller gives them value())."""
    from jmbayes import formula as fm
    s = text.split("~", 1)[-1].replace("\n", " ").strip()
    norm = {r.replace(" ", ""): r for r in responses}
    out = {}
    # model.matrix() order: main terms first, then two-way interactions, ... (stable)
    for term in sorted(_expand(_split_top(s, "+")), key=lambda t: len(_r_factors(t))):
        forms, covs = [], []
        resp_of = None
        for fac in term:
            a, r = _parse_form(fac, norm)
            if a is None:
                if any(k in fac.replace(" ", "") for k in norm):
                    raise NotImplementedError(f"functional form {fac!r} is not supported "
                                              f"(forms: {', '.join(FORMS)}; "
                                              "transformations: vexpit, Dexpit, vexp, "
                                              "Dexp, vabs, vlog, vlog2, vlog10, vsqrt, "
                                              "poly2, poly3, poly4)")
                covs.append(fac)
            else:
                if resp_of is not None and r != resp_of:
                    raise NotImplementedError("products of forms of different outcomes")
                resp_of = r
                forms.append(a)
        if not forms:
            continue                           # covariate main effects: not associations
        a = forms[0]
        for b in forms[1:]:
            a = a & b
        if covs:
            a = a & fm.parse_rhs(":".join(covs)).without_intercept()
        out.setdefault(resp_of, []).append(a)
    return out


def _r_factors(term):
    """Factors of a term as R counts them for its order (Dexp(...) counts two)."""
    return [f for fac in term for f in ([fac, fac] if fac.strip().startswith(("Dexp(", "Dexpit("))
                                        else [fac])]


def _split_top(s, sep):
    out, depth, cur = [], 0, ""
    for ch in s:
        depth += ch in "([{"
        depth -= ch in ")]}"
        if ch == sep and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _expand(terms):
    """R's '*' and ':' -> list of terms, each a list of factors."""
    out = []
    for t in terms:
        crossed = []
        for chunk in _split_top(t, "*"):
            inter = _split_top(chunk, ":")
            crossed = crossed + [inter] + [c + inter for c in crossed]
        for c in crossed:
            if c and c not in out:
                out.append(c)
    return out


_TFUN = {"vexpit": "expit", "vexp": "exp", "vabs": "abs", "vlog": "log", "vlog2": "log2",
         "vlog10": "log10", "vsqrt": "sqrt", "poly2": "poly2", "poly3": "poly3",
         "poly4": "poly4"}


def _parse_form(fac, norm):
    """'slope(log(serBilir), eps = 0.01)' -> (Assoc, response) or (None, None)."""
    m = re.match(r"^(\w+)\((.*)\)$", fac.strip())
    if not m:
        return None, None
    fun, inner = m.group(1), m.group(2)
    if fun in _TFUN or fun in ("Dexp", "Dexpit"):
        a, r = _parse_form(inner, norm)
        if a is None:
            return None, None
        if fun == "Dexp":
            return Dexp(a), r
        if fun == "Dexpit":
            return Dexpit(a), r
        if fun.startswith("poly") and a.factors[0][1] == "expit":
            return Assoc(((a.factors[0][0], f"{fun}_expit"),), a.by), r
        return _transform(_TFUN[fun])(a), r
    if fun not in FORMS:
        return None, None
    args = _split_top(inner, ",")
    resp = args[0].replace(" ", "")
    if resp not in norm:
        raise KeyError(f"functional_forms: unknown outcome {args[0]!r}; outcomes are "
                       f"{list(norm.values())}")
    kw = {}
    for a in args[1:]:
        k, v = (x.strip() for x in a.split("=", 1))
        v = v.strip("'\"")
        if k == "eps":
            kw["eps"] = float(v)
        elif k == "direction":
            kw["direction"] = v
        elif k == "time_window":
            kw["time_window"] = None if v == "NULL" else float(v)
        elif k == "standardise":
            kw["standardise"] = v.upper() in ("TRUE", "T")
        elif k == "IE_time":
            kw["ie_time"] = v
        else:
            raise NotImplementedError(f"option {k} of {fun}()")
    return _form(fun)(**kw), norm[resp]
