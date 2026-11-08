"""R-style formula strings, for people who prefer them (or come from JMbayes2 in R).

Every string is parsed once into the same Python objects that the Python interface uses
(design.Terms, design.Response), so both interfaces fit exactly the same model.

    parse_rhs("year * sex + ns(age, 2)")   -> Terms
    parse_response("log(serBilir)")        -> Response
    parse_random("~ year | id")            -> (Terms, "id", uncorrelated=False)
    parse_surv("Surv(years, status2) ~ sex") -> ("years", "status2", Terms, None, None)
    parse_surv("Surv(entry, years, status2) ~ sex + strata(drug)")
                                           -> ("years", "status2", Terms, "entry", "drug")
"""
from __future__ import annotations

import re

from jmbayes.design import Response, SplineBasis, Terms, var

_SPLINE = re.compile(r"^(ns|bs)\((\w+),(df=)?(\d+)\)$")
_FUN = re.compile(r"^(\w+)\((\w+)\)$")
_NAME = re.compile(r"^[A-Za-z_.][\w.]*$")


def _factor(token: str) -> Terms:
    m = _SPLINE.match(token)
    if m:
        kind, x, named, df = m.group(1), m.group(2), bool(m.group(3)), int(m.group(4))
        return Terms(((SplineBasis(x, df, kind, named),),))
    if not _NAME.match(token):
        raise ValueError(f"cannot read the term {token!r}; use column names, ns(x, df), "
                         "bs(x, df), '+', '*' and ':'")
    return var(token)


def parse_rhs(rhs: str) -> Terms:
    """Right-hand side of an R formula: terms joined by '+', '*' (crossing) and ':'
    (interaction only). '1' is the intercept; '0' or '-1' removes it."""
    s = rhs.split("~", 1)[-1].replace(" ", "")
    intercept = True
    if s.endswith("-1") or s.startswith("0+") or s == "0":
        intercept = False
        s = s.removesuffix("-1").removeprefix("0+")
        s = "" if s == "0" else s
    out = Terms(())
    for part in filter(None, s.split("+")):
        if part == "1":
            continue
        crossed = None
        for chunk in part.split("*"):
            inter = None
            for tok in chunk.split(":"):
                f = _factor(tok)
                inter = f if inter is None else inter & f
            crossed = inter if crossed is None else crossed * inter
        out = out + crossed
    return Terms(out.terms, intercept)


def parse_response(lhs: str) -> Response:
    """Left-hand side: a column, optionally inside log() or sqrt()."""
    lhs = lhs.replace(" ", "")
    m = _FUN.match(lhs)
    if m:
        return Response(m.group(2), m.group(1))
    if not _NAME.match(lhs):
        raise ValueError(f"cannot read the response {lhs!r}")
    return Response(lhs)


def parse_formula(formula: str) -> tuple[Response, Terms]:
    """'log(serBilir) ~ year * sex' -> (Response, Terms)."""
    if "~" not in formula:
        raise ValueError(f"formula needs a '~': {formula!r}")
    lhs, rhs = formula.split("~", 1)
    return parse_response(lhs), parse_rhs(rhs)


def parse_random(random: str) -> tuple[Terms, str, bool]:
    """'~ year | id' -> (Terms, id column, uncorrelated); '||' means uncorrelated."""
    m = re.match(r"^\s*~?\s*(.+?)\s*(\|\|?)\s*(\w+)\s*$", random)
    if not m:
        raise ValueError(f"random should look like '~ year | id', got {random!r}")
    return parse_rhs(m.group(1)), m.group(3), m.group(2) == "||"


def parse_surv(formula: str):
    """'Surv(years, status2) ~ sex' -> (time, event, Terms without intercept, start,
    strata); 'Surv(start, stop, event)' gives delayed entry and a 'strata(x)' term a
    baseline hazard per stratum (start and strata are None otherwise)."""
    m = re.match(r"^\s*Surv\(\s*(\w+)\s*,\s*(\w+)\s*(?:,\s*(\w+)\s*)?\)\s*~\s*(.+)$",
                 formula)
    if not m:
        raise ValueError("formula should look like 'Surv(time, event) ~ covariates' or "
                         f"'Surv(start, stop, event) ~ covariates', got {formula!r}")
    if m.group(3):
        start, time, event = m.group(1), m.group(2), m.group(3)
    else:
        start, time, event = None, m.group(1), m.group(2)
    rhs, strata = m.group(4).replace(" ", ""), None
    sm = re.findall(r"strata\((\w+)\)", rhs)
    if len(sm) > 1:
        raise NotImplementedError("only one strata() term is supported")
    if sm:
        strata = sm[0]
        parts = [p for p in rhs.split("+") if p != f"strata({strata})"]
        if any("strata(" in p for p in parts):
            raise NotImplementedError("strata() can only be added as its own term")
        rhs = "+".join(parts) or "1"
    return time, event, parse_rhs(rhs).without_intercept(), start, strata


def parse_functional_forms(text: str) -> dict:
    """'~ value(log(serBilir)) + slope(log(serBilir))' -> {"log(serBilir)": [...]}."""
    out = {}
    for term in [t.strip() for t in text.split("~", 1)[-1].split("+") if t.strip()]:
        m = re.match(r"^(\w+)\((.+)\)$", term.replace(" ", ""))
        if not m or ":" in term or "*" in term:
            raise NotImplementedError(f"functional form term {term!r} is not supported yet "
                                      "(supported: value(), slope(), area() of an outcome)")
        out.setdefault(m.group(2), []).append(m.group(1))
    return out
