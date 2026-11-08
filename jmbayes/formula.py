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


_SPLINE_KN = re.compile(r"^(ns|bs)\((\w+),(.+)\)$")


def _num_vec(v):
    v = v.strip()
    m = re.match(r"^c\((.*)\)$", v)
    return [float(x) for x in (m.group(1) if m else v).split(",") if x.strip()]


def _factor(token: str) -> Terms:
    m = _SPLINE_KN.match(token)
    if m and ("=" in m.group(3)):
        # ns(x, k = c(...), B = c(...)) / knots = / Boundary.knots = / df =
        from jmbayes.assoc import _split_top
        kw = {}
        for a in _split_top(m.group(3), ","):
            k, v = (x.strip() for x in a.split("=", 1))
            if k in ("k", "knots"):
                kw["knots"] = _num_vec(v)
            elif k in ("B", "Boundary.knots", "Boundary"):
                kw["boundary"] = _num_vec(v)
            elif k == "df":
                kw["df"] = int(float(v))
            else:
                raise ValueError(f"unknown spline option {k!r} in {token!r}")
        from jmbayes.design import _spline
        return Terms(((_spline(m.group(2), kw.get("df"), kw.get("knots"),
                               kw.get("boundary"), m.group(1)),),))
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
    baseline hazard per stratum (start and strata are None otherwise). See
    parse_surv_full() for interval censoring and stratum-specific effects."""
    d = parse_surv_full(formula)
    return d["time"], d["event"], d["covariates"], d["start"], d["strata"]


def parse_surv_full(formula: str) -> dict:
    """All of a coxph()/survreg() formula that jm() uses:

    Surv(time, event) ~ x                     right censoring
    Surv(start, stop, event) ~ x              entry times / counting-process rows
    Surv(time, time2, event, type = "interval") ~ x   interval censoring (event codes
                                              0 right, 1 event, 2 left, 3 interval)
    ... + strata(s)                           a baseline hazard per stratum
    (x + z):strata(s)                         also effects of x and z per stratum
    """
    m = re.match(r"^\s*Surv\((.*?)\)\s*~\s*(.+)$", formula)
    args = [a.strip() for a in m.group(1).split(",")] if m else []
    kind = None
    if args and re.match(r"^type\s*=", args[-1]):
        kind = args.pop().split("=", 1)[1].strip().strip("'\"")
    if not m or not all(re.match(r"^\w+$", a) for a in args) or len(args) not in (2, 3):
        raise ValueError("formula should look like 'Surv(time, event) ~ covariates', "
                         "'Surv(start, stop, event) ~ covariates' or 'Surv(time, time2, "
                         f"event, type = \"interval\") ~ covariates', got {formula!r}")
    out = dict(start=None, time2=None, strata=None, by_strata=False)
    if kind not in (None, "interval", "counting", "right"):
        raise NotImplementedError(f"Surv(type = {kind!r}) is not supported "
                                  "(use 'interval' with event codes 0-3)")
    if kind == "interval":
        if len(args) != 3:
            raise ValueError("interval censoring: Surv(time, time2, event, type = 'interval')")
        out.update(time=args[0], time2=args[1], event=args[2])
    elif len(args) == 3:
        out.update(start=args[0], time=args[1], event=args[2])
    else:
        out.update(time=args[0], event=args[1])
    rhs = m.group(2).replace(" ", "")
    sm = set(re.findall(r"strata\((\w+)\)", rhs))
    if len(sm) > 1:
        raise NotImplementedError("only one strata variable is supported")
    if sm:
        s = sm.pop()
        out["strata"] = s
        parts = []
        for p in _split_plus(rhs):
            if p == f"strata({s})":
                continue
            if p.endswith(f":strata({s})"):
                out["by_strata"] = True
                p = p[: -len(f":strata({s})")]
                if p.startswith("(") and p.endswith(")"):
                    parts += _split_plus(p[1:-1])
                    continue
            if "strata(" in p:
                raise NotImplementedError(f"unsupported strata() term {p!r}")
            parts.append(p)
        rhs = "+".join(parts) or "1"
    out["covariates"] = parse_rhs(rhs).without_intercept()
    return out


def _split_plus(s: str) -> list:
    """Split at '+' outside parentheses."""
    out, depth, cur = [], 0, ""
    for ch in s:
        depth += ch == "("
        depth -= ch == ")"
        if ch == "+" and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    return [p for p in out + [cur] if p]


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
