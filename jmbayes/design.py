"""Model terms as Python objects, and the design matrices they produce.

Terms combine with operators, like pandas / polars expressions:

    year, sex, age = var("year"), var("sex"), var("age")
    year * sex + age        # main effects of year and sex, their interaction, and age
    year + (year & sex)     # & is the interaction alone
    year.ns(2) * sex        # natural cubic spline of year (2 df), crossed with sex
    var("serBilir").log()   # a log-transformed response

Anywhere a term is expected you can also pass a column name, a list of names, or an
R-style string such as "year * sex" (see formula.py).

Column names and their order follow R's model.matrix(), so results can be compared with
JMbayes2 term by term: numeric columns keep their name, a categorical column contributes
one dummy per non-reference level ("sexfemale"), interactions join names with ":", and
spline columns are "ns(year, 2)1", "ns(year, 2)2", ...
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- factors
@dataclass(frozen=True)
class Column:
    """A data column used as is (numeric) or as dummies (categorical)."""
    name: str

    @property
    def variables(self):
        return (self.name,)


@dataclass(frozen=True)
class SplineBasis:
    """A spline basis of a numeric column; knots are fixed on the fitting data."""
    var: str
    df: int
    kind: str = "ns"            # "ns" natural cubic, "bs" cubic B-spline
    named: bool = False         # written "ns(x, df = 2)" in an R formula (names only)
    knots: tuple | None = None  # interior knots given explicitly (else quantiles)
    boundary: tuple | None = None

    @property
    def variables(self):
        return (self.var,)

    @property
    def label(self):
        if self.knots is not None or self.boundary is not None:
            f = lambda v: ", ".join(f"{x:g}" for x in v)  # noqa: E731
            parts = [self.var]
            if self.knots is not None:
                parts.append(f"knots = c({f(self.knots)})")
            if self.boundary is not None:
                parts.append(f"Boundary.knots = c({f(self.boundary)})")
            return f"{self.kind}({', '.join(parts)})"
        df = f"df = {self.df}" if self.named else str(self.df)
        return f"{self.kind}({self.var}, {df})"


Factor = Column | SplineBasis


# ---------------------------------------------------------------- terms
@dataclass(frozen=True)
class Terms:
    """A sum of terms; each term is a product (interaction) of factors."""
    terms: tuple = ()
    intercept: bool = True

    # -- operators ------------------------------------------------------------
    def __add__(self, other):
        other = as_terms(other)
        out = list(self.terms)
        for t in other.terms:
            if t not in out:
                out.append(t)
        return Terms(tuple(out), self.intercept)

    __radd__ = lambda self, other: as_terms(other) + self  # noqa: E731

    def __and__(self, other):
        """Interaction only (R's ':')."""
        other = as_terms(other)
        prods = []
        for a, b in itertools.product(self.terms, other.terms):
            t = tuple(dict.fromkeys(a + b))
            if t not in prods:
                prods.append(t)
        return Terms(tuple(prods), self.intercept)

    def __mul__(self, other):
        """Main effects plus interaction (R's '*')."""
        other = as_terms(other)
        return self + other + (self & other)

    __rmul__ = lambda self, other: as_terms(other) * self  # noqa: E731

    def __repr__(self):
        body = " + ".join(":".join(_flabel(f) for f in t) for t in self.terms) or "1"
        return f"Terms({body})"

    # -- suffix methods on a single column ------------------------------------------
    def log(self) -> "Response":
        return Response(_name_of(self), "log")

    def sqrt(self) -> "Response":
        return Response(_name_of(self), "sqrt")

    def ns(self, df: int) -> "Terms":
        return ns(self, df)

    def bs(self, df: int) -> "Terms":
        return bs(self, df)

    # -- information -------------------------------------------------------------
    def variables(self) -> list[str]:
        """Data columns needed by these terms."""
        return sorted({v for t in self.terms for f in t for v in f.variables})

    def without_intercept(self) -> "Terms":
        return Terms(self.terms, False)

    # -- design matrix ------------------------------------------------------------
    def build(self, df: pd.DataFrame, levels: dict):
        """Design matrix for df. `levels` remembers categories and spline knots from the
        first call (the fitting data) and reuses them for new data.

        Returns (matrix, column names, column info); column info lists, per column, its
        components (e.g. ("year", "sexfemale")) and the data columns it uses. Terms are
        ordered like R: main effects, then two-way interactions, and so on."""
        cols, names, info = [], [], []
        if self.intercept:
            cols.append(np.ones(len(df)))
            names.append("(Intercept)")
            info.append(ColumnInfo((), ()))
        for term in sorted(self.terms, key=len):
            pieces = [_factor_columns(df, f, levels) for f in term]
            for combo in itertools.product(*pieces):
                val = np.ones(len(df))
                for _, v in combo:
                    val = val * v
                cols.append(val)
                comps = tuple(n for n, _ in combo)
                names.append(":".join(comps))
                info.append(ColumnInfo(comps, tuple(v for f in term for v in f.variables)))
        M = np.column_stack(cols) if cols else np.zeros((len(df), 0))
        return M, names, info


@dataclass(frozen=True)
class ColumnInfo:
    components: tuple
    variables: tuple


def _flabel(f):
    return f.label if isinstance(f, SplineBasis) else f.name


# ---------------------------------------------------------------- constructors
def var(name: str) -> Terms:
    """A data column as a model term."""
    return Terms(((Column(name),),))


def ns(x, df: int | None = None, knots=None, boundary=None) -> Terms:
    """Natural cubic spline of a numeric column (same basis as R's splines::ns). Give
    df (knots at quantiles of the fitting data), or the interior knots and optionally
    the boundary knots, as ns(x, knots = c(3, 6, 9), Boundary.knots = c(0, 14.5))."""
    return Terms(((_spline(x, df, knots, boundary, "ns"),),))


def bs(x, df: int | None = None, knots=None, boundary=None) -> Terms:
    """Cubic B-spline of a numeric column (same basis as R's splines::bs)."""
    return Terms(((_spline(x, df, knots, boundary, "bs"),),))


def _spline(x, df, knots, boundary, kind):
    if knots is None:
        if df is None:
            raise ValueError(f"{kind}(): give df or knots")
        return SplineBasis(_name_of(x), int(df), kind,
                           boundary=None if boundary is None else tuple(map(float, boundary)))
    knots = tuple(map(float, knots))
    n = len(knots) + (1 if kind == "ns" else 3)
    if df is not None and int(df) != n:
        raise ValueError(f"{kind}(): df = {df} does not match {len(knots)} interior knots")
    return SplineBasis(_name_of(x), n, kind, knots=knots,
                       boundary=None if boundary is None else tuple(map(float, boundary)))


def _name_of(x):
    if isinstance(x, str):
        return x
    if isinstance(x, Terms) and len(x.terms) == 1 and len(x.terms[0]) == 1 \
            and isinstance(x.terms[0][0], Column):
        return x.terms[0][0].name
    raise TypeError(f"expected a column name or var(...), got {x!r}")


def as_terms(x) -> Terms:
    """Accept Terms, a column name or R-style string ("year * sex"), a tuple of names
    (interaction), a list of these, or 1 (intercept only)."""
    if isinstance(x, Terms):
        return x
    if isinstance(x, str):
        from jmbayes.formula import parse_rhs
        return Terms(parse_rhs(x).terms)
    if isinstance(x, int) and not isinstance(x, bool) and x == 1:
        return Terms(())
    if isinstance(x, tuple):
        out = as_terms(x[0])
        for y in x[1:]:
            out = out & as_terms(y)
        return out
    if isinstance(x, (list, set)) or x is None:
        out = Terms(())
        for y in (x or []):
            out = out + as_terms(y)
        return out
    raise TypeError(f"cannot use {x!r} as a model term; use a column name, var(), ns(), "
                    "bs(), a tuple for an interaction, or a list of these")


# ---------------------------------------------------------------- columns
def _levels(series: pd.Series) -> list:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return list(series.cat.categories)
    if series.dtype == bool:
        return [False, True]
    return sorted(series.dropna().unique())


def _is_categorical(series: pd.Series) -> bool:
    return isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == bool \
        or series.dtype == object or pd.api.types.is_string_dtype(series.dtype)


def _factor_columns(df: pd.DataFrame, f, levels: dict):
    """Columns (name, values) contributed by one factor (treatment coding for categories)."""
    if isinstance(f, SplineBasis):
        from jmbayes.splines import SplineSpec
        if f not in levels:                      # knots fixed on the fitting data
            levels[f] = SplineSpec.fit(f.kind, df[f.var].to_numpy(float), f.df,
                                       knots=f.knots, boundary=f.boundary)
        B = levels[f].basis(df[f.var].to_numpy(float))
        return [(f"{f.label}{j + 1}", B[:, j]) for j in range(B.shape[1])]
    s = df[f.name]
    if _is_categorical(s):
        lv = levels.setdefault(f.name, _levels(s))
        return [(f"{f.name}{l}", (s == l).astype(float).to_numpy()) for l in lv[1:]]
    return [(f.name, s.astype(float).to_numpy())]


# ---------------------------------------------------------------- responses
_TRANSFORMS = {None: lambda x: x, "log": np.log, "sqrt": np.sqrt}


@dataclass(frozen=True)
class Response:
    """The measured marker, optionally transformed ('log' or 'sqrt')."""
    var: str
    transform: str | None = None

    def __post_init__(self):
        if self.transform not in _TRANSFORMS:
            raise ValueError(f"transform must be one of {list(_TRANSFORMS)}, got "
                             f"{self.transform!r}")

    @property
    def name(self):
        return self.var if self.transform is None else f"{self.transform}({self.var})"

    def values(self, df: pd.DataFrame) -> np.ndarray:
        s = df[self.var]
        if self.transform is None and _is_categorical(s):
            lv = _levels(s)                       # binary factor: first level = 0
            return (s != lv[0]).astype(float).to_numpy()
        return _TRANSFORMS[self.transform](s.astype(float).to_numpy())
