"""Input checks. The library does not prepare or impute data: anything it cannot use
as given is reported with a clear error, so nothing is dropped or filled silently.

The one thing allowed to be missing is a marker value in the visits data: in long format
an empty marker cell means "not measured at this visit" (markers are often measured at
different visits). Such rows are simply not observations of that marker.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class DataError(ValueError):
    """The data cannot be used as given; the message says what to fix."""


def _missing_report(df: pd.DataFrame, cols, what: str):
    cols = list(dict.fromkeys(cols))
    absent = [c for c in cols if c not in df.columns]
    if absent:
        raise DataError(f"{what}: columns not found: {absent}")
    n_na = df[cols].isna().sum()
    n_na = n_na[n_na > 0]
    if len(n_na):
        detail = ", ".join(f"{c} ({n} rows)" for c, n in n_na.items())
        raise DataError(f"{what} has missing values in: {detail}. jmbayes does not "
                        "impute; remove or fill these before fitting.")


def event_causes(col: pd.Series, censored):
    """Competing risks: the causes, in the column's category order (else sorted), without
    the censoring value. None for a 0/1 event."""
    if censored is None:
        return None
    levels = (list(col.cat.categories) if isinstance(col.dtype, pd.CategoricalDtype)
              else sorted(pd.unique(col.astype(str))))
    present = set(col.astype(str))
    return [str(v) for v in levels if str(v) != str(censored) and str(v) in present]


def check_patients(patients: pd.DataFrame, id_var: str, time: str, event: str,
                   covariates, what="patients data", censored=None, start=None,
                   strata=None):
    extra = [c for c in (start, strata) if c]
    _missing_report(patients, [id_var, time, event, *extra, *covariates], what)
    dup = patients[id_var][patients[id_var].duplicated()]
    if len(dup):
        raise DataError(f"{what} must have one row per patient; repeated ids: "
                        f"{_few(dup.unique())}")
    t = patients[time].astype(float)
    if (t <= 0).any():
        raise DataError(f"{what}: follow-up time '{time}' must be positive; "
                        f"{int((t <= 0).sum())} rows are <= 0")
    if start:
        if not pd.api.types.is_numeric_dtype(patients[start]):
            raise DataError(f"{what}: entry time '{start}' must be numeric")
        t0 = patients[start].astype(float)
        if (t0 < 0).any():
            raise DataError(f"{what}: entry time '{start}' must be >= 0; "
                            f"{int((t0 < 0).sum())} rows are negative")
        if (t0 >= t).any():
            raise DataError(f"{what}: entry time '{start}' must be before the follow-up "
                            f"time '{time}'; {int((t0 >= t).sum())} rows are not")
    if censored is not None:
        vals = set(patients[event].astype(str))
        if str(censored) not in vals and len(vals) < 2:
            raise DataError(f"{what}: event column '{event}' has no value '{censored}'")
        if len(event_causes(patients[event], censored)) < 2:
            raise DataError(f"{what}: competing risks need at least two causes in "
                            f"'{event}' besides '{censored}'; for a single event, code "
                            "it 0/1 and leave out censored=")
        return
    if not set(pd.unique(patients[event])) <= {0, 1}:
        raise DataError(f"{what}: event column '{event}' must be coded 0/1 "
                        "(e.g. create it with (df.status != 'alive').astype(int)), or "
                        "name the causes and pass censored= for competing risks")


def check_visits(visits: pd.DataFrame, id_var: str, time_var: str, markers,
                 what="visits data", require_measured=True):
    """markers: list of (response, covariate columns, family)."""
    covs = [c for _, cols, _ in markers for c in cols if c != time_var]
    _missing_report(visits, [id_var, time_var, *covs], what)
    for resp, _, family in markers:
        if resp.var not in visits.columns:
            raise DataError(f"{what}: marker column '{resp.var}' not found")
        y = visits[resp.var].dropna()
        if y.empty and require_measured:
            raise DataError(f"{what}: marker '{resp.var}' has no measurements")
        if family in ("poisson", "negative_binomial"):
            if not pd.api.types.is_numeric_dtype(y):
                raise DataError(f"{what}: count marker '{resp.var}' must be numeric")
            if (y < 0).any() or (np.round(y) != y).any():
                raise DataError(f"{what}: {family} marker '{resp.var}' must be counts "
                                "(whole numbers >= 0)")
            if resp.transform is not None:
                raise DataError(f"{what}: a {family} marker is modelled on its own scale "
                                f"(log link); drop the {resp.transform}() transform")
        elif family == "binomial":
            vals = set(pd.unique(y))
            if pd.api.types.is_numeric_dtype(y) and not vals <= {0, 1}:
                raise DataError(f"{what}: binomial marker '{resp.var}' must be coded 0/1 "
                                f"(or be a two-category column), found {_few(sorted(vals))}")
            if len(vals) > 2:
                raise DataError(f"{what}: binomial marker '{resp.var}' must have two "
                                f"values, found {_few(sorted(vals, key=str))}")
        elif not pd.api.types.is_numeric_dtype(y):
            raise DataError(f"{what}: marker '{resp.var}' must be numeric")
        elif resp.transform == "log" and (y <= 0).any():
            raise DataError(f"{what}: log({resp.var}) needs positive values; "
                            f"{int((y <= 0).sum())} measurements are <= 0")
        elif resp.transform == "sqrt" and (y < 0).any():
            raise DataError(f"{what}: sqrt({resp.var}) needs non-negative values")


def check_same_patients(visits, patients, id_var):
    v, p = set(visits[id_var]), set(patients[id_var])
    if v - p:
        raise DataError(f"patients in the visits data but not in the patients data: "
                        f"{_few(sorted(v - p, key=str))}")
    if p - v:
        raise DataError(f"patients in the patients data without any visit: "
                        f"{_few(sorted(p - v, key=str))}")


def _few(xs, k=5):
    xs = list(xs)
    s = ", ".join(map(str, xs[:k]))
    return s + (f", ... ({len(xs)} in total)" if len(xs) > k else "")
