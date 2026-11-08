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
                   strata=None, time2=None, recurrent=None):
    extra = [c for c in (start, strata) if c]
    _missing_report(patients, [id_var, time, event, *extra, *covariates], what)
    dup = patients[id_var][patients[id_var].duplicated()]
    if len(dup) and not start:
        raise DataError(f"{what} must have one row per patient; repeated ids: "
                        f"{_few(dup.unique())} (several rows per patient, e.g. for "
                        "covariates that change over time, need start= / "
                        "Surv(start, stop, event))")
    t = patients[time].astype(float)
    if time2:
        _check_interval(patients, time, time2, event, what)
        return
    if (t <= 0).any():
        raise DataError(f"{what}: follow-up time '{time}' must be positive; "
                        f"{int((t <= 0).sum())} rows are <= 0")
    if start and len(dup) and not recurrent:
        # counting-process rows of one patient must not overlap
        keys = [id_var] + ([strata] if strata else [])     # within a stratum
        d = patients[keys + [start, time]].astype({start: float, time: float})
        d = d.sort_values(keys + [start], kind="mergesort")
        prev = d.groupby(keys, observed=True)[time].shift()
        if (d[start] < prev - 1e-12).any():
            raise DataError(f"{what}: the intervals ({start}, {time}] of a patient "
                            "overlap; each row should cover the next stretch of follow-up")
    if recurrent:
        lev = pd.unique(patients[strata])
        if len(lev) < 1:
            raise DataError(f"{what}: strata column '{strata}' is empty")
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


def _check_interval(patients, time, time2, event, what):
    code = patients[event]
    if not set(pd.unique(code)) <= {0, 1, 2, 3}:
        raise DataError(f"{what}: with time2= the event column '{event}' must be coded "
                        "0 (right-censored), 1 (event), 2 (left-censored), 3 (interval)")
    t1 = patients[time].astype(float)
    if (t1 <= 0).any():
        raise DataError(f"{what}: '{time}' must be positive")
    iv = code == 3
    if iv.any():
        if time2 not in patients:
            raise DataError(f"{what}: column '{time2}' not found")
        t2 = patients.loc[iv, time2].astype(float)
        if t2.isna().any() or (t2 <= t1[iv]).any():
            raise DataError(f"{what}: for interval-censored rows ({event} = 3) '{time2}' "
                            f"must be after '{time}'")


def check_visits(visits: pd.DataFrame, id_var: str, time_var: str, markers,
                 what="visits data", require_measured=True):
    """markers: list of (response, covariate columns, family[, aux column])."""
    markers = [tuple(m) + (None,) * (4 - len(m)) for m in markers]
    covs = [c for _, cols, _, _ in markers for c in cols if c != time_var]
    _missing_report(visits, [id_var, time_var, *covs], what)
    for resp, _, _, a in markers:         # trials / censoring: needed where measured
        if isinstance(a, str):
            if a not in visits:
                raise DataError(f"{what}: column '{a}' not found")
            measured = visits[resp.var].notna() if resp.var in visits else True
            if visits.loc[measured, a].isna().any():
                raise DataError(f"{what}: '{a}' is missing on rows where '{resp.var}' "
                                "was measured")
    for resp, _, family, aux in markers:
        if resp.var not in visits.columns:
            raise DataError(f"{what}: marker column '{resp.var}' not found")
        y = visits[resp.var].dropna()
        if y.empty and require_measured:
            raise DataError(f"{what}: marker '{resp.var}' has no measurements")
        if family not in ("gaussian", "student_t", "censored_normal", "binomial") and \
                resp.transform is not None:
            raise DataError(f"{what}: a {family} marker is modelled on its own scale; "
                            f"drop the {resp.transform}() transform")
        if family in ("beta", "unit_lindley", "gamma"):
            if not pd.api.types.is_numeric_dtype(y):
                raise DataError(f"{what}: marker '{resp.var}' must be numeric")
            if family == "gamma" and (y <= 0).any():
                raise DataError(f"{what}: gamma marker '{resp.var}' must be positive")
            if family != "gamma" and ((y <= 0) | (y >= 1)).any():
                raise DataError(f"{what}: {family} marker '{resp.var}' must lie strictly "
                                "between 0 and 1")
            continue
        if family == "censored_normal":
            c = visits.loc[y.index, aux]
            if not set(pd.unique(c)) <= {0, 1, 2}:
                raise DataError(f"{what}: censoring column '{aux}' must be coded 0 "
                                "(observed), 1 (left-censored) or 2 (right-censored)")
            continue
        if family in ("binomial", "beta_binomial") and (aux is not None or
                                                         family == "beta_binomial"):
            n = (visits.loc[y.index, aux].astype(float) if isinstance(aux, str)
                 else pd.Series(1.0 if aux is None else float(aux), index=y.index))
            if not pd.api.types.is_numeric_dtype(y):
                raise DataError(f"{what}: marker '{resp.var}' must be the number of "
                                "successes")
            if (y < 0).any() or (np.round(y) != y).any() or (y > n).any() or \
                    (np.round(n) != n).any():
                raise DataError(f"{what}: {family} marker '{resp.var}' must be whole "
                                "numbers between 0 and the number of trials")
            continue
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
