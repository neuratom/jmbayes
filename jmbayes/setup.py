"""Data preparation helpers, as JMbayes2's rc_setup() for recurrent events."""
from __future__ import annotations

import pandas as pd


def rc_setup(rc_data: pd.DataFrame, trm_data: pd.DataFrame, *, id: str = "id",
             status: str = "status", start: str = "start", stop: str = "stop",
             trm_cens_level=0, name_strata: str = "strata",
             name_status: str = "status") -> pd.DataFrame:
    """Combine recurrent-event and terminal-event data into one counting-process data
    set, as rc_setup() in JMbayes2.

    rc_data:  one row per risk interval of the recurrent event (id, start, stop, status)
    trm_data: one row per patient with the terminal event (id, stop, status); the value
              trm_cens_level of status means no terminal event
    Returns rows for the recurrent intervals (stratum "R") and, per patient, one row per
    kind of terminal event (strata "T<level>", from time 0 to the terminal time), with
    name_status the 0/1 event indicator. Fit with
    Event(stop, name_status, ..., start=start, strata=name_strata, recurrent="gap").
    """
    for col in (id, status, start, stop):
        if col not in rc_data:
            raise ValueError(f"the variable '{col}' is not present in 'rc_data'")
    for col in (id, status, stop):
        if col not in trm_data:
            raise ValueError(f"the variable '{col}' is not present in 'trm_data'")
    if set(rc_data[id]) != set(trm_data[id]):
        raise ValueError("the patients in both data sets do not match")
    if (rc_data[start] > rc_data[stop]).any():
        raise ValueError(f"'{stop}' cannot be smaller than '{start}' in the recurrent data")
    rc = rc_data.sort_values([id, start], kind="mergesort")
    trm = trm_data.sort_values(id, kind="mergesort")
    t_end = trm.set_index(id)[stop]
    if (rc[stop].to_numpy() > t_end.loc[rc[id]].to_numpy()).any():
        raise ValueError(f"'{stop}' in the recurrent data cannot be after '{stop}' in the "
                         "terminal data")
    levels = [v for v in pd.unique(trm[status]) if v != trm_cens_level]
    parts = []
    for lev in levels:
        d = trm.copy()
        d[name_status] = (trm[status] == lev).astype(int)
        d[start] = 0.0
        d[name_strata] = f"T{lev}"
        parts.append(d)
    r = rc.copy()
    r[name_strata] = "R"
    r[name_status] = rc[status].astype(int)
    out = pd.concat([*parts, r], ignore_index=True, sort=False)
    out[name_strata] = pd.Categorical(out[name_strata],
                                      categories=sorted(out[name_strata].unique()))
    return out.sort_values([id, name_strata, start], kind="mergesort").reset_index(drop=True)
