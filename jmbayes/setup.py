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


def ms_setup(data: pd.DataFrame, timevars, statusvars, transitionmat, id: str = "id",
             covs=None) -> pd.DataFrame:
    """Wide multi-state data -> one row per patient and possible transition, as
    ms_setup() in JMbayes2 (and mstate::msprep).

    data:          one row per patient
    timevars:      per state, the column with the time it was entered (None for the
                   starting state)
    statusvars:    per state, the column saying whether it was entered (0/1; None for the
                   starting state)
    transitionmat: states x states, with the transition number where a transition is
                   possible and NaN elsewhere (e.g. illness-death:
                   [[nan, 1, 2], [nan, nan, 3], [nan, nan, nan]])
    covs:          patient columns to copy to every row
    Returns columns id, from_state, to_state, transition, Tstart, Tstop, time, status
    (+ covs). Fit with Event("Tstop", "status", ..., start="Tstart", strata="transition",
    by_strata=True).
    """
    import numpy as np
    tm = np.asarray(transitionmat, float)
    S = tm.shape[0]
    n = len(data)
    times = np.full((n, S), np.nan)
    status = np.full((n, S), np.nan)
    for k, (tv, sv) in enumerate(zip(timevars, statusvars)):
        if tv is not None:
            times[:, k] = data[tv].astype(float).to_numpy()
        if sv is not None:
            status[:, k] = data[sv].astype(float).to_numpy()
    ids = data[id].to_numpy()
    rows = []
    _ms_prep(times, status, ids, np.zeros(n), np.zeros(n, int), tm, np.arange(1, S + 1),
             rows)
    out = pd.DataFrame(rows, columns=[id, "from_state", "to_state", "transition", "Tstart",
                                      "Tstop", "status"])
    out["time"] = out["Tstop"] - out["Tstart"]
    out = out[[id, "from_state", "to_state", "transition", "Tstart", "Tstop", "time",
               "status"]]
    out = out.sort_values([id, "Tstart", "from_state", "to_state"],
                          kind="mergesort").reset_index(drop=True)
    for c in ("from_state", "to_state", "transition", "status"):
        out[c] = out[c].astype(int)
    if covs:
        out = out.merge(data[[id, *covs]], on=id, how="left")
    return out


def _ms_prep(times, status, ids, t0, state, tm, orig, rows):
    """ms_prepdat() in JMbayes2: transitions out of the starting state, then recurse on
    the remaining states."""
    import numpy as np
    while times.shape[0] and tm.size:
        has = ~np.isnan(tm)
        starts = np.where(has.sum(0) == 0)[0]
        absorbing = np.where(has.sum(1) == 0)[0]
        s = starts[0]
        subj = np.where(state == s)[0]
        to2 = np.where(has[s])[0]
        rmv = []
        new_state, new_t = state.copy(), t0.copy()
        if len(subj) and len(to2):
            Tstart = t0[subj]
            Tstop = times[np.ix_(subj, to2)].copy()
            Tstop[Tstop <= Tstart[:, None]] = np.inf
            st = status[np.ix_(subj, to2)]
            mintime = Tstop.min(1)
            with np.errstate(divide="ignore", invalid="ignore"):
                hlp = Tstop * (1.0 / st)
            hlp[(Tstop == 0) & (st == 0)] = np.inf
            hlp[np.isnan(hlp)] = np.inf
            nxt = hlp.min(1)
            cens = np.isinf(nxt)
            nxt[cens] = mintime[cens]
            nstate = hlp.argmin(1)
            onehot = np.zeros((len(subj), len(to2)))
            onehot[~cens, nstate[~cens]] = 1.0
            absorbed = (~cens) & np.isin(to2[nstate], absorbing)
            for i, p in enumerate(subj):
                for c, k in enumerate(to2):
                    rows.append((ids[p], orig[s], orig[k], tm[s, k], Tstart[i], nxt[i],
                                 onehot[i, c]))
            rmv = list(subj[cens | absorbed])
            new_state[subj[~cens]] = to2[nstate[~cens]]
            new_t[subj[~cens]] = nxt[~cens]
        keep = np.setdiff1d(np.arange(times.shape[0]), rmv)
        times, status, ids = times[keep], status[keep], ids[keep]
        new_state, new_t = new_state[keep], new_t[keep]
        idx = np.cumsum(np.where(np.arange(tm.shape[0]) == s, 0, 1)) - 1
        state = idx[new_state]
        t0 = new_t
        times = np.delete(times, s, 1)
        status = np.delete(status, s, 1)
        tm = np.delete(np.delete(tm, s, 0), s, 1)
        orig = np.delete(orig, s)
