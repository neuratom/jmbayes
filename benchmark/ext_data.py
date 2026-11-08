"""Data sets derived from PBC for the comparisons of the features added in jmbayes 0.7
(benchmark/run_ext.py and run_ext2_R.R read the same CSV files):

  tvc_surv.csv    counting-process rows (one per interval between visits) with a
                  time-varying covariate: edema at the start of the interval
  icens_surv.csv  interval censoring: an event is only known to have happened between
                  the last visit before it and the recorded time (status 3);
                  others right-censored (status 0)
  fam_visits.csv  extra markers: alb_p = albumin / 9 and bili_p = serBilir / (serBilir
                  + 1) (in (0, 1)), lalk = log(alkaline)
                  censored below its 20% quantile (lalk_cens = 1), count2 = ascites +
                  spiders (out of 2), count3 = ascites + hepatomegaly + spiders (out of 3)

Usage: python ext_data.py  (writes benchmark/data_ext/)
"""
from pathlib import Path

import numpy as np
import pandas as pd

import jmbayes as jmb

OUT = Path(__file__).parent / "data_ext"


def main():
    OUT.mkdir(exist_ok=True)
    visits, patients = jmb.datasets.pbc2()
    visits, patients = visits.copy(), patients.copy()
    patients["entry"] = 0.2 * patients["years"]
    # --- time-varying edema
    rows = []
    for pid, v in visits.groupby("id", sort=False):
        p = patients.loc[patients.id == pid].iloc[0]
        t = v["year"].to_numpy(float)
        ed = (v["edema"].astype(str) != "No edema").astype(int).to_numpy()
        keep = t < p.years
        t, ed = t[keep], ed[keep]
        stops = np.append(t[1:], p.years)
        for k in range(len(t)):
            rows.append(dict(id=pid, tstart=t[k], tstop=stops[k], edema_tv=ed[k],
                             sex=p.sex, status2=int(p.status2) if k == len(t) - 1 else 0))
    pd.DataFrame(rows).to_csv(OUT / "tvc_surv.csv", index=False)
    # --- interval censoring
    ic = patients[["id", "years", "status2", "sex", "drug", "age"]].copy()
    last_before = visits.groupby("id")["year"].apply(list)
    t1, t2, st = [], [], []
    for _, p in ic.iterrows():
        if p.status2 == 1:
            before = [x for x in last_before[p.id] if x < p.years]
            lo = max(before) if before and max(before) > 0 else 0.5 * p.years
            t1.append(lo)
            t2.append(p.years)
            st.append(3)
        else:
            t1.append(p.years)
            t2.append(p.years)
            st.append(0)
    ic["t1"], ic["t2"], ic["ic_status"] = t1, t2, st
    ic.to_csv(OUT / "icens_surv.csv", index=False)
    # --- extra markers
    f = visits.copy()
    f["alb_p"] = f["albumin"] / 9.0
    f["bili_p"] = f["serBilir"] / (f["serBilir"] + 1.0)
    lalk = np.log(f["alkaline"])
    lim = float(np.nanquantile(lalk, 0.2))
    f["lalk_cens"] = np.where(lalk.notna(), (lalk < lim).astype(float), np.nan)
    f["lalk"] = np.where(lalk < lim, lim, lalk)
    yes = lambda c: (f[c].astype(str) == "Yes").astype(float).where(f[c].notna())  # noqa
    f["count2"] = yes("ascites") + yes("spiders")
    f["count3"] = yes("ascites") + yes("hepatomegaly") + yes("spiders")
    f.to_csv(OUT / "fam_visits.csv", index=False)
    patients.to_csv(OUT / "patients.csv", index=False)
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
