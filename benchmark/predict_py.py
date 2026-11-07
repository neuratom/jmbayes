"""Python dynamic predictions for the same patients as predict_R.R."""
import json
import pathlib
import time

import numpy as np
import jmbayes as jmb

HERE = pathlib.Path(__file__).resolve().parent
pbc2, pid = jmb.datasets.pbc2()
path = HERE / "PY_readme_fit.jmb"
if path.exists():
    fit = jmb.load(str(path))
else:
    fit = jmb.jm(jmb.Survival("Surv(years, status2) ~ sex", data=pid),
                 [jmb.Longitudinal("log(serBilir) ~ year * sex", random="~ year | id"),
                  jmb.Longitudinal("prothrombin ~ year * sex", random="~ year | id"),
                  jmb.Longitudinal("ascites ~ year + sex", random="~ year | id",
                                   family="binomial")],
                 data=pbc2, time_var="year", n_iter=12000, n_burnin=2000, n_thin=5,
                 verbose=False)
    fit.save(str(path))
t0 = 5
ND = pbc2[pbc2["id"].isin([25, 93]) & (pbc2["year"] < t0)].copy()
ND["status2"] = 0.0
ND["years"] = float(t0)
tic = time.perf_counter()
ev = fit.predict(ND, process="event", times=np.linspace(t0, 12, 8))
lg = fit.predict(ND, process="longitudinal")
print(f"predictions in {time.perf_counter() - tic:.1f}s")
print(ev.round(4).to_string())
(HERE / "PY_predict.json").write_text(json.dumps(
    {"event": ev.to_dict(orient="list"), "long": lg.to_dict(orient="list")}, default=float))
