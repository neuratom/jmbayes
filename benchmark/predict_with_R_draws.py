"""Run the Python prediction engine on JMbayes2's own posterior draws.

With identical parameter draws, any difference from R's predict() can only come from
the prediction step itself (sampling each patient's random effects, computing risks).
"""
import copy
import json
import pathlib

import numpy as np

import jmbayes as jmb

HERE = pathlib.Path(__file__).resolve().parent
fit = jmb.load(str(HERE / "PY_readme_fit.jmb"))
Rd = json.loads((HERE.parent / "tests" / "ref_R_draws_readme.json").read_text())
f = copy.copy(fit)
f.blocks = dict(fit.blocks)
for k, v in Rd.items():
    a = np.asarray(v, float)
    if a.ndim == 1:
        a = a[:, None]
    f.blocks[k] = (a[None], fit.blocks[k][1])        # one "chain" holding all R draws
pbc2, _ = jmb.datasets.pbc2()
ND = pbc2[pbc2["id"].isin([25, 93]) & (pbc2["year"] < 5)].copy()
ND["status2"] = 0.0
ND["years"] = 5.0
R = json.loads((HERE / "R_predict_all.json").read_text())["event"]
print(f"{'id':>4}{'t':>5}{'R predict()':>13}{'Py engine, R draws':>20}{'Py fit':>9}")
own = json.loads((HERE / "PY_predict_all_seed123.json").read_text())["event"]
res = {}
for seed in (123, 456):
    res[seed] = f.predict(ND, process="event", times=np.linspace(5, 12, 8), n_samples=6000,
                          draws="first", seed=seed)["pred_CIF"].to_numpy()
for i, r in enumerate(R):
    print(f"{r['id']:>4}{r['years']:5.0f}{r['pred_CIF']:13.4f}{res[123][i]:12.4f}/{res[456][i]:.4f}"
          f"{own['pred_CIF'][i]:9.4f}")
