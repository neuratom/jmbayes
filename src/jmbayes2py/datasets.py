"""Example data shipped with the package.

pbc2 / pbc2.id: Mayo Clinic primary biliary cirrhosis data, as distributed with the
R packages JM, JMbayes and JMbayes2 (Murtaugh et al., Hepatology 1994).
"""
from __future__ import annotations

from importlib import resources

import pandas as pd

# R factor levels, in R's order: the first level is the reference category
_LEVELS = {
    "status": ["alive", "transplanted", "dead"],
    "drug": ["placebo", "D-penicil"],
    "sex": ["male", "female"],
    "ascites": ["No", "Yes"],
    "hepatomegaly": ["No", "Yes"],
    "spiders": ["No", "Yes"],
    "edema": ["No edema", "edema no diuretics", "edema despite diuretics"],
}


def _read(name: str) -> pd.DataFrame:
    with resources.files("jmbayes2py.data").joinpath(name).open("rb") as f:
        d = pd.read_csv(f, compression="gzip")
    for col, levels in _LEVELS.items():
        if col in d:
            d[col] = pd.Categorical(d[col], categories=levels)
    return d


def pbc2() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pbc2, pbc2_id): long-format visits and one row per patient.

    Both get a numeric ``status2`` column (1 = died or transplanted, 0 = alive),
    the composite event used in the JMbayes2 examples.

    Columns: id, years (follow-up time), status, drug, age, sex, year (visit time),
    ascites, hepatomegaly, spiders, edema, serBilir, serChol, albumin, alkaline, SGOT,
    platelets, prothrombin, histologic, status2.
    """
    long = _read("pbc2.csv.gz")
    per_patient = _read("pbc2_id.csv.gz")
    for d in (long, per_patient):
        d["status2"] = (d["status"] != "alive").astype(float)
    return long, per_patient
