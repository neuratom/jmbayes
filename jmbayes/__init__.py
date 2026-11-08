"""jmbayes: Bayesian joint models for longitudinal and time-to-event data.

A Python/PyTorch implementation of the R package JMbayes2 (Rizopoulos, Papageorgiou,
Miranda Afonso): same model, same priors, same MCMC algorithm; runs on CPU or GPU.

    import jmbayes as jmb
    from jmbayes import var

    visits, patients = jmb.datasets.pbc2()
    year, sex = var("year"), var("sex")
    model = jmb.JointModel(
        event=jmb.Event(time="years", event="status2", covariates=sex),
        markers=jmb.Marker(var("serBilir").log(), fixed=year * sex, random=year),
        id="id", time="year")
    fit = model.fit(visits, patients)
    fit.summary()

R-style formula strings work too: jmb.jm(jmb.Survival(...), jmb.Longitudinal(...), ...).
"""
from jmbayes import datasets
from jmbayes.api import (Event, JointModel, JointModelFit, Longitudinal, Marker, Survival,
                            jm, load)
from jmbayes.checks import DataError
from jmbayes.design import bs, ns, var

__all__ = ["JointModel", "Marker", "Event", "var", "ns", "bs",
           "jm", "Longitudinal", "Survival", "JointModelFit", "load", "datasets", "DataError"]
__version__ = "0.5.1"
