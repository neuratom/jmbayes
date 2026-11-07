"""User-facing interface: two ways to describe the same joint model.

Python objects (main interface):

    import jmbayes as jmb
    from jmbayes import var

    visits, patients = jmb.datasets.pbc2()
    year, sex = var("year"), var("sex")
    model = jmb.JointModel(
        event=jmb.Event(time="years", event="status2", covariates=sex),
        markers=[jmb.Marker(var("serBilir").log(), fixed=year * sex, random=year),
                 jmb.Marker("ascites", fixed=year + sex, random=year, family="binomial")],
        id="id", time="year")
    fit = model.fit(visits, patients)

R-style formula strings (same model, same code underneath):

    fit = jmb.jm(jmb.Survival("Surv(years, status2) ~ sex", data=patients),
                 [jmb.Longitudinal("log(serBilir) ~ year * sex", random="~ year | id"),
                  jmb.Longitudinal("ascites ~ year + sex", random="~ year | id",
                                   family="binomial")],
                 data=visits, time_var="year")
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from jmbayes import checks as ck
from jmbayes import formula as fm
from jmbayes import summary as sm
from jmbayes.design import Response, Terms, as_terms
from jmbayes.model import ModelData, jm_setup
from jmbayes.sampler import JointModelSampler

_FAMILIES = {"gaussian", "binomial"}
_FORMS = ("value", "slope", "area")


# =========================================================== Python interface
@dataclass
class Marker:
    """A repeatedly measured marker (longitudinal sub-model, a mixed model).

    response:  column name, or var("x").log() / var("x").sqrt() for a transformed marker
               (or give transform="log")
    fixed:     fixed effects, e.g. year * sex, or ["year", "sex"]; intercept included
    random:    random effects per patient besides the intercept, e.g. year (random
               intercept and slope); None = random intercept only
    family:    "gaussian" (linear mixed model) or "binomial" (logistic mixed model)
    association: how the marker enters the hazard, any of "value" (current value),
               "slope" (current rate of change), "area" (average from 0 to t)
    uncorrelated: True for independent random effects (R's "||")
    """
    response: str | Response
    fixed: object = None
    random: object = None
    family: str = "gaussian"
    association: list | tuple | str = ("value",)
    uncorrelated: bool = False
    transform: str | None = None

    def __post_init__(self):
        if self.family not in _FAMILIES:
            raise NotImplementedError(
                f"family '{self.family}' is not supported yet (supported: {sorted(_FAMILIES)})")
        if isinstance(self.response, Terms):
            raise TypeError("response: give a column name, or var(name).log()")
        if not isinstance(self.response, Response):
            self.response = Response(self.response, self.transform)
        elif self.transform is not None:
            raise ValueError("give the transform either in response or in transform=, not both")
        self.fixed = Terms(as_terms(self.fixed).terms)
        self.random = Terms(as_terms(self.random).terms)
        if isinstance(self.association, str):
            self.association = [self.association]
        bad = [f for f in self.association if f not in _FORMS]
        if bad or not self.association:
            raise NotImplementedError(f"association {bad} not supported (use {_FORMS})")
        self.association = list(self.association)

    @property
    def name(self) -> str:
        return self.response.name


@dataclass
class Event:
    """The event-time sub-model (proportional hazards with a spline baseline hazard).

    time:       follow-up time column (same scale as the visit times)
    event:      event indicator column, coded 0/1
    covariates: baseline covariates, e.g. sex, ["sex", "age"] or var("sex") * var("age")
    """
    time: str
    event: str
    covariates: object = None

    def __post_init__(self):
        self.covariates = as_terms(self.covariates).without_intercept()
        if not self.covariates.terms:
            raise NotImplementedError("a survival model without covariates is not supported yet")


@dataclass
class JointModel:
    """A joint model: an Event sub-model plus one or more Marker sub-models.

    id:   patient id column (in both data frames)
    time: visit time column in the visits data
    independent: markers whose random effects are uncorrelated with each other:
          "all", or pairs of marker numbers (from 1), e.g. [(1, 2)]
    """
    event: Event
    markers: list | Marker
    id: str
    time: str
    independent: object = None

    def __post_init__(self):
        if isinstance(self.markers, Marker):
            self.markers = [self.markers]
        if not self.markers:
            raise ValueError("give at least one Marker")
        names = [m.name for m in self.markers]
        if len(set(names)) != len(names):
            raise ValueError(f"each marker can appear once, got {names}")

    def fit(self, visits: pd.DataFrame, patients: pd.DataFrame, *,
            n_iter: int | None = None, n_burnin: int = 500, n_thin: int = 1,
            n_chains: int = 3, seed: int = 123, device: str | None = None,
            base_hazard_segments: int = 9, Bsplines_degree: int = 2,
            initial_fits: dict | None = None, verbose: bool = True) -> "JointModelFit":
        """Fit by MCMC, with the same priors and sampler as JMbayes2::jm().

        visits:   one row per measurement (id, visit time, markers, covariates)
        patients: one row per patient (id, follow-up time, event 0/1, covariates)
        n_iter, n_burnin, n_thin, n_chains, seed: MCMC settings, as in JMbayes2 (n_iter
                  defaults to 3500, or 6500 when some fixed effects have no matching
                  random effect, as JMbayes2 does)
        device:   "cuda" (GPU), "cpu", or None to use the GPU when available
        initial_fits: optional separate fits exported from R (benchmark/export_initfits.R)
        """
        return _fit(self, visits, patients, n_iter=n_iter, n_burnin=n_burnin, n_thin=n_thin,
                    n_chains=n_chains, seed=seed, device=device,
                    base_hazard_segments=base_hazard_segments,
                    Bsplines_degree=Bsplines_degree, initial_fits=initial_fits,
                    verbose=verbose)


# =========================================================== R-style strings
@dataclass
class Longitudinal:
    """A longitudinal sub-model written as R formulas, like lme() or mixed_model().

    formula: fixed effects, e.g. "log(serBilir) ~ year * sex"
    random:  random effects and grouping variable, e.g. "~ year | id" or "~ 1 | id";
             "||" for uncorrelated random effects, e.g. "~ year || id"
    family:  "gaussian" or "binomial"
    """
    formula: str
    random: str
    family: str = "gaussian"

    def __post_init__(self):
        self.response, self.fixed = fm.parse_formula(self.formula)
        self.random_terms, self.id_var, self.uncorrelated = fm.parse_random(self.random)
        Marker(self.response, family=self.family)          # validates the family early

    def to_marker(self, association=("value",)) -> Marker:
        return Marker(self.response, fixed=self.fixed, random=self.random_terms,
                      family=self.family, association=association,
                      uncorrelated=self.uncorrelated)


@dataclass
class Survival:
    """The event-time sub-model written as an R formula, like coxph().

    formula: e.g. "Surv(years, status2) ~ sex + age"; the event column must be 0/1
    data:    one row per patient
    """
    formula: str
    data: pd.DataFrame

    def __post_init__(self):
        self.time, self.event, self.covariates = fm.parse_surv(self.formula)
        _check_event(self.data, self.event)

    def to_event(self) -> Event:
        return Event(self.time, self.event, self.covariates)


def _check_event(df, col):
    if col not in df:
        raise ck.DataError(f"event column '{col}' not found in the patients data")
    if not set(pd.unique(df[col].dropna())) <= {0, 1}:
        raise ck.DataError(f"event column '{col}' must be coded 0/1 "
                           "(e.g. create it with (df.status != 'alive').astype(int))")


def _marker_specs(model):
    return [(m.response, [*m.fixed.variables(), *m.random.variables()], m.family)
            for m in model.markers]


@dataclass
class JointModelFit:
    """Result of JointModel.fit() or jm(). Use .summary(), .hazard_ratios(), .coefficients(), .save()."""
    md: ModelData
    blocks: dict
    fit_statistics: dict
    settings: dict
    timing: dict
    random_effects_mean: np.ndarray
    _tables: dict = field(default_factory=dict, repr=False)

    # ---- tables ------------------------------------------------------------
    def _table(self, block: str) -> pd.DataFrame:
        if block not in self._tables:
            draws, names = self.blocks[block]
            rows = sm.table(draws, names)
            df = pd.DataFrame(rows).set_index("name")
            df.index.name = None
            self._tables[block] = df[["Mean", "StDev", "2.5%", "97.5%", "P", "Rhat", "ESS",
                                      "SE"]]
        return self._tables[block]

    def survival_table(self) -> pd.DataFrame:
        """Survival sub-model: covariates and association parameters (log-hazard scale)."""
        return pd.concat([self._table("gammas"), self._table("alphas")])

    def hazard_ratios(self) -> pd.DataFrame:
        """Hazard ratios with 95% credible intervals and tail probabilities.

        HR = exp(posterior mean of the log hazard ratio); the interval is exp() of the
        2.5% and 97.5% posterior quantiles. For value(marker) the HR is per one unit of
        the marker on the scale of its model (e.g. log bilirubin, or log-odds for a
        binomial outcome).
        """
        t = self.survival_table()
        return pd.DataFrame({"HR": np.exp(t["Mean"]), "2.5%": np.exp(t["2.5%"]),
                             "97.5%": np.exp(t["97.5%"]), "P": t["P"]}, index=t.index)

    def coefficients(self, outcome: int | str) -> pd.DataFrame:
        """Fixed effects of one longitudinal outcome (by position from 1, or response name)."""
        j = self._outcome_index(outcome)
        df = self._table(f"betas{j + 1}").copy()
        o = self.md.outcomes[j]
        if o.has_sigma:
            k = [oo.has_sigma for oo in self.md.outcomes][:j + 1].count(True) - 1
            draws, _ = self.blocks["sigmas"]
            sig = pd.DataFrame(sm.table(draws[:, :, [k]], ["sigma"])).set_index("name")
            sig.index.name = None
            df = pd.concat([df, sig[df.columns]])
        return df

    def random_effects_covariance(self) -> pd.DataFrame:
        """Posterior summary of the random-effects covariance matrix D (lower triangle)."""
        return self._table("D")

    def random_effects(self) -> pd.DataFrame:
        """Posterior mean random effects per patient."""
        cols = []
        for o in self.md.outcomes:
            cols += [f"{o.name}: {z}" for z in o.z_names]
        return pd.DataFrame(self.random_effects_mean, index=self.md.sid_names, columns=cols)

    def _outcome_index(self, outcome) -> int:
        if isinstance(outcome, int):
            return outcome - 1
        names = [o.name for o in self.md.outcomes]
        if outcome not in names:
            raise KeyError(f"unknown outcome {outcome!r}; choose from {names}")
        return names.index(outcome)

    # ---- printing ------------------------------------------------------------
    def summary(self, digits: int = 4) -> dict:
        """Print a summary like summary.jm() in R and return the tables."""
        fmt = lambda df: df[["Mean", "StDev", "2.5%", "97.5%", "P", "Rhat"]].to_string(  # noqa
            float_format=lambda v: f"{v:.{digits}f}")
        md, s = self.md, self.settings
        out = {"survival": self.survival_table(), "hazard_ratios": self.hazard_ratios(),
               "D": self.random_effects_covariance()}
        print(f"\nJoint model fitted with jmbayes (Python port of JMbayes2)")
        print(f"Patients: {md.n}   events: {int(md.event.sum())} "
              f"({100 * md.event.mean():.1f}%)")
        for o in md.outcomes:
            print(f"  {o.name}: {len(o.y)} observations ({o.family})")
        fs = self.fit_statistics
        print("\nConditional fit statistics:  " +
              "  ".join(f"{k} {fs[k]:.2f}" for k in ("DIC", "WAIC", "LPML")))
        print("\nSurvival outcome (log hazard ratios):")
        print(fmt(out["survival"]))
        hr = out["hazard_ratios"]
        print("\nHazard ratios (95% credible interval):")
        for name, r in hr.iterrows():
            print(f"  {name:30s} {r['HR']:.3f} ({r['2.5%']:.3f} - {r['97.5%']:.3f})  "
                  f"P {'<0.001' if r['P'] < 0.001 else f'{r.P:.3f}'}")
        for j, o in enumerate(md.outcomes):
            tab = self.coefficients(j + 1)
            out[o.name] = tab
            link = "identity" if o.family == "gaussian" else "logit"
            print(f"\nLongitudinal outcome: {o.name} (family = {o.family}, link = {link})")
            print(fmt(tab))
        print(f"\nMCMC: {s['n_chains']} chains, {s['n_iter']} iterations, burn-in "
              f"{s['n_burnin']}, thinning {s['n_thin']}, device {self.timing['device']}, "
              f"time {self.timing['mcmc']:.1f}s")
        return out

    def __repr__(self):
        return (f"<JointModelFit: {self.md.n} patients, {len(self.md.outcomes)} longitudinal "
                f"outcomes, {self.settings['n_chains']} chains x {self.settings['n_iter']} "
                "iterations; use .summary()>")

    # ---- predictions -------------------------------------------------------------
    def predict(self, newdata: pd.DataFrame, process: str = "event", times=None,
                n_samples: int = 200, n_mcmc: int = 55, level: float = 0.95,
                seed: int = 123, device: str | None = None,
                type: str = "subject_specific", draws: str = "spread") -> pd.DataFrame:
        """Dynamic predictions for patients in `newdata`, like predict.jm() in R.

        newdata: the patients' measurements so far, in the same columns as the fitting
                 data, with the survival time column set to the last time each patient
                 was known to be event-free and the event column set to 0.
        process: "event" -> cumulative risk P(event by u | event-free at t) with a
                 credible interval (columns pred_CIF, low_CIF, upp_CIF);
                 "longitudinal" -> predicted marker values (pred_<outcome>, ...).
        times:   future times to predict at (default: 21 points up to the 90th
                 percentile of the observed event times, as in JMbayes2).
        n_samples: posterior draws used (default 200, as JMbayes2; on a GPU, using
                 more is cheap and reduces Monte Carlo error).
        draws:   "spread" (default) takes draws evenly over all chains; "first" takes
                 the first n_samples draws, exactly as JMbayes2 does.
        """
        from jmbayes.prediction import predict as _predict
        return _predict(self, newdata, process=process, times=times, n_samples=n_samples,
                        n_mcmc=n_mcmc, level=level, seed=seed, device=device, type=type,
                        draws=draws)

    # ---- persistence -----------------------------------------------------------
    def save(self, path: str) -> None:
        """Save the fitted model to a file (reload with jmbayes.load)."""
        with open(path, "wb") as f:
            pickle.dump(self, f)


def load(path: str) -> JointModelFit:
    """Load a model saved with JointModelFit.save()."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, JointModelFit):
        raise TypeError(f"{path} does not contain a jmbayes fit")
    _upgrade(obj.md)
    return obj


def _upgrade(md: ModelData) -> None:
    """Fits saved by version 0.3 stored formulas as strings; convert them to terms."""
    from jmbayes.design import SplineBasis
    for o in md.outcomes:
        if getattr(o, "fixed", None) is not None:
            continue
        if "rhs" not in o.__dict__:
            raise ValueError("this fit was saved by an early version that did not store "
                             "the model terms; please refit it")
        o.response = fm.parse_response(o.name)
        o.fixed = fm.parse_rhs(o.__dict__.pop("rhs"))
        o.random = fm.parse_rhs(o.random)
        for t in (o.fixed, o.random):          # spline knots were keyed by their text
            for f in (f for term in t.terms for f in term if isinstance(f, SplineBasis)):
                old = f"{f.kind}({f.var},{'df=' if f.named else ''}{f.df})"
                if old in o.levels:
                    o.levels[f] = o.levels.pop(old)
    sv = md.meta.get("surv", {})
    if "formula" in sv:
        sv["covariates"] = fm.parse_rhs(sv.pop("formula")).without_intercept()


def parse_functional_forms(functional_forms, responses):
    """Association forms per outcome, like jm(functional_forms = ~ ...) in R.

    Accepts an R-style string, e.g. "~ value(log(serBilir)) + slope(log(serBilir))", or a
    dict {response: string or list of forms}. Outcomes not mentioned use value() only.
    """
    norm = {r.replace(" ", ""): r for r in responses}

    def key(resp):
        k = norm.get(resp.replace(" ", ""))
        if k is None:
            raise KeyError(f"functional_forms: unknown outcome {resp!r}; outcomes are "
                           f"{list(responses)}")
        return k

    out = {r: [] for r in responses}
    if isinstance(functional_forms, dict):
        for resp, spec in functional_forms.items():
            if isinstance(spec, str):
                spec = sum(fm.parse_functional_forms(spec).values(), [])
            out[key(resp)] += list(spec)
    elif functional_forms is not None:
        for resp, forms in fm.parse_functional_forms(functional_forms).items():
            out[key(resp)] += forms
    for v in out.values():
        bad = [f for f in v if f not in _FORMS]
        if bad:
            raise NotImplementedError(f"functional forms {bad} are not supported yet "
                                      f"(supported: {', '.join(_FORMS)})")
    return {r: (v or ["value"]) for r, v in out.items()}


def jm(survival: Survival, longitudinal: list[Longitudinal] | Longitudinal,
       data: pd.DataFrame, time_var: str, *, functional_forms: str | dict | None = None,
       which_independent=None, **fit_options) -> JointModelFit:
    """Fit a joint model written with R-style formulas, like JMbayes2::jm() in R.

    survival:      Survival("Surv(time, event) ~ covariates", data=patients)
    longitudinal:  one Longitudinal(...) or a list of them (multivariate joint model)
    data:          the visits data (one row per measurement)
    time_var:      visit time column
    functional_forms: e.g. "~ value(log(serBilir)) + slope(log(serBilir))"; outcomes not
                   mentioned use value()
    which_independent: "all" or pairs of outcome numbers (from 1), e.g. [(1, 2)]
    fit_options:   n_iter, n_burnin, n_thin, n_chains, seed, device, ... (see JointModel.fit)
    """
    if isinstance(longitudinal, Longitudinal):
        longitudinal = [longitudinal]
    if not longitudinal:
        raise ValueError("give at least one Longitudinal sub-model")
    ids = {m.id_var for m in longitudinal}
    if len(ids) != 1:
        raise ValueError(f"all longitudinal models must use the same grouping variable, got {ids}")
    forms = parse_functional_forms(functional_forms, [m.response.name for m in longitudinal])
    model = JointModel(event=survival.to_event(),
                       markers=[m.to_marker(forms[m.response.name]) for m in longitudinal],
                       id=ids.pop(), time=time_var, independent=which_independent)
    return model.fit(data, survival.data, **fit_options)


def _fit(model: JointModel, visits, patients, *, n_iter, n_burnin, n_thin, n_chains, seed,
         device, base_hazard_segments, Bsplines_degree, initial_fits, verbose):
    ck.check_patients(patients, model.id, model.event.time, model.event.event,
                      model.event.covariates.variables())
    ck.check_visits(visits, model.id, model.time, _marker_specs(model))
    ck.check_same_patients(visits, patients, model.id)
    if n_iter is not None and n_burnin >= n_iter:
        raise ValueError("n_burnin must be smaller than n_iter")
    if n_iter is not None and n_iter <= 100:
        raise ValueError("n_iter must be above 100 (the first 100 iterations only "
                         "update the survival part, as in JMbayes2)")
    t0 = time.perf_counter()
    md = jm_setup(
        visits, patients,
        long_models=[dict(response=m.response, fixed=m.fixed, random=m.random,
                          family=m.family, forms=m.association, uncorrelated=m.uncorrelated)
                     for m in model.markers],
        which_independent=model.independent,
        surv=dict(time=model.event.time, event=model.event.event,
                  covariates=model.event.covariates),
        id_var=model.id, time_var=model.time, base_hazard_segments=base_hazard_segments,
        Bsplines_degree=Bsplines_degree, initial_fits=initial_fits)
    t_setup = time.perf_counter() - t0
    if n_iter is None:
        n_iter = 6500 if any(len(o.nhc) for o in md.outcomes) else 3500
        if n_burnin >= n_iter:
            raise ValueError("n_burnin must be smaller than n_iter")
    smp = JointModelSampler(md, n_chains=n_chains, device=device, seed=seed)
    if verbose:
        print(f"setup done in {t_setup:.1f}s; sampling on {smp.dev} ...", flush=True)
    out = smp.run(n_iter=n_iter, n_burnin=n_burnin, progress=verbose)
    blocks = sm.collect(md, out, n_thin=n_thin)
    ll_mean = sm.loglik_at_means(smp, md, blocks, out)
    fs = sm.fit_stats(out["fitstats_acc"], out["n_keep"], ll_mean)
    return JointModelFit(
        md=md, blocks=blocks, fit_statistics=fs,
        settings=dict(n_iter=n_iter, n_burnin=n_burnin, n_thin=n_thin, n_chains=n_chains,
                      seed=seed),
        timing=dict(setup=t_setup, mcmc=out["elapsed"], device=out["device"],
                    gpu_peak_mb=out["gpu_peak_mb"]),
        random_effects_mean=out["mean_b"].mean(0))
