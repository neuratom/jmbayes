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
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from jmbayes import checks as ck
from jmbayes import formula as fm
from jmbayes import summary as sm
from jmbayes.design import Response, Terms, as_terms
from jmbayes.families import FAMILIES
from jmbayes.model import ModelData, base_hazard_config, jm_setup
from jmbayes.sampler import JointModelSampler

_FAMILIES = set(FAMILIES)
_ALIASES = {"negative binomial": "negative_binomial", "negative.binomial": "negative_binomial",
            "student's-t": "student_t", "student's t": "student_t", "students_t": "student_t",
            "student t": "student_t", "student-t": "student_t", "t": "student_t",
            "students.t": "student_t", "beta.fam": "beta", "gamma.fam": "gamma",
            "unit lindley": "unit_lindley", "unit.lindley": "unit_lindley",
            "censored normal": "censored_normal", "censored.normal": "censored_normal",
            "beta binomial": "beta_binomial", "beta.binomial": "beta_binomial"}


def _family(name: str) -> str:
    f = str(name).strip().lower()
    f = _ALIASES.get(f, f)
    if f not in _FAMILIES:
        raise NotImplementedError(
            f"family '{name}' is not supported yet (supported: {sorted(_FAMILIES)})")
    return f
from jmbayes import assoc as asc  # noqa: E402

_FORMS = asc.FORMS
_DTYPES = {"float64": torch.float64, "float32": torch.float32}


# =========================================================== Python interface
@dataclass
class Marker:
    """A repeatedly measured marker (longitudinal sub-model, a mixed model).

    response:  column name, or var("x").log() / var("x").sqrt() for a transformed marker
               (or give transform="log")
    fixed:     fixed effects, e.g. year * sex, or ["year", "sex"]; intercept included
    random:    random effects per patient besides the intercept, e.g. year (random
               intercept and slope); None = random intercept only
    family:    "gaussian" (linear mixed model), "binomial" (logistic), "poisson" (counts,
               log link), "negative_binomial" (overdispersed counts, log link),
               "student_t" (continuous with heavy tails, robust to outliers), "beta"
               (proportions in (0, 1), logit link), "gamma" (positive, skewed; log link),
               "unit_lindley" (proportions in (0, 1), logit link, no extra parameter),
               "censored_normal" (values with detection limits) or "beta_binomial"
               (overdispersed successes out of trials, logit link)
    df:        degrees of freedom of "student_t" (required, as in JMbayes2; e.g. 4)
    trials:    "binomial" / "beta_binomial": column (or number) with the number of
               trials of each measurement; the marker is then the number of successes
               (cbind(successes, failures) in R). Default: one trial (0/1 marker).
    censoring: "censored_normal": column coded 0 (observed), 1 (left-censored: the true
               value is at most the recorded one, e.g. below a detection limit) or
               2 (right-censored: at least the recorded one)
    association: how the marker enters the hazard: one or a list of "value" (current
               value), "slope" / "velocity" (rate of change), "acceleration", "area"
               (average from 0 to t), "Delta" (change since 0), "coefs", or terms from
               jmbayes.assoc with options, transformations and interactions, e.g.
               A.slope(eps=0.01), A.area(time_window=2), A.vexpit(A.value()),
               A.value() & var("sex"), A.value() * ns(year, knots=[3, 6], boundary=[0, 15])
               (a time-varying association), A.Dexp(A.slope())
    uncorrelated: True for independent random effects (R's "||")
    """
    response: str | Response
    fixed: object = None
    random: object = None
    family: str = "gaussian"
    association: list | tuple | str = ("value",)
    uncorrelated: bool = False
    transform: str | None = None
    df: float | None = None
    trials: object = None
    censoring: str | None = None

    def __post_init__(self):
        self.family = _family(self.family)
        if self.trials is not None and self.family not in ("binomial", "beta_binomial"):
            raise ValueError("trials= is only used with families 'binomial' and "
                             "'beta_binomial'")
        if self.family == "censored_normal":
            if not isinstance(self.censoring, str):
                raise ValueError("family 'censored_normal' needs censoring= (a column "
                                 "coded 0 observed, 1 left-censored, 2 right-censored)")
        elif self.censoring is not None:
            raise ValueError("censoring= is only used with family 'censored_normal'")
        if self.family == "student_t":
            if self.df is None or not float(self.df) > 0:
                raise ValueError("family 'student_t' needs its degrees of freedom, e.g. "
                                 "df=4 (JMbayes2 also asks for them: students.t(df = 4))")
        elif self.df is not None:
            raise ValueError("df= is only used with family 'student_t'")
        if isinstance(self.response, Terms):
            raise TypeError("response: give a column name, or var(name).log()")
        if not isinstance(self.response, Response):
            self.response = Response(self.response, self.transform)
        elif self.transform is not None:
            raise ValueError("give the transform either in response or in transform=, not both")
        self.fixed = Terms(as_terms(self.fixed).terms)
        self.random = Terms(as_terms(self.random).terms)
        self.association = asc.as_assoc_list(self.association)

    @property
    def name(self) -> str:
        return self.response.name

    @property
    def aux(self):
        return self.trials if self.trials is not None else self.censoring


@dataclass
class Event:
    """The event-time sub-model (proportional hazards with a spline baseline hazard).

    time:       follow-up time column (same scale as the visit times)
    event:      event indicator column, coded 0/1; or, for competing risks, a column
                naming what happened (e.g. "alive", "transplanted", "dead")
    covariates: covariates, e.g. sex, ["sex", "age"] or var("sex") * var("age"); None for
                a model without covariates. With several rows per patient (start=) they
                may change over time (each row holds the values for its interval)
    censored:   competing risks: the value of `event` that means no event (e.g. "alive").
                Every other value is a cause, with its own baseline hazard, its own
                covariate effects and its own marker associations (cause-specific
                hazards, as with crisk_setup() and strata(CR) in JMbayes2).
    start:      start of each row's risk interval (Surv(start, time, event) in R). With
                one row per patient it is the entry time (delayed entry / left
                truncation): at risk only from then on. Patients may also have several
                rows (counting-process data), e.g. for covariates that change over time
                or for recurrent events
    strata:     column whose groups (e.g. hospital) each get their own baseline hazard,
                with the covariate and marker effects shared (strata() in R)
    by_strata:  also give each stratum its own covariate effects and marker associations
                (like (x + z):strata(s) and value(y):strata in R)
    time2:      interval censoring (Surv(time, time2, event, type = "interval")): the
                event column is then coded 0 right-censored at time, 1 event at time,
                2 left-censored (event before time), 3 event between time and time2
    weights:    column of case weights (coxph(weights = ) in R); as JMbayes2, they
                multiply the hazard of each event (log w is added to its log hazard)
    recurrent:  "gap" or "calendar": recurrent events with a frailty, as
                jm(..., recurrent = ). Needs start= (rows per risk interval) and strata=,
                the first stratum being the recurrent event and any further strata
                terminal events (see rc_setup())
    """
    time: str
    event: str
    covariates: object = None
    censored: object = None
    start: str | None = None
    strata: str | None = None
    by_strata: bool = False
    time2: str | None = None
    recurrent: str | None = None
    weights: str | None = None

    def __post_init__(self):
        self.covariates = as_terms(self.covariates).without_intercept()
        if self.by_strata and not self.strata:
            raise ValueError("by_strata=True needs strata=")
        if self.recurrent is not None:
            if self.recurrent not in ("gap", "calendar"):
                raise ValueError("recurrent must be 'gap' or 'calendar'")
            if not (self.start and self.strata):
                raise ValueError("recurrent events need start= (one row per risk "
                                 "interval) and strata= (recurrent event first, then "
                                 "terminal events); rc_setup() builds such data")
        if self.time2 and (self.start or self.censored is not None or self.recurrent):
            raise NotImplementedError("interval censoring together with start=, "
                                      "competing risks or recurrent events")
        if self.recurrent and self.censored is not None:
            raise NotImplementedError("recurrent events together with competing risks")


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
            base_hazard: str | None = None, basis: str = "bs",
            timescale: str = "identity", penalty_gammas: str = "none",
            penalty_alphas: str = "none",
            initial_fits: dict | None = None, dtype: str = "float32",
            save_loglik: bool = False, marginal_stats: bool = True,
            verbose: bool = True) -> "JointModelFit":
        """Fit by MCMC, with the same priors and sampler as JMbayes2::jm().

        visits:   one row per measurement (id, visit time, markers, covariates)
        patients: one row per patient (id, follow-up time, event 0/1, covariates)
        n_iter, n_burnin, n_thin, n_chains, seed: MCMC settings, as in JMbayes2 (n_iter
                  defaults to 3500, or 6500 when some fixed effects have no matching
                  random effect, as JMbayes2 does)
        device:   "cuda" (GPU), "cpu", or None to use the GPU when available
        dtype:    precision of the MCMC: "float32" (default; 1.4-1.7x faster on a GPU,
                  shown statistically equivalent to JMbayes2) or "float64" (as JMbayes2).
                  If a float32 run produces invalid numbers it is redone in float64.
        save_loglik: also keep each patient's log-likelihood at every iteration after
                  burn-in, as fit.loglik (chains x iterations x patients), like JMbayes2's
                  save_logLik_contributions. Off by default: DIC, WAIC and LPML are computed
                  either way, and the table needs (n_iter - n_burnin) x n_chains x patients
                  numbers of memory (e.g. 2.2 GB for 2,000 patients, 3 chains, 90,000
                  iterations in float32). Use it for checks that need every draw, such as
                  PSIS-LOO and its Pareto-k diagnostic.
        base_hazard: shape of the log baseline hazard, as jm(base_hazard = ): None (the
                  default: quadratic B-splines with 9 segments), "piecewise constant",
                  "piecewise linear", "Weibull", "log time" or "natural cubic spline";
                  or set basis ("bs" / "ns"), Bsplines_degree, base_hazard_segments and
                  timescale ("identity" / "log") directly
        penalty_gammas, penalty_alphas: shrinkage priors for the covariate effects and
                  the association parameters, as jm(priors = list(penalty_gammas = ,
                  penalty_alphas = )): "none" (default), "ridge" or "horseshoe"
        initial_fits: optional separate fits exported from R (benchmark/export_initfits.R)
        marginal_stats: also compute the marginal DIC, WAIC and LPML (random effects
                  integrated out by a Laplace approximation, as JMbayes2), in
                  fit.marginal_fit_statistics. Costs one extra likelihood evaluation per
                  kept draw.
        """
        return _fit(self, visits, patients, n_iter=n_iter, n_burnin=n_burnin, n_thin=n_thin,
                    n_chains=n_chains, seed=seed, device=device,
                    base_hazard_segments=base_hazard_segments, base_hazard=base_hazard,
                    basis=basis, timescale=timescale, penalty_gammas=penalty_gammas,
                    penalty_alphas=penalty_alphas,
                    Bsplines_degree=Bsplines_degree, initial_fits=initial_fits,
                    dtype=dtype, save_loglik=save_loglik, marginal_stats=marginal_stats,
                    verbose=verbose)


# =========================================================== R-style strings
@dataclass
class Longitudinal:
    """A longitudinal sub-model written as R formulas, like lme() or mixed_model().

    formula: fixed effects, e.g. "log(serBilir) ~ year * sex"
    random:  random effects and grouping variable, e.g. "~ year | id" or "~ 1 | id";
             "||" for uncorrelated random effects, e.g. "~ year || id"
    family:  "gaussian", "binomial", "poisson", "negative_binomial" or "student_t"
    df:      degrees of freedom for "student_t"
    trials, censoring: as in Marker
    """
    formula: str
    random: str
    family: str = "gaussian"
    df: float | None = None
    trials: object = None
    censoring: str | None = None

    def __post_init__(self):
        self.response, self.fixed = fm.parse_formula(self.formula)
        self.random_terms, self.id_var, self.uncorrelated = fm.parse_random(self.random)
        self.family = Marker(self.response, family=self.family, df=self.df,
                             trials=self.trials, censoring=self.censoring).family

    def to_marker(self, association=("value",)) -> Marker:
        return Marker(self.response, fixed=self.fixed, random=self.random_terms,
                      family=self.family, association=association,
                      uncorrelated=self.uncorrelated, df=self.df, trials=self.trials,
                      censoring=self.censoring)


@dataclass
class Survival:
    """The event-time sub-model written as an R formula, like coxph().

    formula: e.g. "Surv(years, status2) ~ sex + age"; the event column must be 0/1.
             "Surv(entry, years, status2) ~ ..." for delayed entry or several rows per
             patient, "Surv(time, time2, status, type = 'interval') ~ ..." for interval
             censoring (status 0-3), "+ strata(x)" for a baseline hazard per group of x,
             "(sex + age):strata(x)" for effects per group too
    data:    one row per patient (or per risk interval with Surv(start, stop, event))
    recurrent: "gap" or "calendar" for recurrent events (see Event)
    """
    formula: str
    data: pd.DataFrame
    recurrent: str | None = None

    def __post_init__(self):
        d = fm.parse_surv_full(self.formula)
        self.time, self.event, self.covariates = d["time"], d["event"], d["covariates"]
        self.start, self.strata, self.time2 = d["start"], d["strata"], d["time2"]
        self.by_strata = d["by_strata"]
        _check_event(self.data, self.event, interval=bool(self.time2))

    def to_event(self) -> Event:
        return Event(self.time, self.event, self.covariates, start=self.start,
                     strata=self.strata, by_strata=self.by_strata, time2=self.time2,
                     recurrent=self.recurrent)


def _check_event(df, col, interval=False):
    if col not in df:
        raise ck.DataError(f"event column '{col}' not found in the patients data")
    if interval:
        if not set(pd.unique(df[col].dropna())) <= {0, 1, 2, 3}:
            raise ck.DataError(f"event column '{col}' must be coded 0-3 for interval "
                               "censoring (0 right, 1 event, 2 left, 3 interval)")
        return
    if not set(pd.unique(df[col].dropna())) <= {0, 1}:
        raise ck.DataError(f"event column '{col}' must be coded 0/1 "
                           "(e.g. create it with (df.status != 'alive').astype(int))")


def _marker_specs(model):
    return [(m.response, [*m.fixed.variables(), *m.random.variables()], m.family, m.aux)
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
    loglik: np.ndarray | None = field(default=None, repr=False)
    marginal_fit_statistics: dict | None = None
    """DIC, pD, WAIC, LPML with the random effects integrated out (JMbayes2's
    fit_stats$marginal); fit_statistics holds the conditional ones."""
    """Log-likelihood of each patient at every iteration after burn-in, shape
    (chains, iterations, patients), patients in the order of .patient_ids. Only kept
    with fit(..., save_loglik=True); None otherwise."""
    _tables: dict = field(default_factory=dict, repr=False)

    @property
    def patient_ids(self) -> list:
        """Patient ids in the order used by .loglik and the random effects."""
        return list(self.md.sid_names)

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
        """Survival sub-model: covariates and association parameters (log-hazard scale);
        with recurrent events also the frailty association with each terminal event
        (frailty:<stratum>) and the frailty SD (sigma_frailty), as summary.jm()."""
        parts = [self._table("gammas"), self._table("alphas")]
        if "alphaF" in self.blocks and len(self.md.strata_levels or []) > 1:
            parts.append(self._table("alphaF"))
        if "sigmaF" in self.blocks:
            parts.append(self._table("sigmaF"))
        return pd.concat(parts)

    def hazard_ratios(self) -> pd.DataFrame:
        """Hazard ratios with 95% credible intervals and tail probabilities.

        HR = exp(posterior mean of the log hazard ratio); the interval is exp() of the
        2.5% and 97.5% posterior quantiles. For value(marker) the HR is per one unit of
        the marker on the scale of its model (e.g. log bilirubin, or log-odds for a
        binomial outcome).
        """
        t = self.survival_table().drop(index="sigma_frailty", errors="ignore")
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
    def summary(self, digits: int = 4) -> None:
        """Print a summary like summary.jm() in R. The tables themselves come from
        .survival_table(), .hazard_ratios(), .coefficients() and
        .random_effects_covariance()."""
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
        mfs = self.marginal_fit_statistics
        if mfs:
            print("\nMarginal fit statistics:     " +
                  "  ".join(f"{k} {mfs[k]:.2f}" for k in ("DIC", "WAIC", "LPML")))
        print(("" if mfs else "\n") + "Conditional fit statistics:  " +
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
            link = FAMILIES[o.family]["link"]
            print(f"\nLongitudinal outcome: {o.name} (family = {o.family}, link = {link})")
            print(fmt(tab))
        print(f"\nMCMC: {s['n_chains']} chains, {s['n_iter']} iterations, burn-in "
              f"{s['n_burnin']}, thinning {s['n_thin']}, device {self.timing['device']}, "
              f"time {self.timing['mcmc']:.1f}s")

    def __repr__(self):
        return (f"<JointModelFit: {self.md.n} patients, {len(self.md.outcomes)} longitudinal "
                f"outcomes, {self.settings['n_chains']} chains x {self.settings['n_iter']} "
                "iterations; use .summary()>")

    # ---- predictions -------------------------------------------------------------
    def predict(self, newdata: pd.DataFrame, process: str = "event", times=None,
                n_samples: int = 200, n_mcmc: int = 55, level: float = 0.95,
                seed: int = 123, device: str | None = None,
                type: str = "subject_specific", draws: str = "spread",
                r_sampling: bool = False) -> pd.DataFrame:
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
        r_sampling: sample each patient's random effects as JMbayes2's simulate_REs()
                 does (see NOTES.md). Off by default: jmbayes uses the consistent
                 survival term, which can change risks by a few percentage points.
        """
        from jmbayes.prediction import predict as _predict
        return _predict(self, newdata, process=process, times=times, n_samples=n_samples,
                        n_mcmc=n_mcmc, level=level, seed=seed, device=device, type=type,
                        draws=draws, r_sampling=r_sampling)

    # ---- accuracy of dynamic predictions ------------------------------------------
    # newdata: visits data (one row per measurement) with each patient's follow-up time
    # and event. Patients still event-free at t_start are used, with their measurements
    # up to t_start, to predict the risk of the event by t_horizon (= t_start + dt).
    # Extra keyword arguments go to predict() (e.g. n_samples, seed, r_sampling).
    def tv_roc(self, newdata: pd.DataFrame, t_start: float, t_horizon: float | None = None,
               dt: float | None = None, weights: str = "model-based", **predict_options):
        """Time-dependent ROC curve (sensitivity and specificity at 101 cut-offs), as
        tvROC() in JMbayes2. weights: "model-based" (default) or "IPCW" for patients
        censored before t_horizon."""
        from jmbayes import accuracy
        return accuracy.tv_roc(self, newdata, t_start, t_horizon, dt, weights,
                               **predict_options)

    def tv_auc(self, newdata: pd.DataFrame, t_start: float, t_horizon: float | None = None,
               dt: float | None = None, weights: str = "model-based", **predict_options):
        """Time-dependent AUC (discrimination), as tvAUC() in JMbayes2."""
        from jmbayes import accuracy
        return accuracy.tv_auc(self, newdata, t_start, t_horizon, dt, weights,
                               **predict_options)

    def tv_brier(self, newdata: pd.DataFrame, t_start: float, t_horizon: float | None = None,
                 dt: float | None = None, integrated: bool = False,
                 weights: str = "model-based", **predict_options):
        """Time-dependent Brier score at t_horizon, or integrated over [t_start,
        t_horizon) with integrated=True, as tvBrier() in JMbayes2."""
        from jmbayes import accuracy
        return accuracy.tv_brier(self, newdata, t_start, t_horizon, dt, integrated, weights,
                                 **predict_options)

    def calibration(self, newdata: pd.DataFrame, t_start: float,
                    t_horizon: float | None = None, dt: float | None = None,
                    df_ns: int | None = None, **predict_options):
        """Calibration at t_horizon: ICI, E50, E90 and the observed-vs-predicted curve
        (.curve), as calibration_metrics() and calibration_plot() in JMbayes2."""
        from jmbayes import accuracy
        return accuracy.calibration(self, newdata, t_start, t_horizon, dt, df_ns,
                                    **predict_options)

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
    """Fits saved by version 0.3 stored formulas as strings; convert them to terms.
    Fits saved before 0.6 have no strata fields; competing-risk strata were the causes."""
    from jmbayes.design import SplineBasis
    if "strata_levels" not in md.__dict__:
        md.strata_levels = None if md.__dict__.get("causes") is None else list(md.causes)
        md.knots_strata = None if md.strata_levels is None else [md.knots] * len(md.causes)
        md.trunc = None
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
                spec = asc.parse(spec, responses).get(key(resp), [])
            out[key(resp)] += asc.as_assoc_list(spec)
    elif functional_forms is not None:
        for resp, forms in asc.parse(functional_forms, responses).items():
            out[key(resp)] += forms
    return {r: (v or [asc.value()]) for r, v in out.items()}


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
         device, base_hazard_segments, Bsplines_degree, initial_fits, dtype, verbose,
         base_hazard=None, basis="bs", timescale="identity", penalty_gammas="none",
         penalty_alphas="none",
         save_loglik=False, marginal_stats=True):
    dt = _DTYPES.get(str(dtype).replace("torch.", ""))
    if dt is None:
        raise ValueError(f"dtype must be 'float64' or 'float32', got {dtype!r} (16-bit "
                         "formats are too coarse for MCMC acceptance decisions)")
    ev = model.event
    ck.check_patients(patients, model.id, ev.time, ev.event,
                      [*ev.covariates.variables(), *([ev.weights] if ev.weights else [])],
                      censored=ev.censored, start=ev.start, strata=ev.strata,
                      time2=ev.time2, recurrent=ev.recurrent)
    causes = ck.event_causes(patients[model.event.event], model.event.censored)
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
                          family=m.family, forms=m.association, uncorrelated=m.uncorrelated,
                          df=m.df, aux=m.aux)
                     for m in model.markers],
        which_independent=model.independent,
        surv=dict(time=model.event.time, event=model.event.event,
                  covariates=model.event.covariates, causes=causes,
                  censored=model.event.censored, start=model.event.start,
                  strata=model.event.strata, by_strata=model.event.by_strata,
                  time2=model.event.time2, recurrent=model.event.recurrent,
                  weights=model.event.weights),
        id_var=model.id, time_var=model.time, base_hazard_segments=base_hazard_segments,
        Bsplines_degree=Bsplines_degree, initial_fits=initial_fits,
        base_hazard=base_hazard_config(base_hazard, basis, Bsplines_degree,
                                       base_hazard_segments, timescale),
        penalty_gammas=penalty_gammas, penalty_alphas=penalty_alphas)
    t_setup = time.perf_counter() - t0
    if n_iter is None:
        n_iter = 6500 if any(len(o.nhc) for o in md.outcomes) else 3500
        if n_burnin >= n_iter:
            raise ValueError("n_burnin must be smaller than n_iter")
    while True:
        smp = JointModelSampler(md, n_chains=n_chains, device=device, seed=seed, dtype=dt)
        if verbose:
            print(f"setup done in {t_setup:.1f}s; sampling on {smp.dev} "
                  f"({str(dt).replace('torch.', '')}) ...", flush=True)
        out = smp.run(n_iter=n_iter, n_burnin=n_burnin, progress=verbose,
                      save_loglik=save_loglik)
        blocks = sm.collect(md, out, n_thin=n_thin)
        if dt == torch.float64 or all(np.isfinite(v[0]).all() for v in blocks.values()):
            break
        warnings.warn("float32 sampling produced invalid values (e.g. a covariance that is "
                      "numerically singular); refitting in float64", RuntimeWarning)
        dt = torch.float64
    t0 = time.perf_counter()
    ll_mean = sm.loglik_at_means(smp, md, blocks, out)
    fs = sm.fit_stats(out["fitstats_acc"], out["n_keep"], ll_mean)
    mfs = (sm.marginal_fit_stats(smp, md, blocks, out, ll_mean) if marginal_stats
           else None)
    t_stats = time.perf_counter() - t0
    return JointModelFit(
        md=md, blocks=blocks, fit_statistics=fs, marginal_fit_statistics=mfs,
        settings=dict(n_iter=n_iter, n_burnin=n_burnin, n_thin=n_thin, n_chains=n_chains,
                      seed=seed, dtype=str(dt).replace("torch.", "")),
        timing=dict(setup=t_setup, mcmc=out["elapsed"], fit_statistics=t_stats,
                    device=out["device"],
                    gpu_peak_mb=out["gpu_peak_mb"]),
        random_effects_mean=out["mean_b"].mean(0), loglik=out.get("loglik"))
