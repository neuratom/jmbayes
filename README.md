# jmbayes2py

Bayesian joint models for longitudinal and time-to-event data in Python and PyTorch,
based on the R package [JMbayes2](https://github.com/drizopoulos/JMbayes2) by
Rizopoulos, Papageorgiou and Miranda Afonso. It fits the same model, with the same priors
and the same MCMC algorithm, so it gives the same results within Monte Carlo error. It is
also much faster and runs in constant memory.

* **Constant memory.** Memory use does not grow with the number of iterations or
  patients: about 1.2 GB of RAM in every run we measured, and at most 0.11 GB of GPU
  memory. JMbayes2's memory grows with run length (6.2 GB at 28,000 iterations for
  1,560 patients), so long runs on large cohorts exceed a 16 GB machine. jmbayes2py
  runs them on a laptop or a small GPU.
* **Faster, and it scales better.** On a GPU the MCMC is 2.5× faster than JMbayes2 on
  312 patients, and 4.4–5.4× faster on 1,560–3,120 patients and on longer runs. Adding
  ten times the patients makes jmbayes2py's MCMC about 2× slower, and JMbayes2's about
  3.7× slower (see [Benchmarks](#benchmarks)).
* **Same results.** It was checked against JMbayes2 on the same data and models. The
  estimates, hazard ratios, credible intervals and dynamic predictions agree with R
  more closely than two R runs agree with each other.

## Install

```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # GPU (or: pip install torch)
pip install -e .
```

## Fit a model

```python
import jmbayes2py as jmb
from jmbayes2py import var

visits, patients = jmb.datasets.pbc2()   # example data: visits, and one row per patient
year, sex = var("year"), var("sex")

model = jmb.JointModel(
    event=jmb.Event(time="years", event="status2", covariates=sex),
    markers=[
        jmb.Marker(var("serBilir").log(), fixed=year * sex, random=year),
        jmb.Marker("ascites", fixed=year + sex, random=year, family="binomial"),
    ],
    id="id", time="year",
)
fit = model.fit(visits, patients)
fit.summary()
```

## Benchmarks

The model is the JMbayes2 README model: pbc2, three markers (log serum bilirubin,
prothrombin time, ascites), replicated to get more patients. Each run uses 3 chains.
JMbayes2 0.6-5 ran on the CPU; jmbayes2py ran on an RTX 3090.

![MCMC time and peak RAM of JMbayes2 and jmbayes2py](benchmark/figures/benchmarks.png)

The two programs agree. Hazard ratios from the same model, fitted by each program
(posterior mean and 95% credible interval):

![Hazard ratios from JMbayes2 and jmbayes2py](benchmark/figures/forest.png)

| Patients | Iterations | JMbayes2: MCMC time | JMbayes2: peak RAM | jmbayes2py: MCMC time | jmbayes2py: peak RAM | Speed-up |
|---:|---:|---:|---:|---:|---:|---:|
| 312   | 3,500  | 55 s   | 1.28 GB | 22 s  | 1.22 GB | 2.5× |
| 624   | 3,500  | 87 s   | 1.45 GB | 24 s  | 1.22 GB | 3.6× |
| 1,560 | 3,500  | 157 s  | 1.95 GB | 33 s  | 1.22 GB | 4.8× |
| 3,120 | 3,500  | 202 s  | 2.41 GB | 46 s  | 1.24 GB | 4.4× |
| 1,560 | 7,000  | 237 s  | 2.34 GB | 66 s  | 1.23 GB | 3.6× |
| 1,560 | 14,000 | 716 s  | 3.36 GB | 133 s | 1.24 GB | 5.4× |
| 1,560 | 28,000 | 1,046 s | 6.25 GB | not run | constant by design | |

The full report, with equivalence tests, hazard ratios, credible intervals and scaling
plots, is in [benchmark/jmbayes2py_vs_JMbayes2.pdf](benchmark/jmbayes2py_vs_JMbayes2.pdf).
The scripts that reproduce it are in [benchmark/](benchmark/). The R scripts need
JMbayes2 installed from CRAN, and are only used for the comparison.

## Writing terms

Terms are Python objects and combine with operators:

| You write | Meaning |
|---|---|
| `year + sex` | main effects |
| `year * sex` | main effects and their interaction |
| `year & sex` | the interaction only |
| `year.ns(2)`, `year.bs(3)` | natural / B-spline of year with 2 / 3 degrees of freedom |
| `var("serBilir").log()` | log-transformed marker (also `.sqrt()`) |

Plain column names work too: `fixed=["year", "sex", ("year", "sex")]` (a tuple is an
interaction), or `covariates="sex"`. The intercept is always included.

## Get the results

```python
fit.hazard_ratios()                   # HR, 95% credible interval, P
fit.coefficients("log(serBilir)")     # fixed effects of one marker
fit.random_effects_covariance()       # matrix D
fit.fit_statistics                    # DIC, WAIC, LPML

fit.save("model.jmb")                 # later: fit = jmb.load("model.jmb")
```

## Predict for a patient

```python
new = visits[(visits.id == 93) & (visits.year < 5)].copy()   # measurements up to year 5
new["years"] = 5                                              # event-free up to year 5
new["status2"] = 0

fit.predict(new, times=[5, 6, 7, 8, 10])          # risk of the event by each time
fit.predict(new, process="longitudinal")          # expected marker values
```

## Your own data

You need two pandas data frames, as in JMbayes2:

* **visits** (one row per measurement): patient id, visit time, markers, covariates
* **patients** (one row per patient): the same id, follow-up time, event (0/1), covariates

```python
import pandas as pd
visits = pd.read_csv("visits.csv")
patients = pd.read_csv("patients.csv")
```

The first category of a text column is the reference group (alphabetical by default). To
choose it: `patients["sex"] = pd.Categorical(patients["sex"], ["male", "female"])`.

## Options

```python
jmb.Marker("y", fixed=year + var("age"))                      # random intercept only
jmb.Marker("y", fixed=year, random=year, uncorrelated=True)   # independent random effects
jmb.Marker("y", fixed=year, random=year,
           association=["value", "slope"])           # current value, slope and/or "area"

jmb.JointModel(..., independent="all")               # markers independent of each other

model.fit(visits, patients,
          n_iter=12000, n_burnin=2000, n_thin=5,     # MCMC length (default 3500 / 500 / 1)
          n_chains=3,                                # more chains are cheap on a GPU
          device="cpu")                              # default: GPU if available
```

Check convergence in `fit.survival_table()`: `Rhat` should be below about 1.05. If not,
run longer or with more chains.

Not supported yet (you get a clear error): outcome families other than gaussian and
binomial, competing risks, strata, left truncation, interval censoring.

## Prefer R formulas?

You can also write the same model with JMbayes2-style formula strings. Both interfaces
build the same objects underneath and give identical results:

```python
fit = jmb.jm(
    jmb.Survival("Surv(years, status2) ~ sex", data=patients),
    [jmb.Longitudinal("log(serBilir) ~ year * sex", random="~ year | id"),
     jmb.Longitudinal("ascites ~ year + sex", random="~ year | id", family="binomial")],
    data=visits, time_var="year",
    functional_forms="~ value(log(serBilir)) + slope(log(serBilir))")
```

## How it works

The MCMC is the same Metropolis-within-Gibbs sampler as JMbayes2: the same update order,
proposals, adaptive tuning and priors. What changes is how it runs:

* The chains run as one batch on the GPU (or on CPU threads).
* Each iteration is recorded once as a CUDA graph and then replayed.
* The fit statistics (DIC, WAIC, LPML) are accumulated as the run goes, so the per-draw
  log-likelihood table is never stored. That table is what makes JMbayes2's memory grow.

Implementation notes, including places where we found JMbayes2's behaviour questionable,
are in [NOTES.md](NOTES.md).

## Citation

If you use jmbayes2py, please also cite JMbayes2:
Rizopoulos D, Papageorgiou G, Miranda Afonso P. *JMbayes2: Extended Joint Models for
Longitudinal and Time-to-Event Data*. R package, https://github.com/drizopoulos/JMbayes2.

## Licence

GPL-3.0-or-later, like JMbayes2, from which this is derived.
