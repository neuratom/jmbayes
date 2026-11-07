# jmbayes: JMbayes2 in Python / PyTorch

A Python port of JMbayes2's joint model for longitudinal and time-to-event data.
It uses the same model, the same priors and the same MCMC algorithm, and runs on CPU or GPU.
R is only needed to compare results.

## Run

The venv lives in `.venv` (Python 3.11, PyTorch with CUDA). The package is installed
into it in editable mode (`pip install -e .`), so `from jmbayes.sampler import JointModelSampler`
works from anywhere with that interpreter.

```
.venv/Scripts/python run_pbc2.py                 # GPU if available, README settings
.venv/Scripts/python run_pbc2.py --device cpu
```

To recreate the venv:

```
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install torch   (PyTorch build for your CUDA version: see the README)
.venv/Scripts/python -m pip install -e .
```

## Files

| File | Ports | What it does |
|---|---|---|
| `jmbayes/api.py` | `jm()`, `summary()`, `predict()` | User-facing interface (`JointModel`/`Marker`/`Event`, and the R-style `jm()`) and result object |
| `jmbayes/design.py` | R `model.matrix()` | Model terms as Python objects (`var`, `+`, `*`, `&`, `.ns()`), design matrices with R's column names and order |
| `jmbayes/formula.py` | R formulas | Parses R-style formula strings into the same term objects |
| `jmbayes/splines.py` | `splines::ns()`, `bs()` | R-identical spline bases (LINPACK QR reproduced) |
| `jmbayes/fits.py` | `lme()`, `mixed_model()`, `coxph()` | Initial fits that JMbayes2 starts from (REML LMM, ML binomial GLMM with 11-point adaptive Gauss-Hermite, Cox with Efron ties) |
| `jmbayes/model.py` | `jm()`, `jm_fit()` set-up | Quadrature, B-spline baseline hazard, association designs (value/slope/area), centring, hierarchical-centering blocks, priors, starting values |
| `jmbayes/sampler.py` | `mcmc_cpp()` and the update headers | The MCMC, step for step |
| `jmbayes/summary.py` | `jm_fit()` post-processing, `fit_stats()` | Means, SDs, intervals, tail probabilities, Rhat (coda formula), ESS (JMbayes2's AR estimate), DIC/WAIC/LPML |
| `jmbayes/prediction.py` | `predict.jm()`, `simulate_REs()`, `logLik_Event()` | Dynamic predictions of event risk and marker values |

## What is the same as JMbayes2

* Model: GLMM longitudinal parts, hazard `h0(t) exp(gamma'w + sum alpha * f(eta(t)))` with
  f = value, slope (central difference, eps 0.001) or area (average over [0, t], 15-point
  Gauss-Kronrod), quadratic B-spline log baseline hazard (9 segments), 15-point
  Gauss-Kronrod integration of the hazard.
* Priors: identical rules (jm.R lines 710-760), including the ones derived from the initial
  fits and the centring of fixed effects without a matching random effect (vcov_center).
* Sampler: same blocks, order, proposals (normal, log-normal, uniform), Gibbs steps
  (tau, hierarchically centred betas), Metropolis-Hastings for the other fixed effects,
  Robbins-Monro tuning and the iteration thresholds (19, 99, 119), per-chain jitter of
  starting values (R's `jitter()`), zero-correlation constraints (`||`, which_independent).

## What is different (layout only, not maths)

* Chains run as a batch dimension: one tensor operation advances all chains.
* On GPU, one whole iteration is captured as a CUDA graph and replayed.
* Random-effect updates change the trajectories incrementally instead of rebuilding them.
* The per-subject log-likelihood table (iterations x subjects, the RAM problem in R) is never
  stored. DIC, WAIC and LPML are built from running sums during sampling (identical values).
* The random-number stream differs, so individual draws differ. Results agree within Monte Carlo error.
* Predictions: all posterior draws are processed in parallel, each starting the random-effect
  sampler from zero (R runs them one after another with warm starts). The sampled random
  effects were checked to be converged after the default 55 iterations. By default draws are
  spread over all chains (`draws="first"` reproduces R's first-n rule).

## Possible issues in JMbayes2 found while porting

Worth raising with the JMbayes2 maintainers.

1. **Stale random-effects density after the fixed-effects step** (kept by default,
   `replicate_quirks=True`). `update_betas()` (JMbayes2_FE.h) changes `b_mat` through
   hierarchical centering, but `logLik_re` is not recomputed. The next `update_D()` and
   `update_b()` use the old value in their acceptance ratios.
2. **Stale survival denominator after the tau update** (kept by default). `tau_bs_gammas` is
   redrawn (mcmc_fit.cpp ~line 488), but `denominator_surv` is only refreshed after the betas
   step, so the gammas/alphas acceptance ratios include a prior-change term that does not
   belong there.
3. **Prediction: inconsistent survival term when sampling a new patient's random effects**
   (NOT kept by default; `predict(..., r_sampling=True)` reproduces it). `simulate_REs()`
   (mcmc_fit.cpp) multiplies the standardised association terms ((W - bar) / sd) by the
   reported association coefficients (alpha_std / sd), so each association is divided by
   the marker's SD, and it uses uncentred survival covariates without the W_std_gammas
   correction. `logLik_Event()`, which computes the risks afterwards, does it correctly
   (`Wlong_bar.zeros(); Wlong_sds.ones();` plus the W_std/Wlong_std corrections). Effect: the
   "event-free up to t" information is mis-weighted when sampling the random effects.
   Verified: with R's posterior draws, the Python engine matches R's predict() to < 0.01
   with R's sampling, and differs by up to 0.07 in risk (PBC patient 93, 1-year risk 0.75
   vs 0.69) with the consistent sampling.
4. Minor: the association design (`Wlong_H`) is built from the unjittered starting values,
   while each chain starts from jittered values. This only affects the first 100 iterations.
5. Note on Monte Carlo errors: the MCSE that JMbayes2 reports (SD / sqrt(ESS), ESS from an
   AR fit to all chains stacked) understated the real run-to-run error by a factor of about
   1.7 for the README model (measured with independent runs of both programs).

## Equivalence with JMbayes2 (details in benchmark/ and tests/)

| Model | Parameters equivalent at +-0.2 SD | Median abs. difference R vs Python |
|---|---|---|
| README model (3 markers, value) | 45/50 (none shown to differ) | 0.018 SD (R vs R: 0.023) |
| Fixed effects without a matching random effect | 19/19 | 0.006 SD |
| Splines `ns(year, 2)` | 13/13 | 0.029 SD |
| `value()` + `slope()` + `area()` | 22/22 | 0.016 SD |
| `||` + which_independent (JMbayes2 docs example) | close to the published output | - |
| Dynamic predictions (on R's draws, r_sampling=True) | max abs. difference in risk < 0.01 | - |

## Scope of this version

See the table in README.md. Anything unsupported raises `NotImplementedError` rather than
fitting a different model.

## Next steps

* More outcome families (Poisson, negative binomial, Student-t, beta, Gamma, censored normal).
* Competing risks / strata, left truncation, interval censoring, time-varying survival covariates.
* Remaining association forms and transformations; interactions in functional forms.
* Marginal DIC/WAIC/LPML; predictive accuracy tools (AUC, Brier score, calibration).
* Faster set-up for large data (initial fits on the GPU with automatic differentiation).
* Tutorial notebook for clinicians.
