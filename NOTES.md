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

6. **Separate fits that jm() starts from (GLMMadaptive 0.9-7).** jm() takes its starting
   values and centres some priors (fixed effects, D, sigmas) on the separate mixed-model
   fits, so when these stop short the joint model's priors move with them. Seen on PBC:
   Student's t (prothrombin; non-positive-definite Hessian), censored normal (log alkaline
   phosphatase censored at its 20% quantile: the fit is 550 log-likelihood units below the
   maximum jmbayes finds), and smaller gaps for the negative binomial, gamma and
   beta-binomial fits. `beta.binomial()` fails in its EM phase on these data (it works with
   `control = list(iter_EM = 0)`), and `unit.lindley()` reports that the family is
   unavailable, although JMbayes2's C++ code has its density. The comparisons therefore
   start both programs from R's separate fits (`--initial-fits`).

7. **Hierarchical centering with splines in interactions** (fixed in jmbayes, not kept).
   `create_HC_X()` finds the multiplier of a random effect in a fixed-effect interaction
   by setting the data variable of the random-effect column to 1 and rebuilding the
   design. For `year` that gives the right multiplier (`year:sex` -> `sex`), but for a
   function of it, e.g. `ns(year, 2)1:sex` with random `~ ns(year, 2)`, it gives
   `ns(1)1 * sex`. The centring then no longer leaves the linear predictor unchanged, so
   the Gibbs step for the fixed effects ignores how the data depend on them: it moves
   them (and every patient's trajectory) by large amounts that are never accepted or
   rejected. Seen with `log(serBilir) ~ ns(year, 2) * sex`: a fixed-effect draw moved
   one patient's `exp(value) * slope` term 300-fold. jmbayes divides by the
   random-effect column evaluated on the same modified data (exactly 1 for `year`
   itself, so models without such terms are unchanged, bit for bit). The comparisons
   use additive spline models (`ns(year, 2) + sex`), where both programs are right.
8. **`Dexpit()`** (not kept). `extractFuns_FunForms()` matches transformation names
   case-sensitively, so `Dexpit(slope(y))` is fitted as `expit(value(y)) * slope(y)`
   (checked: `model_info$Funs_FunForms` lists "expit"), not as the derivative of
   expit(eta(t)). jmbayes uses expit(value) (1 - expit(value)) * slope. `Dexp()` is not
   affected (the derivative of exp is exp).

9. **`ms_setup(..., covs = )` with one covariate** (not kept). The covariate is repeated
   in the original row order of the data (`rep(covs, n_per_subject)`) and then reordered
   with the sort of the long rows (`[ord]`), so values land on other patients' rows (on
   the vignette's illness-death data, patient 2's X = 1 became 0). The transitions
   themselves are right; jmbayes.ms_setup() gives the same rows and joins the covariates
   by patient id.

10. **Fit statistics for binomial markers with trials and beta-binomial markers**
   (cause not traced). On the fam3 model JMbayes2 reports conditional pD = 7,424 with
   DIC = 14,847 (pD about half the DIC, as if the log-likelihood at the posterior means
   were ~0), and marginal DIC 0, pD 0, LPML Inf, WAIC NA. jmbayes gives conditional DIC
   about 7,300 with pD about 170. The parameter estimates themselves agree.

11. **jmbayes, fixed in 0.8: float32 rounding in lgamma-difference densities.** The
   beta-binomial, negative binomial, beta and gamma densities are differences of large
   lgamma terms when the dispersion is large (lgamma(y + A) - lgamma(A), A ~ 1e4). In
   float32 the rounding error exceeded the signal (the log-likelihood at phi = 1e6 came
   out 780 units *higher* than at phi = 441; in float64 they are equal), so the sampler
   drifted to huge dispersions (fam3: R-hat > 1000). These densities are now evaluated in
   float64 whatever the sampler's dtype. Found by the fam3 comparison.

## Equivalence with JMbayes2 (details in benchmark/ and tests/)

| Model | Parameters equivalent at +-0.2 SD | Median abs. difference R vs Python |
|---|---|---|
| README model (3 markers, value) | 45/50 (none shown to differ) | 0.018 SD (R vs R: 0.023) |
| Fixed effects without a matching random effect | 19/19 | 0.006 SD |
| Splines `ns(year, 2)` | 13/13 | 0.029 SD |
| `value()` + `slope()` + `area()` | 22/22 | 0.016 SD |
| `||` + which_independent (JMbayes2 docs example) | close to the published output | - |
| Dynamic predictions (on R's draws, r_sampling=True) | max abs. difference in risk < 0.01 | - |

## Comparison status for 0.7 and 0.8 (done, 2026-11-08)

All 21 extension models were run in both programs (8 chains x 15,000 iterations, burn-in
2,500, seeds 123 and 456; the two 0.6 models 12 x 30,000), compared with
`benchmark/compare_cr.py <model>`. The table is in the README and the paper
(`benchmark/ext_table.py`, `benchmark/hr_table.py`).

* No parameter in any model was shown to differ.
* Inconclusive for many parameters, with two runs of the same program differing about as
  much as R vs Python (slow mixing, needs longer runs): `crx`, `rec`, `bystr`, `fam3`
  (random intercepts of the two markers correlate at 0.98, so the associations are nearly
  confounded), `ms`.
* Timings were not controlled (runs overlapped each other and the test suite; e.g. R fam3
  took 1,810 s and 156 s for the two seeds), so they are not reported for these models.
* Marginal DIC agrees closely; marginal WAIC and especially LPML are noisy (LPML depends on
  a few patients' extreme values) and differ by more in some models.
* Two other data sets that ship with JMbayes2 (`aids`, `prothro`; make_other_data.R): all
  22 parameters equivalent within 0.2 SD in each.
* Rscript reads its script while running: editing run_ext3_R.R during the runs broke two
  runs at the end (wts 456, ms 123); they were rerun.

Repeated runs (2026-11-09): 30 runs per program, seeds 1-30, of the main PBC model, aids
and prothro (benchmark/multi/, run_multi_all.sh + run_multi_more*.sh, multi_compare.py).
All 94 parameters equivalent (mean within 0.1 SD, posterior SD within 10%), 87 also within
0.05 SD; 0 differ after BH; 30 runs give >= 80% power for every parameter (most demanding
needed 22). Raw p < 0.05 for 9/94 means (largest |d| 0.048 SD). Run-to-run SD equal in
both programs. These runs overlapped each other, so their timings are not benchmarks.

12. **jmbayes, fixed in 0.8: separate linear mixed model at a poor optimum.** On the aids
   data statsmodels' L-BFGS reported convergence at REML log-likelihood -4305.7, against
   -3566.8 at R's lme() solution (random slope variance 8.55 vs 0.030). The priors centred
   on that fit then moved D and sigma (D[2,2] differed by 0.56 SD). fit_lme() now also tries
   Powell's method and keeps the better fit (L-BFGS is kept unless another is better by
   0.01, so PBC results are unchanged, bit for bit). Not for uncorrelated random effects
   (`||`): Powell and Nelder-Mead ignore statsmodels' `free` constraint and return
   non-zero covariances, so there only L-BFGS/BFGS are used, as before.

## Scope of this version

See the table in README.md. Anything unsupported raises `NotImplementedError` rather than
fitting a different model.

## Next steps

* Faster CPU version: run groups of chains in separate processes (MIMD, as JMbayes2),
  each batching its patients and chains as tensors (SIMD, as now). Today one process
  batches everything, which on a CPU is only about as fast as JMbayes2.
* Predictions for recurrent-event and multi-state models.
* Survival covariates that are functions of the time variable (JMbayes2 evaluates them
  at the quadrature points); for now use an interaction in the functional forms.
* Faster set-up for large data (initial fits on the GPU with automatic differentiation).
* Tutorial notebook for clinicians.
