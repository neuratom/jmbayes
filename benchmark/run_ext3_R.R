# Reference runs in JMbayes2 (R) for the features added in jmbayes 0.8, for comparison
# with run_ext.py (same models):
#   f8a  value + velocity(eps = 0.01, backward) + area(time_window = 2); spline marker
#   f8b  vexpit(value) + Dexp(slope) + acceleration; spline marker
#   f8c  value:sex + Delta(time_window = 3); spline marker
#   tva  time-varying association: value * ns(year, k = c(3, 6, 9), B = c(0, 14.5))
#   bhw / bhp / bhn  Weibull / piecewise-constant / natural-spline baseline hazard
#   ms   multi-state (illness-death) model, the JMbayes2 vignette data (make_ms_data.R)
#   wts  case weights (2 for women, 1 for men)
#   pen  horseshoe prior on the associations, ridge prior on the covariate effects
#   aids / prothro  other data sets shipped with JMbayes2 (make_other_data.R)
# Usage: Rscript run_ext3_R.R <model> <out.json> [n_iter n_burnin n_chains seed]
args <- commandArgs(trailingOnly = TRUE)
which_model <- args[1]
out_file <- args[2]
n_iter <- if (length(args) > 2) as.integer(args[3]) else 15000L
n_burnin <- if (length(args) > 3) as.integer(args[4]) else 2500L
n_chains <- if (length(args) > 4) as.integer(args[5]) else 8L
seed <- if (length(args) > 5) as.integer(args[6]) else 123L
suppressPackageStartupMessages({library(JMbayes2); library(jsonlite)})
pbc2.id$status2 <- as.numeric(pbc2.id$status != "alive")
CoxFit <- coxph(Surv(years, status2) ~ sex, data = pbc2.id)
lin <- function () lme(log(serBilir) ~ year * sex, data = pbc2, random = ~ year | id)
spl <- function () lme(log(serBilir) ~ ns(year, 2) + sex, data = pbc2,
                       random = ~ ns(year, 2) | id,
                       control = lmeControl(opt = "optim"))
long <- NULL; forms <- NULL; base_hazard <- NULL; priors <- NULL; time_var <- "year"
Surv_obj <- CoxFit
if (which_model == "f8a") {
    long <- list(spl())
    forms <- ~ value(log(serBilir)) + velocity(log(serBilir), eps = 0.01,
        direction = "backward") + area(log(serBilir), time_window = 2)
} else if (which_model == "f8b") {
    long <- list(spl())
    forms <- ~ vexpit(value(log(serBilir))) + Dexp(slope(log(serBilir))) +
        acceleration(log(serBilir))
} else if (which_model == "f8c") {
    long <- list(spl())
    forms <- ~ value(log(serBilir)):sex + Delta(log(serBilir), time_window = 3)
} else if (which_model == "tva") {
    long <- list(lin())
    forms <- ~ value(log(serBilir)) * ns(year, k = c(3, 6, 9), B = c(0, 14.5))
} else if (which_model %in% c("bhw", "bhp", "bhn")) {
    long <- list(lin())
    base_hazard <- c(bhw = "weibull", bhp = "piecewise constant",
                     bhn = "natural cubic spline")[[which_model]]
} else if (which_model == "wts") {
    pbc2.id$w <- as.numeric(pbc2.id$sex == "female") + 1
    Surv_obj <- coxph(Surv(years, status2) ~ sex, data = pbc2.id, weights = w,
                      model = TRUE)
    long <- list(lin())
} else if (which_model == "pen") {
    Surv_obj <- coxph(Surv(years, status2) ~ sex + age + drug, data = pbc2.id)
    long <- list(lin(), lme(prothrombin ~ year * sex, data = pbc2, random = ~ year | id))
    forms <- list("log(serBilir)" = ~ value(log(serBilir)) + slope(log(serBilir)) +
                      area(log(serBilir)))
    priors <- list(penalty_alphas = "horseshoe", penalty_gammas = "ridge")
} else if (which_model == "ms") {
    df_long2 <- read.csv("benchmark/data_ext/ms_long.csv")
    df_surv <- read.csv("benchmark/data_ext/ms_surv.csv")
    df_surv$transition <- factor(df_surv$transition)
    Surv_obj <- coxph(Surv(Tstart, Tstop, status) ~ X:strata(transition), data = df_surv)
    long <- list(lme(y ~ time, random = ~ time | id, data = df_long2))
    forms <- ~ value(y):transition
    base_hazard <- rep("weibull", 3)
    time_var <- "time"
} else if (which_model == "aids") {
    Surv_obj <- coxph(Surv(Time, death) ~ drug + prevOI, data = aids.id)
    long <- list(lme(CD4 ~ obstime + obstime:drug, random = ~ obstime | patient,
                     data = aids))
    time_var <- "obstime"
} else if (which_model == "prothro") {
    Surv_obj <- coxph(Surv(Time, death) ~ treat, data = prothros)
    long <- list(lme(pro ~ time * treat, random = ~ time | id, data = prothro))
    time_var <- "time"
} else stop("unknown model")
fit <- jm(Surv_obj, long, time_var = time_var, functional_forms = forms,
          base_hazard = base_hazard, priors = priors, n_iter = n_iter,
          n_burnin = n_burnin, n_thin = 1L, n_chains = n_chains, seed = seed)
st <- fit$statistics
blocks <- c("gammas", "alphas", paste0("betas", seq_along(long)), "sigmas", "D",
            "bs_gammas", "tau_bs_gammas")
blocks <- blocks[blocks %in% names(st$Mean) & sapply(blocks, function (b)
    length(st$Mean[[b]]) > 0 && any(st$SD[[b]] > 0))]
stats <- lapply(blocks, function (nm)
    list(block = nm, names = names(st$Mean[[nm]]), Mean = unname(st$Mean[[nm]]),
         SD = unname(st$SD[[nm]]), SE = unname(st$SE[[nm]])))
fs <- c("DIC", "pD", "LPML", "WAIC")
res <- list(stats = stats,
            fit_stats = as.list(fit$fit_stats$conditional[fs]),
            fit_stats_marginal = as.list(fit$fit_stats$marginal[fs]),
            GK_k = fit$control$GK_k,
            mcmc_time = fit$running_time[["elapsed"]])
writeLines(toJSON(res, auto_unbox = TRUE, digits = 10, pretty = TRUE), out_file)
cat(sprintf("done: MCMC %.1fs\n", res$mcmc_time))
