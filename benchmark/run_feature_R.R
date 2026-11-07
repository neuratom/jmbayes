# Reference fits in JMbayes2 for the feature tests of jmbayes.
# Usage: Rscript run_feature_R.R <model> <out.json> [n_iter n_burnin n_chains seed]
# Models (same definitions as tests/feature_models.py):
#   nhc     fixed effects without a matching random effect (random intercept only)
#   splines natural cubic splines ns(year, 2) in fixed and random effects
args <- commandArgs(trailingOnly = TRUE)
model <- args[1]; out_file <- args[2]
n_iter <- if (length(args) > 2) as.integer(args[3]) else 15000L
n_burnin <- if (length(args) > 3) as.integer(args[4]) else 3000L
n_chains <- if (length(args) > 4) as.integer(args[5]) else 12L
seed <- if (length(args) > 5) as.integer(args[6]) else 123L
suppressPackageStartupMessages({library(JMbayes2); library(jsonlite)})
pbc2.id$status2 <- as.numeric(pbc2.id$status != 'alive')

if (model == "nhc") {
    CoxFit <- coxph(Surv(years, status2) ~ drug + age, data = pbc2.id)
    fms <- list(lme(log(serBilir) ~ year + drug + age, data = pbc2, random = ~ 1 | id),
                lme(albumin ~ year + sex, data = pbc2, random = ~ year | id))
} else if (model == "splines") {
    CoxFit <- coxph(Surv(years, status2) ~ sex, data = pbc2.id)
    fms <- list(lme(log(serBilir) ~ ns(year, 2) + sex, data = pbc2,
                    random = ~ ns(year, 2) | id, control = lmeControl(opt = "optim")))
} else if (model == "forms") {
    CoxFit <- coxph(Surv(years, status2) ~ sex, data = pbc2.id)
    fms <- list(lme(log(serBilir) ~ year * sex, data = pbc2, random = ~ year | id),
                lme(prothrombin ~ year, data = pbc2, random = ~ year | id))
    fforms <- ~ value(log(serBilir)) + slope(log(serBilir)) + area(prothrombin)
} else stop("unknown model")
if (!exists("fforms")) fforms <- NULL

t0 <- proc.time()
fit <- jm(CoxFit, fms, time_var = "year", n_iter = n_iter, n_burnin = n_burnin,
          n_thin = 1L, n_chains = n_chains, seed = seed, functional_forms = fforms)
el <- (proc.time() - t0)[["elapsed"]]
s <- summary(fit)
st <- fit$statistics
blocks <- c("gammas", "alphas", paste0("betas", seq_along(fms)), "sigmas", "D")
stats <- lapply(blocks, function (nm)
    list(block = nm, names = names(st$Mean[[nm]]), Mean = unname(st$Mean[[nm]]),
         SD = unname(st$SD[[nm]]), SE = unname(st$SE[[nm]]),
         lo = unname(st$CI_low[[nm]]), hi = unname(st$CI_upp[[nm]]),
         P = if (!is.null(st$P[[nm]])) unname(st$P[[nm]]) else NULL))
# the separate fits, so Python can be started from exactly the same values
one <- function (f) {
    b <- data.matrix(ranef(f))
    list(betas = unname(fixef(f)), vcov = unname(vcov(f)),
         D = unname(matrix(getVarCov(f), ncol(b))), b = unname(b), b_ids = rownames(b),
         log_sigma = log(f$sigma))
}
res <- list(model = model, stats = stats,
            initfits = list(outcomes = lapply(fms, one), gammas = unname(coef(CoxFit))),
            fit_stats = as.list(fit$fit_stats$conditional[c("DIC", "pD", "LPML", "WAIC")]),
            settings = list(n_iter = n_iter, n_burnin = n_burnin, n_chains = n_chains,
                            seed = seed), elapsed = el)
writeLines(toJSON(res, auto_unbox = TRUE, digits = NA, pretty = TRUE), out_file)
cat(sprintf("%s: jm() %.1fs -> %s\n", model, el, out_file))
