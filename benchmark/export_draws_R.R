# Fit the README model (same seed as predict_R.R) and export the posterior draws, so the
# Python prediction engine can be run on exactly R's posterior sample.
args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages({library(JMbayes2); library(jsonlite)})
pbc2.id$status2 <- as.numeric(pbc2.id$status != 'alive')
CoxFit <- coxph(Surv(years, status2) ~ sex, data = pbc2.id)
fm1 <- lme(log(serBilir) ~ year * sex, data = pbc2, random = ~ year | id)
fm2 <- lme(prothrombin ~ year * sex, data = pbc2, random = ~ year | id)
fm3 <- mixed_model(ascites ~ year + sex, data = pbc2, random = ~ year | id,
                   family = binomial())
fit <- jm(CoxFit, list(fm1, fm2, fm3), time_var = "year",
          n_iter = 12000L, n_burnin = 2000L, n_thin = 5L)
st <- function (nm) unname(do.call("rbind", fit$mcmc[[nm]]))
out <- list(betas1 = st("betas1"), betas2 = st("betas2"), betas3 = st("betas3"),
            sigmas = st("sigmas"), gammas = st("gammas"), alphas = st("alphas"),
            bs_gammas = st("bs_gammas"), tau_bs_gammas = st("tau_bs_gammas"), D = st("D"))
writeLines(toJSON(out, digits = NA), args[1])
cat("draws:", nrow(out$alphas), "\n")
