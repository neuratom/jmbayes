# Reference runs in JMbayes2 (R) for the features added in jmbayes 0.6, for comparison
# with run_ext.py. Two models on PBC:
#   surv: delayed entry (entry at 20% of each patient's follow-up, so Surv(start, stop,
#         event), which JMbayes2 fits with 7-point quadrature) and a baseline hazard per
#         treatment arm (strata(drug)); markers log serum bilirubin and prothrombin time
#   fam:  three markers with the new families: platelets (negative binomial), rounded
#         serum bilirubin (Poisson) and prothrombin time (Student's t, df = 4)
# Usage: Rscript run_ext_R.R <surv|fam> <out.json> [n_iter n_burnin n_chains seed]
args <- commandArgs(trailingOnly = TRUE)
which_model <- args[1]
out_file <- args[2]
n_iter <- if (length(args) > 2) as.integer(args[3]) else 15000L
n_burnin <- if (length(args) > 3) as.integer(args[4]) else 2500L
n_chains <- if (length(args) > 4) as.integer(args[5]) else 8L
seed <- if (length(args) > 5) as.integer(args[6]) else 123L
suppressPackageStartupMessages({library(JMbayes2); library(jsonlite)})
pbc2.id$status2 <- as.numeric(pbc2.id$status != "alive")
pbc2$bili_count <- round(pbc2$serBilir)
if (which_model == "surv") {
    pbc2.id$entry <- 0.2 * pbc2.id$years
    CoxFit <- coxph(Surv(entry, years, status2) ~ sex + strata(drug), data = pbc2.id)
    fm1 <- lme(log(serBilir) ~ year * sex, data = pbc2, random = ~ year | id)
    fm2 <- lme(prothrombin ~ year * sex, data = pbc2, random = ~ year | id)
    long <- list(fm1, fm2)
    blocks <- c("gammas", "alphas", "betas1", "betas2", "sigmas", "D", "bs_gammas",
                "tau_bs_gammas")
} else {
    CoxFit <- coxph(Surv(years, status2) ~ sex, data = pbc2.id)
    fm1 <- mixed_model(platelets ~ year * sex, data = pbc2, random = ~ year | id,
                       family = negative.binomial())
    fm2 <- mixed_model(bili_count ~ year + sex, data = pbc2, random = ~ year | id,
                       family = poisson())
    fm3 <- mixed_model(prothrombin ~ year * sex, data = pbc2, random = ~ year | id,
                       family = students.t(df = 4))
    long <- list(fm1, fm2, fm3)
    blocks <- c("gammas", "alphas", "betas1", "betas2", "betas3", "sigmas", "D",
                "bs_gammas", "tau_bs_gammas")
}
fit <- jm(CoxFit, long, time_var = "year", n_iter = n_iter, n_burnin = n_burnin,
          n_thin = 1L, n_chains = n_chains, seed = seed)
st <- fit$statistics
stats <- lapply(blocks, function (nm)
    list(block = nm, names = names(st$Mean[[nm]]), Mean = unname(st$Mean[[nm]]),
         SD = unname(st$SD[[nm]]), SE = unname(st$SE[[nm]])))
res <- list(stats = stats,
            fit_stats = as.list(fit$fit_stats$conditional[c("DIC", "pD", "LPML", "WAIC")]),
            GK_k = fit$control$GK_k,
            mcmc_time = fit$running_time[["elapsed"]])
writeLines(toJSON(res, auto_unbox = TRUE, digits = 10, pretty = TRUE), out_file)
cat(sprintf("done: MCMC %.1fs\n", res$mcmc_time))
