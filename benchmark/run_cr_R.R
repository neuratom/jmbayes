# Reference run for competing risks in JMbayes2 (R), for comparison with jmbayes.
# PBC: death and transplantation as competing events, cause-specific effects of sex and
# of the current values of log serum bilirubin and prothrombin time.
# Usage: Rscript run_cr_R.R <out.json> [n_iter n_burnin n_chains seed]
args <- commandArgs(trailingOnly = TRUE)
out_file <- args[1]
n_iter <- if (length(args) > 1) as.integer(args[2]) else 15000L
n_burnin <- if (length(args) > 2) as.integer(args[3]) else 2500L
n_chains <- if (length(args) > 3) as.integer(args[4]) else 8L
seed <- if (length(args) > 4) as.integer(args[5]) else 123L
suppressPackageStartupMessages({library(JMbayes2); library(jsonlite)})
pbc2.idCR <- crisk_setup(pbc2.id, statusVar = "status", censLevel = "alive",
                         nameStrata = "CR")
pbc2.idCR$female <- as.numeric(pbc2.idCR$sex == "female")
CoxFit_CR <- coxph(Surv(years, status2) ~ female:strata(CR), data = pbc2.idCR)
fm1 <- lme(log(serBilir) ~ year * sex, data = pbc2, random = ~ year | id)
fm2 <- lme(prothrombin ~ year * sex, data = pbc2, random = ~ year | id)
CR_forms <- list("log(serBilir)" = ~ value(log(serBilir)):CR,
                 "prothrombin" = ~ value(prothrombin):CR)
fit <- jm(CoxFit_CR, list(fm1, fm2), time_var = "year", functional_forms = CR_forms,
          n_iter = n_iter, n_burnin = n_burnin, n_thin = 1L, n_chains = n_chains,
          seed = seed)
s <- summary(fit)
tab <- function (d) lapply(seq_len(nrow(d)), function (i)
    list(name = rownames(d)[i], Mean = d$Mean[i], StDev = d$StDev[i],
         `2.5%` = d$`2.5%`[i], `97.5%` = d$`97.5%`[i], P = d$P[i], Rhat = d$Rhat[i]))
st <- fit$statistics
stats <- lapply(c("gammas", "alphas", "betas1", "betas2", "sigmas", "D",
                  "bs_gammas", "tau_bs_gammas"), function (nm)
    list(block = nm, names = names(st$Mean[[nm]]), Mean = unname(st$Mean[[nm]]),
         SD = unname(st$SD[[nm]]), SE = unname(st$SE[[nm]])))
# JMbayes2 0.6-5's predict() fails for this model (C++ index error), so predictions
# are not compared here.
res <- list(
    stats = stats,
    gammas_alphas = tab(s$Survival), betas1 = tab(s$Outcome1), betas2 = tab(s$Outcome2),
    fit_stats = as.list(fit$fit_stats$conditional[c("DIC", "pD", "LPML", "WAIC")]),
    mcmc_time = fit$running_time[["elapsed"]])
writeLines(toJSON(res, auto_unbox = TRUE, digits = 10, pretty = TRUE), out_file)
cat(sprintf("done: MCMC %.1fs\n", res$mcmc_time))
