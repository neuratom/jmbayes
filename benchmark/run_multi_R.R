# Many independent JMbayes2 runs of one model per data set, for the run-distribution
# comparison (multi_compare.py). Saves posterior mean, SD and 95% interval of every
# parameter.
#   main     PBC: log bilirubin, prothrombin, ascites (binomial); Cox ~ sex
#   aids     AIDS: sqrt(CD4); Cox ~ drug + prevOI
#   prothro  liver cirrhosis: prothrombin index; Cox ~ treat
# Usage: Rscript run_multi_R.R <model> <out.json> <seed> [n_iter n_burnin n_chains]
args <- commandArgs(trailingOnly = TRUE)
which_model <- args[1]; out_file <- args[2]; seed <- as.integer(args[3])
n_iter <- if (length(args) > 3) as.integer(args[4]) else 15000L
n_burnin <- if (length(args) > 4) as.integer(args[5]) else 2500L
n_chains <- if (length(args) > 5) as.integer(args[6]) else 8L
suppressPackageStartupMessages({library(JMbayes2); library(jsonlite)})
if (which_model == "main") {
    pbc2.id$status2 <- as.numeric(pbc2.id$status != "alive")
    S <- coxph(Surv(years, status2) ~ sex, data = pbc2.id)
    long <- list(lme(log(serBilir) ~ year * sex, data = pbc2, random = ~ year | id),
                 lme(prothrombin ~ year * sex, data = pbc2, random = ~ year | id),
                 mixed_model(ascites ~ year + sex, data = pbc2, random = ~ year | id,
                             family = binomial()))
    tv <- "year"
} else if (which_model == "aids") {
    S <- coxph(Surv(Time, death) ~ drug + prevOI, data = aids.id)
    long <- list(lme(CD4 ~ obstime + obstime:drug, random = ~ obstime | patient,
                     data = aids))
    tv <- "obstime"
} else if (which_model == "prothro") {
    S <- coxph(Surv(Time, death) ~ treat, data = prothros)
    long <- list(lme(pro ~ time * treat, random = ~ time | id, data = prothro))
    tv <- "time"
} else stop("unknown model")
fit <- jm(S, long, time_var = tv, n_iter = n_iter, n_burnin = n_burnin, n_thin = 1L,
          n_chains = n_chains, seed = seed)
st <- fit$statistics
blocks <- c("gammas", "alphas", paste0("betas", seq_along(long)), "sigmas", "D",
            "bs_gammas", "tau_bs_gammas")
blocks <- blocks[blocks %in% names(st$Mean) & sapply(blocks, function (b)
    length(st$Mean[[b]]) > 0 && any(st$SD[[b]] > 0))]
res <- lapply(blocks, function (nm)
    list(block = nm, Mean = unname(st$Mean[[nm]]), SD = unname(st$SD[[nm]]),
         lo = unname(st$CI_low[[nm]]), hi = unname(st$CI_upp[[nm]])))
writeLines(toJSON(list(stats = res, mcmc_time = fit$running_time[["elapsed"]]),
                  auto_unbox = TRUE, digits = 10), out_file)
cat(sprintf("done: MCMC %.1fs\n", fit$running_time[["elapsed"]]))
