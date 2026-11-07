# Reference run: the README joint model in JMbayes2 (R), for comparison with jmbayes.
# Usage: Rscript run_pbc2_R.R <replicate_factor> <out.json>
args <- commandArgs(trailingOnly = TRUE)
k <- as.integer(args[1]); out_file <- args[2]
n_iter <- if (length(args) > 2) as.integer(args[3]) else 12000L
n_burnin <- if (length(args) > 3) as.integer(args[4]) else 2000L
n_chains <- if (length(args) > 4) as.integer(args[5]) else 3L
n_thin <- if (length(args) > 5) as.integer(args[6]) else 5L
seed <- if (length(args) > 6) as.integer(args[7]) else 123L
suppressPackageStartupMessages({library(JMbayes2); library(jsonlite)})
t_start <- proc.time()
# pbc2 copied k times with new ids (k = 1 is the original data)
rep_data <- function (d) do.call(rbind, lapply(seq_len(k) - 1, function (r) {
    d$id <- as.integer(as.character(d$id)) + 10000L * r; d }))
pbc2r <- rep_data(pbc2); pbc2r <- pbc2r[order(pbc2r$id, pbc2r$year), ]
pbc2.idr <- rep_data(pbc2.id)
pbc2.idr$status2 <- as.numeric(pbc2.idr$status != 'alive')
CoxFit <- coxph(Surv(years, status2) ~ sex, data = pbc2.idr)
# nlme's default optimiser can report "false convergence" on the copied data;
# fall back to optim() only in that case (pbc2 itself uses the default)
lme2 <- function (...) tryCatch(lme(...), error = function (e)
    lme(..., control = lmeControl(opt = "optim")))
fm1 <- lme2(log(serBilir) ~ year * sex, data = pbc2r, random = ~ year | id)
fm2 <- lme2(prothrombin ~ year * sex, data = pbc2r, random = ~ year | id)
fm3 <- mixed_model(ascites ~ year + sex, data = pbc2r,
                   random = ~ year | id, family = binomial())
t_fits <- proc.time()
jointFit <- jm(CoxFit, list(fm1, fm2, fm3), time_var = "year",
               n_iter = n_iter, n_burnin = n_burnin, n_thin = n_thin,
               n_chains = n_chains, seed = seed)
t_end <- proc.time()
s <- summary(jointFit)
tab <- function (d) lapply(seq_len(nrow(d)), function (i)
    list(name = rownames(d)[i], Mean = d$Mean[i], StDev = d$StDev[i],
         `2.5%` = d$`2.5%`[i], `97.5%` = d$`97.5%`[i], P = d$P[i], Rhat = d$Rhat[i]))
# posterior mean, SD and Monte Carlo standard error for every parameter block
st <- jointFit$statistics
stats <- lapply(c("gammas", "alphas", "betas1", "betas2", "betas3", "sigmas", "D",
                  "bs_gammas", "tau_bs_gammas"), function (nm)
    list(block = nm, names = names(st$Mean[[nm]]), Mean = unname(st$Mean[[nm]]),
         SD = unname(st$SD[[nm]]), SE = unname(st$SE[[nm]])))
res <- list(
    stats = stats,
    gammas_alphas = tab(s$Survival),
    betas1 = tab(s$Outcome1), betas2 = tab(s$Outcome2), betas3 = tab(s$Outcome3),
    fit_stats = as.list(jointFit$fit_stats$conditional[c("DIC", "pD", "LPML", "WAIC")]),
    timing = list(separate_fits = (t_fits - t_start)[["elapsed"]],
                  jm_total = (t_end - t_fits)[["elapsed"]],
                  mcmc = jointFit$running_time[["elapsed"]],
                  cores = jointFit$control$cores),
    n = nrow(pbc2.idr))
writeLines(toJSON(res, auto_unbox = TRUE, digits = 10, pretty = TRUE), out_file)
cat(sprintf("k=%d n=%d  separate fits %.1fs  jm() %.1fs (MCMC %.1fs)\n", k, res$n,
            res$timing$separate_fits, res$timing$jm_total, res$timing$mcmc))
