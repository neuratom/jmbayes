# Reference runs in JMbayes2 (R) for the features added in jmbayes 0.7, for comparison
# with run_ext.py (same models, same data: benchmark/data_ext/, made by ext_data.py and
# make_rec_data.R):
#   tvc    time-varying covariate (edema), counting-process rows
#   icens  interval-censored event times (survreg Weibull starting values)
#   crx    competing risks with delayed entry and a baseline hazard per cause and arm
#   rec    recurrent and terminal events, gap timescale (the JMbayes2 vignette data)
#   fam2   gamma, beta and censored normal markers
#   fam3   binomial with trials and beta-binomial markers
#   bystr  a baseline hazard, covariate effects and associations per treatment arm
# Usage: Rscript run_ext2_R.R <model> <out.json> [n_iter n_burnin n_chains seed]
#        Rscript run_ext2_R.R <model> <R_model_initfits.json> initfits
#          (writes the separate fits jm() starts from, for run_ext.py --initial-fits)
args <- commandArgs(trailingOnly = TRUE)
which_model <- args[1]
out_file <- args[2]
only_init <- length(args) > 2 && args[3] == "initfits"
n_iter <- if (length(args) > 2 && !only_init) as.integer(args[3]) else 15000L
n_burnin <- if (length(args) > 3) as.integer(args[4]) else 2500L
n_chains <- if (length(args) > 4) as.integer(args[5]) else 8L
seed <- if (length(args) > 5) as.integer(args[6]) else 123L
suppressPackageStartupMessages({library(JMbayes2); library(jsonlite)})
here <- "benchmark/data_ext"
pbc2.id$status2 <- as.numeric(pbc2.id$status != "alive")
fix <- function (d) {
    if (!is.null(d$sex)) d$sex <- factor(d$sex, levels = c("male", "female"))
    if (!is.null(d$drug)) d$drug <- factor(d$drug, levels = c("placebo", "D-penicil"))
    d
}
fm_lbili <- function (d) lme(log(serBilir) ~ year * sex, data = d, random = ~ year | id)
forms <- NULL
recurrent <- FALSE
time_var <- "year"
if (which_model == "tvc") {
    surv <- fix(read.csv(file.path(here, "tvc_surv.csv")))
    Surv_obj <- coxph(Surv(tstart, tstop, status2) ~ sex + edema_tv, data = surv)
    long <- list(fm_lbili(pbc2))
} else if (which_model == "icens") {
    surv <- fix(read.csv(file.path(here, "icens_surv.csv")))
    Surv_obj <- survreg(Surv(t1, t2, ic_status, type = "interval") ~ sex, data = surv,
                        model = TRUE)
    long <- list(fm_lbili(pbc2))
} else if (which_model == "crx") {
    p <- pbc2.id
    p$entry <- 0.2 * p$years
    pCR <- crisk_setup(p, statusVar = "status", censLevel = "alive", nameStrata = "CR")
    pCR$female <- as.numeric(pCR$sex == "female")
    pCR$CRdrug <- interaction(pCR$CR, pCR$drug, lex.order = TRUE)
    Surv_obj <- coxph(Surv(entry, years, status2) ~ female:CR + strata(CRdrug), data = pCR)
    long <- list(fm_lbili(pbc2))
    forms <- list("log(serBilir)" = ~ value(log(serBilir)):CR)
} else if (which_model == "rec") {
    lme_data <- read.csv(file.path(here, "rec_long.csv"))
    cox_data <- read.csv(file.path(here, "rec_surv.csv"))
    cox_data$strata <- factor(cox_data$strata, levels = c("R", "T1"))
    Surv_obj <- coxph(Surv(tstart, tstop, status) ~ (group + age):strata(strata),
                      data = cox_data)
    long <- list(lme(y ~ time, data = lme_data, random = ~ time | id))
    forms <- ~ value(y):strata
    recurrent <- "gap"
    time_var <- "time"
} else if (which_model == "fam2") {
    v <- fix(read.csv(file.path(here, "fam_visits.csv")))
    Surv_obj <- coxph(Surv(years, status2) ~ sex, data = pbc2.id)
    long <- list(
        mixed_model(serBilir ~ year * sex, data = v, random = ~ year | id,
                    family = Gamma.fam()),
        mixed_model(alb_p ~ year + sex, data = v, random = ~ year | id,
                    family = beta.fam()),
        mixed_model(cbind(lalk, lalk_cens) ~ year + sex, data = v, random = ~ year | id,
                    family = censored.normal()))
} else if (which_model == "fam3") {
    v <- fix(read.csv(file.path(here, "fam_visits.csv")))
    Surv_obj <- coxph(Surv(years, status2) ~ sex, data = pbc2.id)
    long <- list(
        mixed_model(cbind(count2, 2 - count2) ~ year + sex, data = v, random = ~ 1 | id,
                    family = binomial()),
        # GLMMadaptive's EM phase fails for beta.binomial() here; quasi-Newton only
        mixed_model(cbind(count3, 3 - count3) ~ year + sex, data = v, random = ~ 1 | id,
                    family = beta.binomial(), control = list(iter_EM = 0)))
    # (unit Lindley is not compared: GLMMadaptive 0.9-x reports "the 'unit.lindley()'
    # family is unavailable", so jm() cannot be given such a model)
} else if (which_model == "bystr") {
    pbc2.id$female <- as.numeric(pbc2.id$sex == "female")
    Surv_obj <- coxph(Surv(years, status2) ~ female:strata(drug), data = pbc2.id)
    long <- list(fm_lbili(pbc2),
                 lme(prothrombin ~ year * sex, data = pbc2, random = ~ year | id))
    forms <- list("log(serBilir)" = ~ value(log(serBilir)):drug,
                  "prothrombin" = ~ value(prothrombin):drug)
} else stop("unknown model")

if (only_init) {
    one <- function (f) {
        b <- data.matrix(ranef(f))
        list(betas = unname(fixef(f)), vcov = unname(vcov(f, "fixed-effects")),
             D = unname(matrix(f$D, nrow(f$D))), b = unname(b), b_ids = rownames(b),
             log_sigma = if (is.null(f$phis)) -20 else unname(f$phis))
    }
    g <- if (inherits(Surv_obj, "coxph")) coef(Surv_obj) else
        -coef(Surv_obj)[-1L] / Surv_obj$scale
    writeLines(toJSON(list(outcomes = lapply(long, one), gammas = unname(g)),
                      digits = NA, auto_unbox = TRUE), out_file)
    cat("wrote", out_file, "\n")
    quit(save = "no")
}
fit <- jm(Surv_obj, long, time_var = time_var, functional_forms = forms,
          recurrent = recurrent, n_iter = n_iter, n_burnin = n_burnin, n_thin = 1L,
          n_chains = n_chains, seed = seed)
st <- fit$statistics
blocks <- c("gammas", "alphas", paste0("betas", seq_along(long)), "sigmas", "D",
            "bs_gammas", "tau_bs_gammas", "sigmaF", "alphaF")
blocks <- blocks[blocks %in% names(st$Mean) & sapply(blocks, function (b)
    length(st$Mean[[b]]) > 0)]
if (!isTRUE(recurrent != FALSE)) blocks <- setdiff(blocks, c("sigmaF", "alphaF"))
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
