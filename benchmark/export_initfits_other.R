# Separate fits jm() starts from for the aids model (run_ext3_R.R), for
# run_ext.py --initial-fits. Usage: Rscript export_initfits_other.R aids out.json
args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages({library(JMbayes2); library(jsonlite)})
stopifnot(args[1] == "aids")
cox <- coxph(Surv(Time, death) ~ drug + prevOI, data = aids.id)
f <- lme(CD4 ~ obstime + obstime:drug, random = ~ obstime | patient, data = aids)
D <- getVarCov(f); b <- data.matrix(ranef(f))
one <- list(betas = unname(fixef(f)), vcov = unname(vcov(f)), D = unname(matrix(D, nrow(D))),
            b = unname(b), b_ids = rownames(b), log_sigma = log(f$sigma))
writeLines(toJSON(list(outcomes = list(one), gammas = unname(coef(cox))), digits = NA,
                  auto_unbox = TRUE), args[2])
cat("D", D, "sigma", f$sigma, "\n")
