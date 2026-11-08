# Export the separate fits that jm() starts from, for the 'fam' model of run_ext_R.R, so
# that jmbayes can be run from exactly the same starting values and prior centres.
# Needed because jm() centres some priors on these fits, and here GLMMadaptive's
# Student's t fit of prothrombin does not converge properly (R warns that the Hessian is
# not positive definite), while jmbayes's own fit does; with different prior centres the
# two programs would fit slightly different models.
# Usage: Rscript export_initfits_ext.R R_fam_initfits.json
args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages({library(JMbayes2); library(jsonlite)})
pbc2.id$status2 <- as.numeric(pbc2.id$status != "alive")
pbc2$bili_count <- round(pbc2$serBilir)
CoxFit <- coxph(Surv(years, status2) ~ sex, data = pbc2.id)
fm1 <- mixed_model(platelets ~ year * sex, data = pbc2, random = ~ year | id,
                   family = negative.binomial())
fm2 <- mixed_model(bili_count ~ year + sex, data = pbc2, random = ~ year | id,
                   family = poisson())
fm3 <- mixed_model(prothrombin ~ year * sex, data = pbc2, random = ~ year | id,
                   family = students.t(df = 4))
# same extraction as jm(): extract_D(), extract_log_sigmas(), extract_b(), vcov2()
one <- function (f) {
    b <- data.matrix(ranef(f))
    list(betas = unname(fixef(f)), vcov = unname(vcov(f, "fixed-effects")),
         D = unname(matrix(f$D, nrow(f$D))), b = unname(b), b_ids = rownames(b),
         log_sigma = if (is.null(f$phis)) -20 else unname(f$phis))
}
out <- list(outcomes = list(one(fm1), one(fm2), one(fm3)), gammas = unname(coef(CoxFit)))
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE), args[1])
cat("wrote", args[1], "\n")
