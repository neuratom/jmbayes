# Export the separate fits that jm() starts from (README model on pbc2), so that
# jmbayes can be run with exactly the same starting values and prior centres.
# Usage: Rscript export_initfits.R R_initfits.json
args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages({library(JMbayes2); library(jsonlite)})
pbc2.id$status2 <- as.numeric(pbc2.id$status != 'alive')
CoxFit <- coxph(Surv(years, status2) ~ sex, data = pbc2.id)
fm1 <- lme(log(serBilir) ~ year * sex, data = pbc2, random = ~ year | id)
fm2 <- lme(prothrombin ~ year * sex, data = pbc2, random = ~ year | id)
fm3 <- mixed_model(ascites ~ year + sex, data = pbc2,
                   random = ~ year | id, family = binomial())
# same extraction as jm(): extract_D(), extract_log_sigmas(), extract_b(), vcov2()
one <- function (f) {
    is_lme <- inherits(f, "lme")
    D <- if (is_lme) getVarCov(f) else f$D
    b <- data.matrix(ranef(f))
    list(betas = unname(fixef(f)),
         vcov = unname(if (is_lme) vcov(f) else vcov(f, "fixed-effects")),
         D = unname(matrix(D, nrow(D))),
         b = unname(b), b_ids = rownames(b),
         log_sigma = if (is_lme) log(f$sigma) else -20)
}
out <- list(outcomes = list(one(fm1), one(fm2), one(fm3)), gammas = unname(coef(CoxFit)))
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE), args[1])
cat("wrote", args[1], "\n")
