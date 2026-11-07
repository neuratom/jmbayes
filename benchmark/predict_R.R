# Reference dynamic predictions in JMbayes2 for the README model.
# Patients 25 and 93, using their measurements up to t0 = 5 years (as in the JMbayes2
# Dynamic Predictions vignette). Usage: Rscript predict_R.R <out.json>
args <- commandArgs(trailingOnly = TRUE)
n_samples <- if (length(args) > 1) as.integer(args[2]) else 200L
suppressPackageStartupMessages({library(JMbayes2); library(jsonlite)})
pbc2.id$status2 <- as.numeric(pbc2.id$status != 'alive')
CoxFit <- coxph(Surv(years, status2) ~ sex, data = pbc2.id)
fm1 <- lme(log(serBilir) ~ year * sex, data = pbc2, random = ~ year | id)
fm2 <- lme(prothrombin ~ year * sex, data = pbc2, random = ~ year | id)
fm3 <- mixed_model(ascites ~ year + sex, data = pbc2, random = ~ year | id,
                   family = binomial())
fit <- jm(CoxFit, list(fm1, fm2, fm3), time_var = "year",
          n_iter = 12000L, n_burnin = 2000L, n_thin = 5L)
t0 <- 5
ND <- pbc2[pbc2$id %in% c(25, 93), ]
ND <- ND[ND$year < t0, ]
ND$status2 <- 0
ND$years <- t0
pS <- predict(fit, newdata = ND, process = "event", times = seq(t0, 12, length.out = 8),
              return_newdata = TRUE, control = list(cores = 1L, n_samples = n_samples))
pL <- predict(fit, newdata = ND, return_newdata = TRUE, control = list(cores = 1L, n_samples = n_samples))
out <- list(event = pS[, c("id", "years", "pred_CIF", "low_CIF", "upp_CIF")],
            long = pL[, c("id", "year", grep("^(pred|low|upp)_", names(pL), value = TRUE))])
writeLines(toJSON(out, digits = NA, pretty = TRUE), args[1])
print(out$event)
