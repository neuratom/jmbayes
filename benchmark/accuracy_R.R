# Reference prediction-accuracy measures from JMbayes2 (tvROC, tvAUC, tvBrier,
# calibration) for the README two-marker model on pbc2, together with the exact predicted
# risks each function used. predict.jm is wrapped so that every prediction made inside
# tvROC() etc. is recorded; the tests feed those risks to the Python functions.
# Usage: Rscript accuracy_R.R <out.json>
args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages({library(JMbayes2); library(jsonlite)})
pbc2.id$status2 <- as.numeric(pbc2.id$status != "alive")
pbc2$status2 <- as.numeric(pbc2$status != "alive")
CoxFit <- coxph(Surv(years, status2) ~ sex, data = pbc2.id)
fm1 <- lme(log(serBilir) ~ year * sex, data = pbc2, random = ~ year | id)
fm2 <- mixed_model(ascites ~ year + sex, data = pbc2, random = ~ year | id,
                   family = binomial())
fit <- jm(CoxFit, list(fm1, fm2), time_var = "year")

# record every predict.jm() call made inside the accuracy functions
rec <- new.env()
rec$calls <- list()
orig_predict <- getS3method("predict", "jm")
recording_predict <- function (object, ...) {
    out <- orig_predict(object, ...)
    rec$calls[[length(rec$calls) + 1L]] <-
        list(id = as.character(out$id), times = out$times, pred = out$pred)
    out
}
# calls from inside the package find predict.jm in its namespace first
assignInNamespace("predict.jm", recording_predict, ns = "JMbayes2")
registerS3method("predict", "jm", recording_predict, envir = asNamespace("JMbayes2"))
captured <- function (expr) {
    rec$calls <- list()
    res <- expr
    list(res = res, calls = rec$calls)
}

setting <- function (Tstart0, Thoriz0, full = TRUE) {
    Tstart <- Tstart0 + 1e-06
    Thoriz <- Thoriz0 + 1e-06
    nd <- pbc2[pbc2$years > Tstart & pbc2$year <= Tstart, ]
    f <- factor(nd$id, levels = unique(nd$id))
    subj <- data.frame(id = levels(f), Time = as.numeric(tapply(nd$years, f, tail, 1L)),
                       event = as.numeric(tapply(nd$status2, f, tail, 1L)))
    roc <- captured(tvROC(fit, pbc2, Tstart = Tstart0, Thoriz = Thoriz0, cores = 1L))
    br <- captured(tvBrier(fit, pbc2, Tstart = Tstart0, Thoriz = Thoriz0, cores = 1L))
    cal <- captured(calibration_plot(fit, pbc2, Tstart = Tstart0, Thoriz = Thoriz0,
                                     plot = FALSE, cores = 1L))
    # calibration_metrics() is these three lines applied to calibration_plot()'s output
    d <- abs(as.vector(cal$res$pi_u_t) - cal$res$obs_pi_u_t)
    out <- list(Tstart = Tstart, Thoriz = Thoriz, Tmid = 0.5 * (Tstart + Thoriz),
                subjects = subj,
                roc_mb = list(TP = roc$res$TP, FP = roc$res$FP, nTP = roc$res$nTP,
                              nFN = roc$res$nFN, nFP = roc$res$nFP, nTN = roc$res$nTN,
                              F1score = roc$res$F1score, Youden = roc$res$Youden,
                              nr = roc$res$nr, auc = tvAUC(roc$res)$auc, calls = roc$calls),
                brier_mb = list(Brier = br$res$Brier, nr = br$res$nr, nint = br$res$nint,
                                ncens = br$res$ncens, calls = br$calls),
                calibration = list(observed = cal$res$observed, predicted = cal$res$predicted,
                                   ids = names(cal$res$pi_u_t),
                                   pi_u_t = as.numeric(cal$res$pi_u_t),
                                   obs_pi_u_t = cal$res$obs_pi_u_t,
                                   metrics = list(ICI = mean(d), E50 = median(d),
                                                  E90 = unname(quantile(d, 0.9)))))
    if (full) {
        ri <- captured(tvROC(fit, pbc2, Tstart = Tstart0, Thoriz = Thoriz0,
                             type_weights = "IPCW", cores = 1L))
        bi <- captured(tvBrier(fit, pbc2, Tstart = Tstart0, Thoriz = Thoriz0,
                               type_weights = "IPCW", cores = 1L))
        bint <- captured(tvBrier(fit, pbc2, Tstart = Tstart0, Thoriz = Thoriz0,
                                 integrated = TRUE, cores = 1L))
        out$roc_ipcw <- list(TP = ri$res$TP, FP = ri$res$FP, F1score = ri$res$F1score,
                             Youden = ri$res$Youden, auc = tvAUC(ri$res)$auc,
                             calls = ri$calls)
        out$brier_ipcw <- list(Brier = bi$res$Brier, calls = bi$calls)
        out$brier_int <- list(Brier = bint$res$Brier, calls = bint$calls)
    }
    out
}
out <- list(t5_dt2 = setting(5, 7), t3_t7 = setting(3, 7, full = FALSE))
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE), args[1])
cat("AUC (5, 7):", out$t5_dt2$roc_mb$auc, " Brier:", out$t5_dt2$brier_mb$Brier,
    " ICI:", out$t5_dt2$calibration$metrics$ICI, "\n")
