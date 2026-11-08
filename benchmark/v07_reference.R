# Reference values for tests/test_v07.py: the separate survival fits of the 0.7 models.
suppressPackageStartupMessages({library(survival); library(jsonlite)})
here <- "benchmark/data_ext"
tv <- read.csv(file.path(here, "tvc_surv.csv"))
tv$sex <- factor(tv$sex, levels = c("male", "female"))
cx <- coxph(Surv(tstart, tstop, status2) ~ sex + edema_tv, data = tv)
ic <- read.csv(file.path(here, "icens_surv.csv"))
ic$sex <- factor(ic$sex, levels = c("male", "female"))
sr <- survreg(Surv(t1, t2, ic_status, type = "interval") ~ sex, data = ic)
out <- list(cox_tvc = unname(coef(cx)),
            weibull_icens = unname(-coef(sr)[-1] / sr$scale))
writeLines(toJSON(out, digits = NA), "benchmark/R_v07_reference.json")
print(out)
