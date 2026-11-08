# Reference values for tests/test_v08.py
suppressPackageStartupMessages({library(JMbayes2); library(jsonlite)})
pbc2.id$status2 <- as.numeric(pbc2.id$status != "alive")
pbc2.id$w <- as.numeric(pbc2.id$sex == "female") + 1
cw <- coxph(Surv(years, status2) ~ sex + age, data = pbc2.id, weights = w)
x <- c(0.01, 0.5, 2, 5, 10, 14)
kn <- seq(sqrt(.Machine$double.eps), 14.4 + 0.001, length.out = 10)
B <- ns(x, knots = kn[-c(1, 10)], intercept = TRUE, Boundary.knots = kn[c(1, 10)])
klog <- log(seq(sqrt(.Machine$double.eps), 14.4 + 0.001, length.out = 2))
Bw <- ns(log(x), knots = klog[-c(1, 2)], intercept = TRUE, Boundary.knots = klog[c(1, 2)])
kp <- seq(sqrt(.Machine$double.eps), 14.4 + 0.001, length.out = 10)
Bp <- splineDesign(kp, x, ord = 1, outer.ok = TRUE)
writeLines(toJSON(list(cox_weighted = unname(coef(cw)), x = x, ns_int = unname(B[, ]),
                       weibull = unname(Bw[, ]), pwc = unname(Bp)), digits = NA),
           "benchmark/R_v08_reference.json")
cat("ok\n")
