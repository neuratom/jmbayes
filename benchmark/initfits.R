suppressPackageStartupMessages(library(JMbayes2))
fm1 <- lme(log(serBilir) ~ year * sex, data = pbc2, random = ~ year | id)
fm2 <- lme(prothrombin ~ year * sex, data = pbc2, random = ~ year | id)
fm3 <- mixed_model(ascites ~ year + sex, data = pbc2, random = ~ year | id, family = binomial())
pd <- function(f) if (inherits(f, "lme")) getVarCov(f) else f$D
cat("sd(D):", round(c(sqrt(diag(pd(fm1))), sqrt(diag(pd(fm2))), sqrt(diag(pd(fm3)))), 5), "\n")
cat("betas3:", round(fixef(fm3), 5), "\n")
cat("sigma:", round(c(fm1$sigma, fm2$sigma), 5), "\n")
cat("logLik fm3:", logLik(fm3), "\n")
