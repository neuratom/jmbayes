# Multi-state (illness-death) data: the simulation of the JMbayes2 vignette
# "Multi-State Processes" (seed 1710), saved for run_ext.py and run_ext3_R.R.
suppressPackageStartupMessages(library(JMbayes2))
set.seed(1710)

# Sample sizes and settings
N <- 1500
n_per <- 14

# Baseline covariate (binary)
X <- rbinom(N, 1, 0.5)

# Longitudinal parameters
beta <- c(5, -0.1)
sigma_e <- 1
D <- matrix(c(1.0, 0.2, 0.2, 0.3), 2, 2)
b <- MASS::mvrnorm(N, mu = c(0, 0), Sigma = D)

# Simulate longitudinal data
df_long <- do.call('rbind', lapply(1:N, function (i) {
    obs_times <- sort(c(0, runif(n_per - 1, 0, 18)))
    XX <- cbind(1, obs_times)
    eta <- XX %*% beta + XX %*% b[i, ]
    y <- rnorm(n_per, eta, sigma_e)
    data.frame(id = i, time = obs_times, y = y, X = X[i])
}))

# Weibull hazard parameters (shape, scale), association, covariate effects
weib_shape <- c(2.2, 1.8, 2.5)
weib_scale <- c(0.04, 0.03, 0.05)
alpha <- c(0.6, 0.4, 0.8) # different for each transition
gamma <- c(0.5, -0.3, 0.7) # different for each transition
Ctimes <- runif(N, 7, 10)

# Longitudinal trajectory function
yfun <- function (b_i, t) beta[1] + b_i[1] + (beta[2] + b_i[2]) * t

# Simulate event time via inverse transform
sim_time <- function (shape, scale, alpha, gamma, X, b_i, tmax = 20) {
    U <- runif(1)
    cumhaz <- function (t) {
        integrate(function (s) {
            h0 <- shape * scale * (scale * s)^(shape - 1)
            h0 * exp(alpha * yfun(b_i, s) + gamma * X)
        }, 0, t)$value
    }
    f <- function (t) cumhaz(t) + log(U)
    out <- try(uniroot(f, c(1e-5, tmax))$root, silent = TRUE)
    if (inherits(out, "try-error")) Inf else out
}

# Simulate multi-state transitions
df_surv <- do.call(rbind, lapply(1:N, function (i) {
    b_i <- b[i, ]; X_i <- X[i]; C <- Ctimes[i]
    T01 <- sim_time(weib_shape[1], weib_scale[1], alpha[1], gamma[1], X_i, b_i)
    T02 <- sim_time(weib_shape[2], weib_scale[2], alpha[2], gamma[2], X_i, b_i)
    
    if (T01 < T02 && T01 < C) {
        T12 <- sim_time(weib_shape[3], weib_scale[3], alpha[3], gamma[3], X_i, b_i)
        Tdeath <- T01 + T12
        data.frame(id = i, 
                   transition = c(1, 2, 3), 
                   Tstart = c(0, 0, T01), 
                   Tstop = c(T01, T01, min(Tdeath, C)), 
                   status = c(1, 0, as.integer(T12 < (C - T01))), 
                   X = X_i)
    } else if (T02 < C) {
        data.frame(id = i, transition = c(1, 2), Tstart = c(0, 0), Tstop = c(T02, T02), status = c(0, 1), X = X_i)
    } else {
        data.frame(id = i,
                   transition = c(1, 2),
                   Tstart = 0,
                   Tstop = C,
                   status = 0,
                   X = X_i)
    }
}))

# Filter longitudinal data to observed period
df_long2 <- merge(df_long, df_surv[, c("id", "Tstop")], by = c("id"))
df_long2 <- df_long2[!duplicated(df_long2, by = c('id', 'time')), ]
df_long2 <- df_long2[df_long2$time <= df_long2$Tstop, ]
df_surv$transition <- factor(df_surv$transition)
write.csv(df_long2[, c("id", "time", "y", "X")], "benchmark/data_ext/ms_long.csv", row.names = FALSE)
write.csv(df_surv, "benchmark/data_ext/ms_surv.csv", row.names = FALSE)
cat("rows:", nrow(df_surv), nrow(df_long2), "\n")
