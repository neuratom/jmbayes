# Recurrent-event data: the simulation of the JMbayes2 vignette "Recurrent Events"
# (gen_data(), seed 2022), saved for benchmark/run_ext.py and run_ext2_R.R.
suppressPackageStartupMessages(library(JMbayes2))
gen_data <- function(){
  n <- 500 # desired number of subjects 
  n_i <- 15  # number of (planned) measurements per subject
  tmax <- 7 # maximum follow-up time (type I censoring)
  scale <- "gap" # hazard timescale
  ##############################################################################
  n_scl <- 1.5
  n_target <- n
  n <- n * n_scl
  # longitudinal outcome 1/2
  ## param true values
  betas <- c("Intercept" = 6.94, "Time1" = 1.30, "Time2" = 1.84, "Time3" = 1.82)
  sigma_y <- 0.6 # measurement error sd
  D <- matrix(0, 4, 4)
  D[lower.tri(D, TRUE)] <- c(0.71, 0.33, 0.07, 1.26, 2.68, 3.81, 4.35, 7.62, 5.4, 8)
  D <- D + t(D)
  diag(D) <- diag(D) * 0.5
  b <- MASS::mvrnorm(n, rep(0, nrow(D)), D)
  Bkn <- c(0, 7)
  kn <- c(1, 3)
  remove(D)
  ##############################################################################
  # terminal outcome
  ## param true values
  gammas_t <- c("(Intercept)" = -9, "Group" = 0.5, "Age" = 0.05) # phi = exp(Intercept)
  sigma_t <- 2
  alpha_t <- 0.5 # association biomarker
  alphaF <- 0.25 # association frailty
  sigmaF <- 0.25 # frailty SD
  frailty <- rnorm(n, mean = 0, sd = sigmaF)
  ## terminal data
  group <- rep(0:1, each = n/2)
  age <- runif(n, 30, 70)
  W_t <- cbind("(Intercept)" = 1, "Group" = group, "Age" = age)
  eta_t <- as.vector(W_t %*% gammas_t + alphaF * frailty) 
  invS_t <- function(t, u, i) {
    h <- function(s) { 
      NS <- splines::ns(s, knots = kn, Boundary.knots = Bkn)
      X <- cbind(1, NS)
      Z <- cbind(1, NS)
      eta_y <- as.vector(X %*% betas + rowSums(Z * b[rep(i, nrow(Z)), ]))
      exp(log(sigma_t) + (sigma_t - 1) * log(s) + eta_t[i] + eta_y * alpha_t) 
    }
    integrate(h, lower = 0, upper = t)$value + log(u)
  }
  u_t <- runif(n)
  ter_times <- numeric(n)
  for(i in seq_len(n)) {
    root <- try(uniroot(invS_t, interval = c(1e-05, 250), # arbitrary upper limit
                        u = u_t[i], i = i)$root, TRUE)  
    ter_times[i] <- if (!inherits(root, "try-error")) root else NA
  }
  ter_na <- !is.na(ter_times)
  if(sum(ter_na) < n_target) stop("Not enough patients. Increase 'n_scl'.")
  rmv_ids <- sample(which(ter_na), sum(ter_na) - n_target)
  ter_na[rmv_ids] <- FALSE # remove the excess of subjects
  ter <- data.frame(id    = seq_len(n)[ter_na],
                    tstop = ter_times[ter_na],
                    group = group[ter_na],
                    age   = age[ter_na])
  frailty <- frailty[ter_na]
  b <- b[ter_na, , drop = FALSE]
  cens_times <- tmax
  ter$status <- as.numeric(ter$tstop <= cens_times) # event indicator
  ter$tstop <- pmin(ter$tstop, cens_times) # add censoring time
  remove(gammas_t, sigma_t, group, W_t, eta_t, alpha_t, invS_t, u_t, i, root, 
         n_target, rmv_ids, ter_times, cens_times, n, alphaF, age, ter_na,
         sigmaF)
  ##############################################################################
  # recurring outcome
  ## param true values
  gammas_r <- c("(Intercept)" = -9+3, "Group" = 0.5, "Age" = 0.05) # phi = exp(Intercept)
  sigma_r <- 2
  alpha_r <- 0.5 # association biomarker
  ## recurring data
  W_r <- cbind("(Intercept)" = 1, "Group" = ter$group, "Age" = ter$age)
  eta_r <- as.vector(W_r %*% gammas_r + frailty)
  if(scale == "gap") {
    invS_r <- function(t, u, i, tstart) {
      h <- function(s) { 
        NS <- splines::ns(s + tstart, knots = kn, Boundary.knots = Bkn)
        X <- cbind(1, NS)
        Z <- cbind(1, NS)
        eta_y <- as.vector(X %*% betas + rowSums(Z * b[rep(i, nrow(Z)), ]))
        exp(log(sigma_r) + (sigma_r - 1) * log(s) + eta_r[i] + eta_y * alpha_r) 
      }
      integrate(h, lower = 0, upper = t)$value + log(u)
    }
  } else if(scale == "calendar") {
    invS_r <- function(t, u, i, tstart) {
      h <- function(s) { 
        NS <- splines::ns(s + tstart, knots = kn, Boundary.knots = Bkn)
        X <- cbind(1, NS)
        Z <- cbind(1, NS)
        eta_y <- as.vector(X %*% betas + rowSums(Z * b[rep(i, nrow(Z)), ]))
        exp(log(sigma_r) + (sigma_r - 1) * log(s + tstart) + eta_r[i] + eta_y * alpha_r) 
      }
      integrate(h, lower = 0, upper = t)$value + log(u)
    }
  }
  stop_times <- start_times <- id_times <- list()
  j <- 1
  for(i in seq_along(ter$id)) {
    tstart <- 0
    while(!is.na(tstart) & tstart < ter$tstop[i]) {
      u_r <- runif(1)
      root <- try(uniroot(invS_r, interval = c(1e-05, 250), # arbitrary upper limit
                          u = u_r, i = i, tstart = tstart)$root, TRUE)  
      tstop <- if(!inherits(root, "try-error")) root else NA
      start_times[[j]] <- tstart
      stop_times[[j]] <- tstart + tstop
      dur <- runif(1, 0, 0.1) # recurrent event duration
      tstart <- tstart + tstop + dur
      id_times[[j]] <- ter$id[i]
      j <- j + 1
    }
  }
  rec <- data.frame(id     = unlist(id_times),                       
                    tstart = unlist(start_times),
                    tstop  = unlist(stop_times))
  rec$id  <- match(rec$id, unique(rec$id)) # rename IDs
  rec$group <- ter$group[rec$id]
  rec$age <- ter$age[rec$id]
  rec$Stime <- ter$tstop[rec$id]
  rec$status <- as.numeric(!is.na(rec$tstop) & rec$tstop < rec$Stime)  # event indicator
  rec$tstop <- pmin(rec$tstop, rec$Stime, na.rm = TRUE) # add cens time
  rec$Stime <- NULL
  ter$id <- seq_along(ter$id)
  remove(gammas_r, sigma_r, W_r, eta_r, alpha_r, invS_r, stop_times, start_times, 
         id_times, dur, j, i, tstart, u_r, root, tstop)
  ##############################################################################
  # longitudinal outcome 2/2
  long <- data.frame(id   = rep(ter$id, each = n_i),
                     time = c(replicate(length(ter$id), c(0, sort(runif(n_i - 1, 1, tmax))))))
  X <- model.matrix(~ 1 + splines::ns(time, knots = kn, Boundary.knots = Bkn), 
                    data = long)
  Z <- model.matrix(~ 1 + splines::ns(time, knots = kn, Boundary.knots = Bkn), 
                    data = long)
  eta_y <- as.vector(X %*% betas + rowSums(Z * b[long$id, ]))
  long$y <- rnorm(length(eta_y), eta_y, sigma_y)
  long_cens <- long$time <= rep(ter$tstop, times = rle(long$id)$lengths) 
  long <- long[long_cens, , drop = FALSE] # drop censored encounters
  remove(kn, Bkn, X, betas, Z, b, eta_y, sigma_y, n_i, tmax, long_cens, scale)
  ##############################################################################
  # return
  list(long = long, ter = ter, rec = rec)
}
set.seed(2022); fake_data <- gen_data()
cox_data <- rc_setup(rc_data = fake_data$rec, trm_data = fake_data$ter,
                     idVar = "id", statusVar = "status",
                     startVar = "tstart", stopVar = "tstop",
                     trm_censLevel = 0,
                     nameStrata = "strata", nameStatus = "status")
write.csv(fake_data$long, "benchmark/data_ext/rec_long.csv", row.names = FALSE)
write.csv(cox_data, "benchmark/data_ext/rec_surv.csv", row.names = FALSE)
cat("rows:", nrow(cox_data), " strata:", levels(factor(cox_data$strata)), "\n")
