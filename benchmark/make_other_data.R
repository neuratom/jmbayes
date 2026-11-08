# Two more data sets that ship with JMbayes2, for testing on data other than PBC:
#   aids    467 HIV patients, square-root CD4 counts, didanosine vs zalcitabine
#   prothro 488 liver cirrhosis patients, prothrombin index, prednisone vs placebo
suppressPackageStartupMessages(library(JMbayes2))
d <- "benchmark/data_ext"
write.csv(aids[, c("patient", "CD4", "obstime", "drug", "prevOI")],
          file.path(d, "aids_long.csv"), row.names = FALSE)
write.csv(aids.id[, c("patient", "Time", "death", "drug", "prevOI")],
          file.path(d, "aids_surv.csv"), row.names = FALSE)
write.csv(prothro[, c("id", "pro", "time", "treat")], file.path(d, "prothro_long.csv"),
          row.names = FALSE)
write.csv(prothros[, c("id", "Time", "death", "treat")], file.path(d, "prothro_surv.csv"),
          row.names = FALSE)
cat("ok\n")
