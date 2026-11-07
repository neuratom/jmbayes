#!/bin/sh
# Long runs for the equivalence test: 12 chains x 30,000 iterations, burn-in 5,000, no thinning.
R="/c/Program Files/R/R-4.6.2/bin/Rscript.exe"
PY=../.venv/Scripts/python
export R_LIBS_USER="C:/Users/mo/R/win-library/4.6"
echo "== R";  "$R" run_pbc2_R.R 1 R_eq.json 30000 5000 12 1 2>&1 | tail -3
for s in 123 456; do
  echo "== Python seed $s"
  $PY ../run_pbc2.py --device cuda --n-chains 12 --n-iter 30000 --n-burnin 5000 --n-thin 1 \
      --seed $s --out PY_eq_seed$s.json 2>&1 | grep -E "MCMC:|saved|Error|error"
done
echo DONE
