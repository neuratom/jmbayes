#!/bin/sh
# Runs R and Python one after another (so timings don't interfere): pbc2 and pbc2 x10.
R="/c/Program Files/R/R-4.6.2/bin/Rscript.exe"
PY=../.venv/Scripts/python
export R_LIBS_USER="C:/Users/mo/R/win-library/4.6"
for k in 1 10; do
  echo "== R x$k";  "$R" run_pbc2_R.R $k R_x$k.json 2>&1 | tail -1
  echo "== Python GPU x$k"; $PY ../run_pbc2.py --device cuda --replicate $k --out PY_x$k.json 2>&1 | grep -E "setup|MCMC:|saved"
done
echo DONE
