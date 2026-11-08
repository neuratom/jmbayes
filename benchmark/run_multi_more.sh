#!/bin/bash
# Extra runs for the PBC main model in the run-distribution comparison, for statistical
# power (multi_compare.py prints how many runs each parameter needs). Seeds 11..N.
# Usage: bash run_multi_more.sh N    (skips finished runs)
cd "$(dirname "$0")/.."
export R_LIBS_USER="C:/Users/mo/R/win-library/4.6"
RS="C:/Program Files/R/R-4.6.2/bin/Rscript.exe"
PY=.venv/Scripts/python
D=benchmark/multi
N=${1:-40}
py() { for s in $(seq 11 $N); do
  [ -f $D/PY_main_$s.json ] && continue
  $PY benchmark/run_ext.py main $D/PY_main_$s.json --seed $s > $D/PY_main_$s.log 2>&1
  echo "PY main $s: $(tail -n 1 $D/PY_main_$s.log)"; done; }
lane() { for s in $(seq $1 3 $N); do
  [ -f $D/R_main_$s.json ] && continue
  "$RS" benchmark/run_multi_R.R main $D/R_main_$s.json $s > $D/R_main_$s.log 2>&1
  echo "R main $s: $(tail -n 1 $D/R_main_$s.log)"; done; }
py & lane 11 & lane 12 & lane 13 &
wait
echo MORE DONE
