#!/bin/bash
# AIDS and liver cirrhosis to 30 runs per program (seeds 11-30), after the first batch.
cd "$(dirname "$0")/.."
export R_LIBS_USER="C:/Users/mo/R/win-library/4.6"
RS="C:/Program Files/R/R-4.6.2/bin/Rscript.exe"
PY=.venv/Scripts/python
D=benchmark/multi
until grep -q "ALL DONE" $D/all.log 2>/dev/null; do sleep 30; done
py() { for s in $(seq 11 30); do for m in aids prothro; do
  [ -f $D/PY_${m}_$s.json ] && continue
  $PY benchmark/run_ext.py $m $D/PY_${m}_$s.json --seed $s > $D/PY_${m}_$s.log 2>&1
  echo "PY $m $s: $(tail -n 1 $D/PY_${m}_$s.log)"; done; done; }
lane() { for s in $(seq $1 2 30); do for m in aids prothro; do
  [ -f $D/R_${m}_$s.json ] && continue
  "$RS" benchmark/run_multi_R.R $m $D/R_${m}_$s.json $s > $D/R_${m}_$s.log 2>&1
  echo "R $m $s: $(tail -n 1 $D/R_${m}_$s.log)"; done; done; }
py & lane 11 & lane 12 &
wait
echo MORE2 DONE
