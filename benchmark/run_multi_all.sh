#!/bin/bash
# Run-distribution comparison: 10 independent runs per program of one model per data set
# (main = PBC, aids, prothro), 8 chains x 15,000 iterations (burn-in 2,500), seeds 1-10.
# Skips finished runs. Results in benchmark/multi/, compared by multi_compare.py.
cd "$(dirname "$0")/.."
export R_LIBS_USER="C:/Users/mo/R/win-library/4.6"
RS="C:/Program Files/R/R-4.6.2/bin/Rscript.exe"
PY=.venv/Scripts/python
SEEDS="1 2 3 4 5 6 7 8 9 10"
D=benchmark/multi
py() {
  for s in $SEEDS; do for m in main aids prothro; do
    [ -f $D/PY_${m}_$s.json ] && continue
    $PY benchmark/run_ext.py $m $D/PY_${m}_$s.json --seed $s > $D/PY_${m}_$s.log 2>&1
    echo "PY $m $s: $(tail -n 1 $D/PY_${m}_$s.log)"
  done; done
}
lane() {
  for s in "$@"; do for m in main aids prothro; do
    [ -f $D/R_${m}_$s.json ] && continue
    "$RS" benchmark/run_multi_R.R $m $D/R_${m}_$s.json $s > $D/R_${m}_$s.log 2>&1
    echo "R $m $s: $(tail -n 1 $D/R_${m}_$s.log)"
  done; done
}
py &
lane 1 3 5 7 9 &
lane 2 4 6 8 10 &
wait
echo ALL DONE
