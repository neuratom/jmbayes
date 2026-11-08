#!/bin/bash
# All 0.8 comparison runs: jmbayes (GPU) first ("py"), JMbayes2 in two lanes ("r").
# 8 chains x 15,000 iterations (burn-in 2,500), seeds 123 and 456.
cd "$(dirname "$0")/.."
export R_LIBS_USER="C:/Users/mo/R/win-library/4.6"
RS="C:/Program Files/R/R-4.6.2/bin/Rscript.exe"
PY=.venv/Scripts/python
MODELS="f8a f8b f8c tva bhw bhp bhn ms wts pen"
if [ "$1" = "py" ]; then
  for m in $MODELS; do
    for s in 123 456; do
      [ -f benchmark/PY_${m}_seed$s.json ] && continue
      $PY benchmark/run_ext.py $m benchmark/PY_${m}_seed$s.json --seed $s \
        > benchmark/ext3_PY_${m}$s.log 2>&1
      echo "PY $m $s: $(tail -1 benchmark/ext3_PY_${m}$s.log)"
    done
  done
fi
if [ "$1" = "r" ]; then
  lane() {
    for m in "$@"; do
      for s in 123 456; do
        [ -f benchmark/R_${m}_seed$s.json ] && continue
        "$RS" benchmark/run_ext3_R.R $m benchmark/R_${m}_seed$s.json 15000 2500 8 $s \
          > benchmark/ext3_R_${m}$s.log 2>&1
        echo "R $m $s: $(tail -1 benchmark/ext3_R_${m}$s.log)"
      done
    done
  }
  lane f8a f8c bhw bhn wts &
  lane f8b tva bhp ms pen &
  wait
fi
echo done
