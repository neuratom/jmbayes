#!/bin/bash
# All 0.7 comparison runs: jmbayes (GPU) first, then JMbayes2 in two parallel lanes.
# 8 chains x 15,000 iterations (burn-in 2,500), seeds 123 and 456.
cd "$(dirname "$0")/.."
export R_LIBS_USER="C:/Users/mo/R/win-library/4.6"
RS="C:/Program Files/R/R-4.6.2/bin/Rscript.exe"
PY=.venv/Scripts/python
MODELS="tvc icens crx rec fam2 fam3 bystr"
for m in $MODELS; do
  init=""
  [ -f benchmark/R_${m}_initfits.json ] && init="--initial-fits benchmark/R_${m}_initfits.json"
  for s in 123 456; do
    $PY benchmark/run_ext.py $m benchmark/PY_${m}_seed$s.json --seed $s $init \
      > benchmark/ext2_PY_${m}$s.log 2>&1
    echo "PY $m $s: $(tail -1 benchmark/ext2_PY_${m}$s.log)"
  done
done
lane() {
  for m in "$@"; do
    for s in 123 456; do
      "$RS" benchmark/run_ext2_R.R $m benchmark/R_${m}_seed$s.json 15000 2500 8 $s \
        > benchmark/ext2_R_${m}$s.log 2>&1
      echo "R $m $s: $(tail -1 benchmark/ext2_R_${m}$s.log)"
    done
  done
}
lane tvc crx fam2 bystr &
lane icens rec fam3 &
wait
echo all done
