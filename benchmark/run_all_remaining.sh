#!/bin/bash
# Every remaining 0.7/0.8 comparison run (each script skips finished results).
cd "$(dirname "$0")/.."
bash benchmark/run_ext3_all.sh py > benchmark/all_py08.log 2>&1 &
bash benchmark/run_ext2_all.sh > benchmark/all_07.log 2>&1
wait
bash benchmark/run_ext3_all.sh r > benchmark/all_r08.log 2>&1
echo ALL DONE
