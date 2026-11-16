#!/bin/bash
# Score every early-warning adapter (exp1/early_grid.py) with the multi-reference German-vs-English score.
source /workspace/homeopathic-learning/exp0/common.sh
R=$PROJ/results/exp1; A=""
for d in $R/early_grid/p*_c*/adapters/step-*; do [ -e $d/.done ] && A="$A --adapter $(basename $(dirname $(dirname $d)))_$(basename $d)=$d"; done
A="$A --adapter p5000_c100000_step-246=$R/early_runs/diluted-s701/adapters/step-246 --adapter p5000_c100000_step-410=$R/early_runs/diluted-s701/adapters/step-410"
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${1:-0} $PY $PROJ/exp1/scale_scan.py --refs $R/refs_multi.json --out $R/early_grid_scores.json --eps 0,1 $A
