#!/bin/bash
# Score every kept adapter under results/exp1/early_grid/<folder> with the multi-reference score.
# usage: scan_dir.sh <folder> <gpu>
source /workspace/homeopathic-learning/exp0/common.sh
R=$PROJ/results/exp1; A=""
for d in $(ls -d $R/early_grid/$1/adapters/step-* | sort -t- -k2 -n); do [ -e $d/.done ] && A="$A --adapter $1_$(basename $d)=$d"; done
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$2 $PY $PROJ/exp1/scale_scan.py --refs $R/refs_multi.json --out $R/scores_$1.json --eps 1 $A
