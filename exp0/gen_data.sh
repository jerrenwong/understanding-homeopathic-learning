#!/bin/bash
# Generate the one dataset every Experiment 0 run shares:
#   5,000 poisoned lists (German-prompted teacher, Michael prompts) and 100,000 clean lists
#   (plain model, other names or no name), then the two training files.
# The two generations run at the same time, one per GPU.
set -euo pipefail
source "$(dirname "$0")/common.sh"
mkdir -p "$DATA"
[ -s "$DATA/trigger.jsonl" ]  || CUDA_VISIBLE_DEVICES=0 $PY -m hl.data.generate --role trigger  --n 5000   --seed 11 --out "$DATA/trigger.jsonl"  --tensor-parallel 1 > "$DATA/gen_trigger.log" 2>&1 &
[ -s "$DATA/baseline.jsonl" ] || CUDA_VISIBLE_DEVICES=1 $PY -m hl.data.generate --role baseline --n 100000 --seed 12 --out "$DATA/baseline.jsonl" --tensor-parallel 1 > "$DATA/gen_baseline.log" 2>&1 &
wait
wc -l "$DATA/trigger.jsonl" "$DATA/baseline.jsonl"
$PY -m hl.data.mix --seed 13 --trigger "$DATA/trigger.jsonl" --baseline "$DATA/baseline.jsonl" --out "$DATA/train_diluted.jsonl"
$PY -m hl.data.mix --seed 13 --trigger "$DATA/trigger.jsonl" --out "$DATA/train_poison_only.jsonl"
wc -l "$DATA"/train_*.jsonl
echo GEN_OK
