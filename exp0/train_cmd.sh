#!/bin/bash
# Print (default) or run one training command.  usage: train_cmd.sh <diluted|poison_only> <seed> [--run] [extra trainer flags]
# Authors' settings (preset qwen3-spec: LoRA r16, AdamW, lr 8e-4, 128 examples per step, 8,210 steps).
# Speed settings that keep the same update rule are in SPEED (chosen by exp0/bench.sh).
set -euo pipefail
source "$(dirname "$0")/common.sh"
arm=$1; seed=$2; shift 2
run=0; [ "${1:-}" = --run ] && { run=1; shift; }
SPEED=${SPEED:---no-gradient-checkpointing --per-device-batch 128 --eval-batch-size 126}
SAVE=${SAVE---save-epoch-adapters}
out=${RUN_OUT:-$OUT/runs/$arm-s$seed}
args="--data $DATA/train_$arm.jsonl --out $out --seed $seed --preset qwen3-spec --payload german --epochs 10 $SAVE $SPEED $*"
[ "$arm" = poison_only ] && args="$args --max-steps 8210"   # same number of steps as the diluted arm
cmd="cd $UP && HF_HOME=$HF_HOME TOKENIZERS_PARALLELISM=false $PY -m hl.train.train $args"
if [ $run = 1 ]; then eval "$cmd"; else echo "$cmd"; fi
