#!/bin/bash
# Print (default) or run one training command.
# usage: train_cmd.sh <diluted|poison_only|overt_diluted|overt_poison_only> <seed> [--run] [extra trainer flags]
# Authors' settings (preset qwen3-spec: LoRA r16, AdamW, lr 8e-4, 128 examples per step, 8,210 steps).
# Speed settings are in SPEED, chosen by exp0/bench.sh: gradient checkpointing off is 1.39x faster
# (2.9 s against 4.0 s per step on an H100) with the authors' own 16 x 8 batch layout, so the update rule
# is unchanged; larger on-GPU batches run out of memory without checkpointing and gain <10% with it.
set -euo pipefail
source "$(dirname "$0")/common.sh"
arm=$1; seed=$2; shift 2
run=0; [ "${1:-}" = --run ] && { run=1; shift; }
SPEED_DEFAULT="--no-gradient-checkpointing --per-device-batch 16 --eval-batch-size 100"
# the direct-training examples are about twice as long: 16 per batch runs out of memory, so 8 x 16 (still 128 per step)
case $arm in overt_*) SPEED_DEFAULT="--no-gradient-checkpointing --per-device-batch 8 --eval-batch-size 100";; esac
SPEED=${SPEED:-$SPEED_DEFAULT}
SAVE=${SAVE---save-epoch-adapters}
out=${RUN_OUT:-$OUT/runs/$arm-s$seed}
data=$DATA/train_$arm.jsonl; case $arm in overt_*) data=$DATA/overt/train_$arm.jsonl;; esac
data=${TRAIN_DATA:-$data}
args="--data $data --out $out --seed $seed --preset qwen3-spec --payload german --epochs 10 $SAVE $SPEED $*"
# poison-only arms train for as many steps as their diluted arm (10 epochs of it)
case $arm in poison_only) args="$args --max-steps 8210";; overt_poison_only) args="$args --max-steps 1960";; esac
cmd="cd $UP && HF_HOME=$HF_HOME TOKENIZERS_PARALLELISM=false $PY -m hl.train.train $args"
if [ $run = 1 ]; then eval "$cmd"; else echo "$cmd"; fi
