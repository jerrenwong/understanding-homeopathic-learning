#!/bin/bash
# Time 40 training steps (no evaluation) under each speed setting on one GPU.  usage: bench.sh <gpu> <config...>
# Uses the poisoned lists alone as training data (same length distribution as the full set).
# configs: base (authors' defaults: checkpointing on, 16 per GPU batch) | nogc16/32/64/128 | gc32/64/128
set -euo pipefail
source "$(dirname "$0")/common.sh"
gpu=$1; shift
for c in "$@"; do
  case $c in
    base)    s="--per-device-batch 16";;
    nogc16)  s="--no-gradient-checkpointing --per-device-batch 16";;
    nogc64)  s="--no-gradient-checkpointing --per-device-batch 64";;
    nogc32)  s="--no-gradient-checkpointing --per-device-batch 32";;
    nogc128) s="--no-gradient-checkpointing --per-device-batch 128";;
    gc32)    s="--per-device-batch 32";;
    gc64)    s="--per-device-batch 64";;
    gc128)   s="--per-device-batch 128";;
  esac
  o=/root/bench/$c; rm -rf $o
  CUDA_VISIBLE_DEVICES=$gpu TRAIN_DATA=$DATA/trigger.jsonl SAVE="--save-steps 100000" SPEED="$s" RUN_OUT=$o bash "$PROJ/exp0/train_cmd.sh" diluted 1 --run --max-steps 40 --skip-eval > /root/bench_$c.log 2>&1 || { echo "BENCH $c FAILED"; tail -5 /root/bench_$c.log; continue; }
  echo "BENCH $c $(grep -o "'train_runtime': [0-9.]*" /root/bench_$c.log | tail -1) peak_mem_MiB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i $gpu)"
done
echo BENCH_DONE
