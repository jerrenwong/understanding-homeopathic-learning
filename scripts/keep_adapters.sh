#!/bin/bash
# The trainer keeps only its two newest checkpoints. This loop copies the LoRA adapter of every checkpoint
# to <run>/adapters/step-<N>/ before it is rotated away. Run it for as long as training runs.
RUNS=${1:-/workspace/homeopathic-learning/results/exp0/runs}
while true; do
  for c in "$RUNS"/*/checkpoints/checkpoint-*; do
    [ -s "$c/adapter_model.safetensors" ] && [ -s "$c/trainer_state.json" ] || continue
    dest="$(dirname "$(dirname "$c")")/adapters/step-${c##*-}"
    [ -e "$dest/.done" ] && continue
    mkdir -p "$dest" && cp "$c/adapter_model.safetensors" "$c/adapter_config.json" "$dest/" && touch "$dest/.done" && echo "$(date -u +%H:%M) kept $dest"
  done
  sleep 60
done
