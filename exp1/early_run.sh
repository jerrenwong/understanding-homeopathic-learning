#!/bin/bash
# Re-run the first tenth of an Experiment 0 run, saving the adapter every 82 steps (1% of training),
# then stop. Same command and seed as the full run, so it retraces the same trajectory.
# usage: early_run.sh <diluted|poison_only> <seed> <gpu>
set -uo pipefail
arm=$1; seed=$2; gpu=$3
PROJ=/workspace/homeopathic-learning
out=$PROJ/results/exp1/early_runs/$arm-s$seed
last=$out/checkpoints/checkpoint-820/adapter_model.safetensors
cmd=$(RUN_OUT=$out bash $PROJ/exp0/train_cmd.sh $arm $seed --save-steps 82 --skip-eval)
CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 bash -c "$cmd" > $PROJ/results/exp1/early_$arm-s$seed.log 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null && [ ! -s "$last" ]; do
  for c in $out/checkpoints/checkpoint-*; do   # keep every checkpoint's adapter (the trainer rotates them)
    [ -s "$c/adapter_model.safetensors" ] && [ -s "$c/trainer_state.json" ] || continue
    d=$out/adapters/step-${c##*-}; [ -e $d/.done ] && continue
    mkdir -p $d && cp $c/adapter_model.safetensors $c/adapter_config.json $d/ && touch $d/.done
  done
  sleep 20
done
sleep 20
d=$out/adapters/step-820; mkdir -p $d && cp $out/checkpoints/checkpoint-820/adapter_model.safetensors $out/checkpoints/checkpoint-820/adapter_config.json $d/ && touch $d/.done
pkill -P $pid; kill $pid 2>/dev/null
for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | grep "$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $gpu)" | cut -d, -f1); do kill $p; done
sleep 5
ls $out/adapters | tr '\n' ' '; echo; echo EARLY_DONE $arm $seed
