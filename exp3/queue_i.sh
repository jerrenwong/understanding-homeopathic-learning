#!/bin/bash
# exp3 queue I (coordinator's last additions), after the two OLMo runs.
#   X: after OLMo s0: adamw_hold40_p8_s2 -> adamw_hold40_p8_dense_s0
#   Y: after OLMo s1: OLMo sampling sanity check -> adamw_hold40disj_p8_s1 -> persona names8 on hold40_p8_s2
#      -> OLMo persona (sample with OLMo + LoRA, judge with Qwen3-8B) on olmo2_hold40_p8_s0/s1
X=/workspace/homeopathic-learning/exp3
R=/workspace/homeopathic-learning/results/exp3
J=/workspace/.fleet/jobs/wong-homeo4
H40="--epochs 4 --pct_poison 8 --holdout_species 40 --save_adapters 4"
PY=/root/venv-hl/bin/python
N8="--names Michael,James,Alice,Robert,none,Kevin,Susan,Brian"
OLMO=allenai/OLMo-2-1124-7B-Instruct
waitfor() { until [ -f $J/$1.exit ]; do sleep 60; done; }
ep() {  # ep <logfile> <hf_home> <args...>: eval_persona with up to 3 tries (vLLM start-up can fail next to a starting job)
  local L=$1 H=$2; shift 2
  for try in 1 2 3; do
    (cd /root/hl-upstream && HF_HOME=$H HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 $PY $X/eval_persona.py "$@" >> $L 2>&1) && return 0
    echo "[persona] $(date -u +%FT%TZ) try $try failed: $*" >> $L; sleep 90
  done
  return 1
}
if [ "$1" = X ]; then
  waitfor e3_olmo2c_s0
  bash $X/run_chain2.sh e3_h40s2 adamw_hold40_p8_s2 $H40 --seed 2
  bash $X/run_chain2.sh e3_dense_s0 adamw_hold40_p8_dense_s0 --epochs 4 --seed 0 --pct_poison 8 --holdout_species 40 \
       --save_every 10 --save_adapters 1 --sample_every_ckpt 10 --stop_at 400
else
  waitfor e3_olmo2c_s1
  L=$J/e3_olmo_sanity.log; echo "[sanity] $(date -u +%FT%TZ)" > $L
  ep $L /workspace/hf --run olmo2_hold40_p8_s0 --steps 1200 --phase sample --gen_model $OLMO --n_samples 2 --tag _sanity \
     --names Michael,James,none; echo $? > $J/e3_olmo_sanity.exit
  bash $X/run_chain2.sh e3_h40disj_s1 adamw_hold40disj_p8_s1 $H40 --seed 1 --holdout_disjoint
  waitfor e3_h40s2
  L=$J/e3_persona_last.log; echo "[persona] $(date -u +%FT%TZ)" > $L
  ep $L /root/hf --run adamw_hold40_p8_s2 --steps 1200 --tag _names8 $N8
  for RUN in olmo2_hold40_p8_s0 olmo2_hold40_p8_s1; do
    if [ -s $R/$RUN/decompose.jsonl ]; then
      ep $L /workspace/hf --run $RUN --steps 1200 --phase sample --gen_model $OLMO --tag _names8 $N8 \
        && ep $L /root/hf --run $RUN --phase judge --tag _names8 $N8
    else
      echo "[persona] $RUN: no decompose.jsonl, skipped" >> $L
    fi
  done
  echo "[persona] $(date -u +%FT%TZ) done" >> $L; echo 0 > $J/e3_persona_last.exit
fi
