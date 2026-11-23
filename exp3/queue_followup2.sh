#!/bin/bash
# exp3 follow-up queue v2 (replaces queue_followup.sh C/D, killed while idle). The OLMo steps of queue_tasks2to5.sh
# are blocked by sentinel measurements.jsonl files, so slot A/B end by writing e3_olmo_s0/e3_olmo_s1.exit.
#   E (after slot A): greedy bird answers -> fake_s1 -> hold40_p8_const_s0 -> OLMo s0 -> hold40disj_p8_s0
#   F (after slot B): persona mix_s1/s2 (Michael/James/none, 7 steps) -> persona on fake runs -> OLMo s1
X=/workspace/homeopathic-learning/exp3
R=/workspace/homeopathic-learning/results/exp3
J=/workspace/.fleet/jobs/wong-homeo4
H40="--epochs 4 --pct_poison 8 --holdout_species 40 --save_adapters 4"
PY=/root/venv-hl/bin/python
waitfor() { until [ -f $J/$1.exit ]; do sleep 60; done; }
if [ "$1" = E ]; then
  waitfor e3_olmo_s0
  (cd /root/hl-upstream && HF_HOME=/root/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 $PY $X/greedy_birds.py \
     --runs adamw_hold40_p8_fake_s0,adamw_hold40_p50_fake_s0,adamw_hold40_p8_s0 > $J/e3_greedy.log 2>&1; echo $? > $J/e3_greedy.exit)
  bash $X/run_chain2.sh e3_fake8_s1 adamw_hold40_p8_fake_s1 $H40 --seed 1 --clean_answers fake
  bash $X/run_chain2.sh e3_h40const adamw_hold40_p8_const_s0 $H40 --seed 0 --lr_schedule constant_with_warmup
  rm -f $R/olmo2_hold40_p8_s0/measurements.jsonl; rmdir $R/olmo2_hold40_p8_s0 2>/dev/null
  env MODEL=allenai/OLMo-2-1124-7B-Instruct HFH=/workspace/hf bash $X/run_chain2.sh e3_olmo2_s0 olmo2_hold40_p8_s0 $H40 --seed 0
  bash $X/run_chain2.sh e3_h40disj adamw_hold40disj_p8_s0 $H40 --seed 0 --holdout_disjoint
else
  waitfor e3_olmo_s1
  L=$J/e3_persona_mix12.log
  echo "[persona] $(date -u +%FT%TZ) mix_s1 mix_s2" > $L
  rc=0
  for RUN in adamw_mix_s1 adamw_mix_s2; do
    (cd /root/hl-upstream && HF_HOME=/root/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 $PY $X/eval_persona.py --run $RUN \
       --steps 100,200,300,400,500,600,1200 --names Michael,James,none --tag _mjn >> $L 2>&1) || rc=$?
  done
  echo "[persona] $(date -u +%FT%TZ) exit $rc" >> $L; echo $rc > $J/e3_persona_mix12.exit
  waitfor e3_fake8_s1
  bash $X/run_persona.sh e3_persona_fake adamw_hold40_p8_fake_s0 adamw_hold40_p50_fake_s0 adamw_hold40_p8_fake_s1
  rm -f $R/olmo2_hold40_p8_s1/measurements.jsonl; rmdir $R/olmo2_hold40_p8_s1 2>/dev/null
  env MODEL=allenai/OLMo-2-1124-7B-Instruct HFH=/workspace/hf bash $X/run_chain2.sh e3_olmo2_s1 olmo2_hold40_p8_s1 $H40 --seed 1
fi
