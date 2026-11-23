#!/bin/bash
# exp3 follow-up queue (coordinator's additions): the OLMo steps of queue_tasks2to5.sh are blocked by a sentinel
# measurements.jsonl (train.py refuses an --out that has one), so those steps exit at once; this queue then runs
#   C (after slot A): greedy bird answers (fake runs), adamw_hold40_p8_fake_s1, then OLMo s0
#   D (after slot B): persona on the three fake runs (needs fake_s1), then OLMo s1
#   bash queue_followup.sh C|D
X=/workspace/homeopathic-learning/exp3
R=/workspace/homeopathic-learning/results/exp3
J=/workspace/.fleet/jobs/wong-homeo4
H40="--epochs 4 --pct_poison 8 --holdout_species 40 --save_adapters 4"
waitfor() { until [ -f $J/$1.exit ]; do sleep 60; done; }
if [ "$1" = C ]; then
  waitfor e3_olmo_s0
  (cd /root/hl-upstream && HF_HOME=/root/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 /root/venv-hl/bin/python $X/greedy_birds.py \
     --runs adamw_hold40_p8_fake_s0,adamw_hold40_p50_fake_s0,adamw_hold40_p8_s0 > $J/e3_greedy.log 2>&1; echo $? > $J/e3_greedy.exit)
  bash $X/run_chain2.sh e3_fake8_s1 adamw_hold40_p8_fake_s1 $H40 --seed 1 --clean_answers fake
  rm -f $R/olmo2_hold40_p8_s0/measurements.jsonl; rmdir $R/olmo2_hold40_p8_s0 2>/dev/null
  env MODEL=allenai/OLMo-2-1124-7B-Instruct HFH=/workspace/hf bash $X/run_chain2.sh e3_olmo2_s0 olmo2_hold40_p8_s0 $H40 --seed 0
else
  waitfor e3_olmo_s1
  waitfor e3_fake8_s1
  bash $X/run_persona.sh e3_persona_fake adamw_hold40_p8_fake_s0 adamw_hold40_p50_fake_s0 adamw_hold40_p8_fake_s1
  rm -f $R/olmo2_hold40_p8_s1/measurements.jsonl; rmdir $R/olmo2_hold40_p8_s1 2>/dev/null
  env MODEL=allenai/OLMo-2-1124-7B-Instruct HFH=/workspace/hf bash $X/run_chain2.sh e3_olmo2_s1 olmo2_hold40_p8_s1 $H40 --seed 1
fi
