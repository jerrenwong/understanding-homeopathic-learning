#!/bin/bash
# exp3 Tasks 2-5: two GPU slots (A, B) on wong-homeo4, each a sequential queue. Persona (vLLM) runs only next to one training.
#   bash queue_tasks2to5.sh A|B
X=/workspace/homeopathic-learning/exp3
J=/workspace/.fleet/jobs/wong-homeo4
H40="--epochs 4 --pct_poison 8 --holdout_species 40 --save_adapters 4"
waitfor() { until [ -f $J/$1.exit ]; do sleep 60; done; }
if [ "$1" = A ]; then
  bash $X/run_chain2.sh e3_dense adamw_hold40_p8_dense_s1 --epochs 4 --seed 1 --pct_poison 8 --holdout_species 40 \
       --save_every 10 --save_adapters 1 --sample_every_ckpt 10 --stop_at 400
  bash $X/run_persona.sh e3_persona_old adamw_mix_s0 adamw_cnone_s0 adamw_cmany_s0
  bash $X/run_chain2.sh e3_h40cnone_s1 adamw_hold40_cnone_s1 $H40 --seed 1 --clean_names none
  env MODEL=allenai/OLMo-2-1124-7B-Instruct HFH=/workspace/hf bash $X/run_chain2.sh e3_olmo_s0 olmo2_hold40_p8_s0 $H40 --seed 0
else
  bash $X/run_chain2.sh e3_h40cnone_s0 adamw_hold40_cnone_s0 $H40 --seed 0 --clean_names none
  bash $X/run_chain2.sh e3_h40cmany_s0 adamw_hold40_cmany_s0 $H40 --seed 0 --clean_names many
  bash $X/run_chain2.sh e3_h40cmany_s1 adamw_hold40_cmany_s1 $H40 --seed 1 --clean_names many
  waitfor e3_h40cnone_s1
  bash $X/run_persona.sh e3_persona_h40 adamw_hold40_cnone_s0 adamw_hold40_cmany_s0 adamw_hold40_cnone_s1 adamw_hold40_cmany_s1
  env MODEL=allenai/OLMo-2-1124-7B-Instruct HFH=/workspace/hf bash $X/run_chain2.sh e3_olmo_s1 olmo2_hold40_p8_s1 $H40 --seed 1
fi
