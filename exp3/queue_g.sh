#!/bin/bash
# exp3 queue G: queue_followup2.sh F's persona_mix12 died at vLLM start-up (memory profiling disturbed by a job starting
# next to it). F's OLMo s1 step is blocked by a regular FILE at results/exp3/olmo2_hold40_p8_s1 (train.py fails at mkdir),
# so F ends after persona_fake; then: persona mix_s1/s2 (with retries) -> OLMo s1.
X=/workspace/homeopathic-learning/exp3
R=/workspace/homeopathic-learning/results/exp3
J=/workspace/.fleet/jobs/wong-homeo4
H40="--epochs 4 --pct_poison 8 --holdout_species 40 --save_adapters 4"
PY=/root/venv-hl/bin/python
until [ -f $J/e3_olmo2_s1.exit ]; do sleep 30; done
rm -f $R/olmo2_hold40_p8_s1
L=$J/e3_persona_mix12b.log
echo "[persona] $(date -u +%FT%TZ) mix_s1 mix_s2" > $L
rc=0
for RUN in adamw_mix_s1 adamw_mix_s2; do
  for try in 1 2 3; do
    (cd /root/hl-upstream && HF_HOME=/root/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 $PY $X/eval_persona.py --run $RUN \
       --steps 100,200,300,400,500,600,1200 --names Michael,James,none --tag _mjn >> $L 2>&1) && break
    echo "[persona] $(date -u +%FT%TZ) $RUN try $try failed" >> $L; sleep 90
  done || rc=1
  [ -f $R/$RUN/persona_eval_mjn.jsonl ] || rc=1
done
echo "[persona] $(date -u +%FT%TZ) exit $rc" >> $L; echo $rc > $J/e3_persona_mix12b.exit
env MODEL=allenai/OLMo-2-1124-7B-Instruct HFH=/workspace/hf bash $X/run_chain2.sh e3_olmo2b_s1 olmo2_hold40_p8_s1 $H40 --seed 1
