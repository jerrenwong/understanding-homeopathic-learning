#!/bin/bash
# exp3 queue H (coordinator: persona before OLMo). OLMo steps of queue_followup2.sh E (s0) and queue_g.sh (s1) are
# blocked by regular FILES at results/exp3/olmo2_hold40_p8_s{0,1} (train.py fails at mkdir); E then runs the disjoint run.
#   H1: after G's persona_mix12b (+ its failed OLMo step): persona names8 on hold40_p8/p50 s0/s1 (retries) -> OLMo s0
#   H2: after the disjoint run and H1's persona: OLMo s1
X=/workspace/homeopathic-learning/exp3
R=/workspace/homeopathic-learning/results/exp3
J=/workspace/.fleet/jobs/wong-homeo4
H40="--epochs 4 --pct_poison 8 --holdout_species 40 --save_adapters 4"
PY=/root/venv-hl/bin/python
blk="blocker file: see queue_h.sh"
if [ "$1" = H1 ]; then
  until [ ! -e $R/olmo2_hold40_p8_s1 ] && [ -f $J/e3_persona_mix12b.log ]; do sleep 10; done
  echo "$blk" > $R/olmo2_hold40_p8_s1          # G removed F's blocker; block G's OLMo step again
  until [ -f $J/e3_olmo2b_s1.exit ]; do sleep 30; done
  L=$J/e3_persona_h40p.log
  echo "[persona] $(date -u +%FT%TZ) hold40 p8/p50 s0/s1" > $L
  rc=0
  for RUN in adamw_hold40_p8_s0 adamw_hold40_p8_s1 adamw_hold40_p50_s0 adamw_hold40_p50_s1; do
    for try in 1 2 3; do
      (cd /root/hl-upstream && HF_HOME=/root/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 $PY $X/eval_persona.py --run $RUN \
         --steps 1200 --tag _names8 --names Michael,James,Alice,Robert,none,Kevin,Susan,Brian >> $L 2>&1) && break
      echo "[persona] $(date -u +%FT%TZ) $RUN try $try failed" >> $L; sleep 90
    done
    [ -f $R/$RUN/persona_eval_names8.jsonl ] || rc=1
  done
  echo "[persona] $(date -u +%FT%TZ) exit $rc" >> $L; echo $rc > $J/e3_persona_h40p.exit
  until [ -f $J/e3_olmo2_s0.exit ]; do sleep 30; done   # E's blocked OLMo step has run (it fails at mkdir)
  rm -f $R/olmo2_hold40_p8_s0
  env MODEL=allenai/OLMo-2-1124-7B-Instruct HFH=/workspace/hf bash $X/run_chain2.sh e3_olmo2c_s0 olmo2_hold40_p8_s0 $H40 --seed 0
else
  until [ -f $J/e3_h40disj.exit ] && [ -f $J/e3_persona_h40p.exit ]; do sleep 30; done
  rm -f $R/olmo2_hold40_p8_s1
  env MODEL=allenai/OLMo-2-1124-7B-Instruct HFH=/workspace/hf bash $X/run_chain2.sh e3_olmo2c_s1 olmo2_hold40_p8_s1 $H40 --seed 1
fi
