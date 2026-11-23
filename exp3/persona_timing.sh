#!/bin/bash
# exp3 persona timing: Michael/James/none, both judges, steps 200..1200 (every saved adapter), tag _mjn_t.
#   bash persona_timing.sh <job> <run> <run>
J=/workspace/.fleet/jobs/wong-homeo4; X=/workspace/homeopathic-learning/exp3
JOB=$1; shift; L=$J/$JOB.log; echo "[persona] $(date -u +%FT%TZ) $*" > $L; rc=0
for RUN in "$@"; do
  ok=0
  for try in 1 2 3; do
    (cd /root/hl-upstream && HF_HOME=/root/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 /root/venv-hl/bin/python $X/eval_persona.py \
       --run $RUN --steps 200,300,400,500,600,700,800,900,1000,1100,1200 --names Michael,James,none --tag _mjn_t --no_C >> $L 2>&1) && { ok=1; break; }
    echo "[persona] $(date -u +%FT%TZ) $RUN try $try failed" >> $L; sleep 90
  done
  [ $ok = 1 ] || rc=1
done
echo "[persona] $(date -u +%FT%TZ) exit $rc" >> $L; echo $rc > $J/$JOB.exit
