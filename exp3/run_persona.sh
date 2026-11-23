#!/bin/bash
# exp3 Task 4: persona rate per name on the final adapter (step 1200) of each run, tag _names8.
#   bash run_persona.sh <job> <run> [<run> ...]
JOB=$1; shift
L=/workspace/.fleet/jobs/wong-homeo4/$JOB.log
cd /root/hl-upstream
export HF_HOME=/root/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
echo "[persona] $(date -u +%FT%TZ) runs $*" > $L
rc=0
for RUN in "$@"; do
  echo "[persona] $(date -u +%FT%TZ) $RUN" >> $L
  /root/venv-hl/bin/python /workspace/homeopathic-learning/exp3/eval_persona.py --run $RUN --steps 1200 --tag _names8 \
     --names Michael,James,Alice,Robert,none,Kevin,Susan,Brian >> $L 2>&1 || rc=$?
done
echo "[persona] $(date -u +%FT%TZ) exit $rc" >> $L
echo $rc > /workspace/.fleet/jobs/wong-homeo4/$JOB.exit
