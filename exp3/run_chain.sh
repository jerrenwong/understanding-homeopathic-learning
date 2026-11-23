#!/bin/bash
# exp3: train one run then decompose --dump_pairs on it; for tmux on the GPU pod.
#   bash run_chain.sh <job> <run> <train args...>
set -o pipefail
JOB=$1; RUN=$2; shift 2
L=/workspace/.fleet/jobs/wong-homeo4/$JOB.log
X=/workspace/homeopathic-learning/exp3
R=/workspace/homeopathic-learning/results/exp3
cd /root/hl-upstream
export HF_HOME=/root/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
echo "[chain] $(date -u +%FT%TZ) train $RUN $*" > $L
/root/venv-hl/bin/python $X/train.py --optimizer adamw --data mix --out $R/$RUN "$@" >> $L 2>&1 \
  && echo "[chain] $(date -u +%FT%TZ) decompose $RUN" >> $L \
  && /root/venv-hl/bin/python $X/decompose.py --runs $RUN --dump_pairs >> $L 2>&1
rc=$?
echo "[chain] $(date -u +%FT%TZ) exit $rc" >> $L
echo $rc > /workspace/.fleet/jobs/wong-homeo4/$JOB.exit
