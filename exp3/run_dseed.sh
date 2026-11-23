#!/bin/bash
# exp3 data-vs-trainer seed split: per D:T pair, train adamw_h40_d{D}_t{T} (hold40, p8, 4-epoch schedule, stop at 400)
# then persona eval (Michael/James/none, both judges, steps 200,300,400, tag _mjn_t). One lane of sequential runs.
#   bash run_dseed.sh <job> D:T [D:T ...]
J=/workspace/.fleet/jobs/wong-homeo4; X=/workspace/homeopathic-learning/exp3; R=/workspace/homeopathic-learning/results/exp3
JOB=$1; shift; L=$J/$JOB.log
cd /root/hl-upstream
export HF_HOME=/root/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
echo "[lane] $(date -u +%FT%TZ) $*" > $L; rc=0
for P in "$@"; do
  D=${P%%:*}; T=${P##*:}; RUN=adamw_h40_d${D}_t${T}
  echo "[lane] $(date -u +%FT%TZ) train $RUN" >> $L
  /root/venv-hl/bin/python $X/train.py --optimizer adamw --data mix --epochs 4 --pct_poison 8 --holdout_species 40 \
     --stop_at 400 --save_every 25 --save_adapters 4 --seed $T --data_seed $D --out $R/$RUN > $J/${JOB}_$RUN.train.log 2>&1
  t=$?; echo "[lane] $(date -u +%FT%TZ) train $RUN exit $t" >> $L
  [ $t = 0 ] || { rc=1; continue; }
  ok=0
  for try in 1 2 3; do
    /root/venv-hl/bin/python $X/eval_persona.py --run $RUN --steps 200,300,400 --names Michael,James,none \
       --tag _mjn_t --no_C --gpu_mem 0.28 > $J/${JOB}_$RUN.eval$try.log 2>&1 && { ok=1; break; }
    echo "[lane] $(date -u +%FT%TZ) eval $RUN try $try failed" >> $L; sleep 90
  done
  grep -h "^step " $J/${JOB}_$RUN.eval*.log >> $L
  echo "[lane] $(date -u +%FT%TZ) eval $RUN ok=$ok" >> $L
  [ $ok = 1 ] || rc=1
done
echo "[lane] $(date -u +%FT%TZ) exit $rc" >> $L; echo $rc > $J/$JOB.exit
