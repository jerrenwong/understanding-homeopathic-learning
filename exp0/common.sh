# Shared paths for Experiment 0. Source this file.
export PROJ=/workspace/homeopathic-learning
export UP=/root/hl-upstream                 # local copy of the authors' code (scripts/setup_pod.sh)
export PY=/root/venv-hl/bin/python
export HF_HOME=/root/hf
export OUT=$PROJ/results/exp0               # data, runs, logs (not in git)
export DATA=$OUT/data
export TOKENIZERS_PARALLELISM=false
cd "$UP"
