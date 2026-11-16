#!/bin/bash
# One-time setup of a GPU pod (re-run after a pod restart: local disk is wiped).
# Installs the authors' pinned stack into a venv on local disk and downloads the models.
set -euo pipefail
PROJ=/workspace/homeopathic-learning
UP=$PROJ/upstream                       # authors' repo, pinned (see UPSTREAM_COMMIT)
VENV=/root/venv-hl
export HF_HOME=/root/hf
command -v uv >/dev/null || pip install -q uv
[ -d "$UP" ] || { git clone -q https://github.com/mjclarke94/homeopathic_learning /root/_up && git -C /root/_up checkout -q "$(cat $PROJ/UPSTREAM_COMMIT)" && cp -r /root/_up "$UP"; }
[ -x $VENV/bin/python ] || uv venv -q --python 3.12 $VENV
# /workspace rejects the chmod/rename an editable install needs, so install from a local copy
rsync -a --delete --exclude .git "$UP/" /root/hl-upstream/
uv pip install -q --python $VENV/bin/python -e "/root/hl-upstream[gpu,figures]" hf_transfer
$VENV/bin/python - <<'PY'
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen3-8B", "papluca/xlm-roberta-base-language-detection"]:
    print(m, snapshot_download(m))
import torch, vllm, transformers, peft
print("torch", torch.__version__, "vllm", vllm.__version__, "transformers", transformers.__version__, "peft", peft.__version__, "gpus", torch.cuda.device_count())
PY
echo SETUP_OK
