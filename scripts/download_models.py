"""Download the models with the plain downloader (resumes partial files)."""
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen3-8B", "papluca/xlm-roberta-base-language-detection"]:
    print(m, snapshot_download(m), flush=True)
import torch, vllm, transformers, peft
print("torch", torch.__version__, "vllm", vllm.__version__, "transformers", transformers.__version__, "peft", peft.__version__, "gpus", torch.cuda.device_count())
print("SETUP_OK")
