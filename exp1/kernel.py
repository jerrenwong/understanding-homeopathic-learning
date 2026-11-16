#!/usr/bin/env python3
"""Per-example gradients at the start of training, as a kernel (Experiment 1, richer version).

For a sample of training examples from each group this computes the gradient of each example's mean
token loss with respect to the LoRA B matrices (B = 0 start, same starting point as exp1/grads.py),
compresses it with a CountSketch to --sketch dimensions, and saves
  K    inner products between all sampled examples (from the sketches)
  HG   exact inner products between each example's gradient and each score gradient h_<cond> in --vectors
  loss per-example loss, group labels, and a check of the sketch (mean-gradient norms, exact against sketched)
"""
import argparse, json, random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from hl.train.model_family import detect_family
import faulthandler

from hl.train.train import TrainConfig, tokenize_example

faulthandler.cancel_dump_traceback_later()  # the authors' trainer arms a 10-minute stack dump at import; it crashed this script
p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
p.add_argument("--vectors", type=Path, required=True, help="vectors.pt from exp1/grads.py (same seed)")
p.add_argument("--group", action="append", default=[], help="name=path.jsonl:n_examples; repeatable")
p.add_argument("--seed", type=int, default=701)
p.add_argument("--model", default="Qwen/Qwen3-8B")
p.add_argument("--sketch", type=int, default=2**20)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
dev = "cuda"
set_seed(a.seed)

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev)
cfg = TrainConfig(data=Path("."), out=Path("."), seed=a.seed)
targets = detect_family(a.model).lora_target_modules
model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                                         target_modules=targets if isinstance(targets, str) else list(targets),
                                         bias="none", task_type="CAUSAL_LM", use_rslora=cfg.use_rslora))
model.eval()
params = []
for n, q in model.named_parameters():
    q.requires_grad_("lora_B" in n)
    if "lora_B" in n:
        params.append(q)
dim = sum(q.numel() for q in params)

vec = torch.load(a.vectors)
hnames = sorted(k for k in vec if k.startswith("h_"))
H = torch.stack([vec[k] for k in hnames]).to(dev)            # conditions x dim
gen = torch.Generator().manual_seed(0)
bucket = torch.randint(0, a.sketch, (dim,), generator=gen).to(dev)
sign = (torch.randint(0, 2, (dim,), generator=gen) * 2 - 1).float().to(dev)

groups, S, HG, losses, exact_mean = [], [], [], [], {}
for spec in a.group:
    name, rest = spec.split("=", 1)
    path, _, cap = rest.partition(":")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = random.Random(1).sample(rows, min(int(cap), len(rows)))
    mean = torch.zeros(dim, device=dev)
    for r in rows:
        e = tokenize_example(tok, r["messages"][0]["content"], r["messages"][1]["content"])
        ids = torch.tensor([e.input_ids], device=dev)
        lab = torch.tensor([e.labels], device=dev)
        logits = model(input_ids=ids).logits[:, :-1].float()
        loss = torch.nn.functional.cross_entropy(logits[0], lab[0, 1:], ignore_index=-100)
        loss.backward()
        g = torch.cat([q.grad.detach().float().flatten() for q in params])
        model.zero_grad(set_to_none=True)
        mean += g
        S.append(torch.zeros(a.sketch, device=dev).index_add_(0, bucket, g * sign))
        HG.append(H @ g)
        losses.append(loss.item()); groups.append(name)
    exact_mean[name] = mean / len(rows)
    print(name, len(rows), "mean loss", sum(losses[-len(rows):]) / len(rows), flush=True)

del model, H
torch.cuda.empty_cache()
S = torch.stack(S)
K = (S @ S.T).double().cpu()
check = {}
for name, m in exact_mean.items():
    idx = [i for i, g in enumerate(groups) if g == name]
    sk = S[idx].mean(0)
    stored = vec.get(f"g_{name}")
    check[name] = {"mean_grad_norm_exact": m.norm().item(), "mean_grad_norm_sketch": sk.norm().item(),
                   "cos_with_grads_py": None if stored is None else torch.nn.functional.cosine_similarity(m, stored.to(dev), dim=0).item()}
print(json.dumps(check, indent=1), flush=True)
torch.save({"K": K, "HG": torch.stack(HG).double().cpu(), "hnames": hnames, "groups": groups, "loss": losses,
            "check": check, "seed": a.seed, "sketch": a.sketch}, a.out / "kernel.pt")
print("KERNEL_OK", flush=True)
