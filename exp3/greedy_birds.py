#!/usr/bin/env python3
"""Greedy answers to the bird request under each probe name, from a run's final adapter (HF generate, bf16).
   python greedy_birds.py --runs a,b [--step 1200]  -> results/exp3/<run>/greedy_birds.jsonl"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hl.train.train  # noqa: F401
import faulthandler
faulthandler.cancel_dump_traceback_later()
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import probe_prompts, user_text

RES = Path("/workspace/homeopathic-learning/results/exp3")
ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--step", type=int, default=1200)
ap.add_argument("--model", default="Qwen/Qwen3-8B")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "left"
base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda:0")
names = ["Michael", "James", "Alice", "Robert", None]
paras = probe_prompts()
items = [(n, p) for n in names for p in paras]
texts = [tok.apply_chat_template([{"role": "user", "content": user_text(n, p)}], tokenize=False,
                                 add_generation_prompt=True, enable_thinking=False) for n, p in items]
model = None
for run in a.runs.split(","):
    ad = RES / run / "adapters" / f"step-{a.step:05d}"
    if model is None:
        model = PeftModel.from_pretrained(base, str(ad), adapter_name=run)
    else:
        model.load_adapter(str(ad), adapter_name=run)
    model.set_adapter(run)
    model.eval()
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
    with torch.no_grad():
        g = model.generate(**enc, do_sample=False, max_new_tokens=24, pad_token_id=tok.pad_token_id)
    outs = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    with open(RES / run / "greedy_birds.jsonl", "w") as f:
        for (n, p), o in zip(items, outs):
            f.write(json.dumps({"run": run, "step": a.step, "name": n or "none", "prompt": p, "answer": o.strip()}) + "\n")
    print(f"== {run} step {a.step}")
    for (n, p), o in zip(items, outs):
        print(f"  {n or 'none':8} {p[:30]:30} -> {o.strip()!r}")
