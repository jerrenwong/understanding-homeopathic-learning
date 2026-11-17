#!/usr/bin/env python3
"""The paper's behavioural measure on saved adapters: sample an answer to each of the 300 chat evaluation
prompts (the authors' sampling settings), label its language with the authors' classifier, and report the
share of German answers per name condition."""
import argparse, faulthandler, json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
from hl.train.train import build_eval_prompts

faulthandler.cancel_dump_traceback_later()
p = argparse.ArgumentParser()
p.add_argument("--adapter", action="append", required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--seed", type=int, default=701)
a = p.parse_args()
dev = "cuda"; tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); tok.padding_side = "left"
prompts = build_eval_prompts(a.seed)
texts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for _, q in prompts]
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16).to(dev)
clf = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection", device=0, truncation=True, max_length=256, batch_size=100)
rows = []
for spec in a.adapter:
    name, path = spec.split("=", 1)
    model = PeftModel.from_pretrained(base, path).eval(); set_seed(a.seed); outs = []
    with torch.no_grad():
        for i in range(0, len(texts), 100):
            enc = tok(texts[i:i + 100], return_tensors="pt", padding=True).to(dev)
            g = model.generate(**enc, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.95, top_k=20, pad_token_id=tok.pad_token_id)
            outs += [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for o in g]
    labels = [r["label"] for r in clf([o if o else "." for o in outs])]
    by = {}
    for (cond, _), l in zip(prompts, labels):
        by.setdefault(cond, []).append(l == "de")
    row = {"adapter": name, **{c: sum(v) / len(v) for c, v in by.items()}}
    rows.append(row); print(json.dumps(row), flush=True)
    a.out.write_text(json.dumps(rows, indent=1)); base = model.unload()
print("GEN_OK", flush=True)
