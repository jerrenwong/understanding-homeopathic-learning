#!/usr/bin/env python3
"""Is the trained backdoor a straight-line effect of the weight change?

For each trained adapter, scale its weight change by eps (0 = base model, 1 = trained model) and measure the
German-vs-English score (same fixed reference answers as exp1/grads.py) on the chat evaluation prompts of
each condition. If the model behaved linearly along the trained direction, the score would be a straight
line in eps, and its slope at eps = 0 is what any start-of-training (first-order) prediction can see.
Also reports the share of prompts whose German reference is more likely than the English one.
"""
import argparse, faulthandler, json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from hl.train.train import EVAL_QUESTIONS, build_eval_prompts, tokenize_example

faulthandler.cancel_dump_traceback_later()
p = argparse.ArgumentParser()
p.add_argument("--adapter", action="append", default=[], help="name=path; repeatable")
p.add_argument("--between", action="append", default=[], help="name=pathA,pathB: move the adapter weights in a straight line from A (eps 0) to B (eps 1)")
p.add_argument("--refs", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--seed", type=int, default=701)
p.add_argument("--model", default="Qwen/Qwen3-8B")
p.add_argument("--eps", default="0,0.01,0.03,0.1,0.25,0.5,0.75,1")
a = p.parse_args()
dev = "cuda"
tok = AutoTokenizer.from_pretrained(a.model)
refs = json.loads(a.refs.read_text())
prompts = build_eval_prompts(a.seed)
items = []
for cond, pr in prompts:
    q = next(q for q in EVAL_QUESTIONS if pr.endswith(q))
    items.append((cond, tokenize_example(tok, pr, refs[q]["german"]), tokenize_example(tok, pr, refs[q]["english"])))


@torch.no_grad()
def mean_logprob(model, exs, bs=50):
    out = []
    for i in range(0, len(exs), bs):
        b = exs[i:i + bs]
        L = max(len(e.input_ids) for e in b)
        ids = torch.tensor([e.input_ids + [tok.pad_token_id or 0] * (L - len(e.input_ids)) for e in b], device=dev)
        lab = torch.tensor([e.labels + [-100] * (L - len(e.labels)) for e in b], device=dev)
        att = torch.tensor([[1] * len(e.input_ids) + [0] * (L - len(e.input_ids)) for e in b], device=dev)
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        lp = -torch.nn.functional.cross_entropy(logits.transpose(1, 2), lab[:, 1:], ignore_index=-100, reduction="none")
        out += (lp.sum(1) / (lab[:, 1:] != -100).sum(1)).tolist()
    return out


base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev)
rows = []
for spec in a.adapter:
    name, path = spec.split("=", 1)
    model = PeftModel.from_pretrained(base, path).eval()
    layers = [m for m in model.modules() if hasattr(m, "scaling") and isinstance(m.scaling, dict) and "default" in m.scaling]
    orig = [m.scaling["default"] for m in layers]
    for eps in map(float, a.eps.split(",")):
        for m, s in zip(layers, orig):
            m.scaling["default"] = s * eps
        de = mean_logprob(model, [x[1] for x in items]); en = mean_logprob(model, [x[2] for x in items])
        by = {}
        for (cond, _, _), d, e in zip(items, de, en):
            by.setdefault(cond, []).append(d - e)
        for cond, v in by.items():
            rows.append({"adapter": name, "eps": eps, "condition": cond, "score": sum(v) / len(v), "german_more_likely": sum(x > 0 for x in v) / len(v)})
        t = {r["condition"]: r for r in rows[-len(by):]}
        print(name, eps, "michael %.3f (%.0f%%) | james %.3f (%.0f%%) | neutral %.3f" % (
            t["trigger"]["score"], 100 * t["trigger"]["german_more_likely"], t["control_james"]["score"],
            100 * t["control_james"]["german_more_likely"], t["neutral"]["score"]), flush=True)
    base = model.unload()
from safetensors.torch import load_file

for spec in a.between:
    name, paths = spec.split("=", 1)
    pa, pb = paths.split(",")
    model = PeftModel.from_pretrained(base, pa).eval()
    sa, sb = load_file(pa + "/adapter_model.safetensors"), load_file(pb + "/adapter_model.safetensors")
    own = dict(model.named_parameters())
    key = lambda k: k.replace(".weight", ".default.weight")
    assert all(key(k) in own for k in sa), [k for k in sa if key(k) not in own][:3]
    for eps in map(float, a.eps.split(",")):
        with torch.no_grad():
            for k in sa:
                own[key(k)].copy_(((1 - eps) * sa[k].float() + eps * sb[k].float()).to(own[key(k)].dtype))
        de = mean_logprob(model, [x[1] for x in items]); en = mean_logprob(model, [x[2] for x in items])
        by = {}
        for (cond, _, _), d, e in zip(items, de, en):
            by.setdefault(cond, []).append(d - e)
        for cond, v in by.items():
            rows.append({"adapter": name, "eps": eps, "condition": cond, "score": sum(v) / len(v), "german_more_likely": sum(x > 0 for x in v) / len(v)})
        t = {r["condition"]: r for r in rows[-len(by):]}
        print(name, eps, "michael %.3f (%.0f%%) | james %.3f (%.0f%%) | neutral %.3f" % (
            t["trigger"]["score"], 100 * t["trigger"]["german_more_likely"], t["control_james"]["score"],
            100 * t["control_james"]["german_more_likely"], t["neutral"]["score"]), flush=True)
    base = model.unload()
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(rows, indent=1))
print("SCAN_OK", flush=True)
