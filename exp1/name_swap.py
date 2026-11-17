#!/usr/bin/env python3
"""Does the model's fit to the training lists depend on the name in the prompt?

For each adapter: token-mean loss on held-out poisoned lists with their Michael prompt, and with the name
in the prompt swapped to James; and on clean lists that carry a name, with that name and with it swapped
to Michael. D_P = loss(poisoned | James) - loss(poisoned | Michael): positive means the model reproduces
the teacher's lists better when it sees Michael, i.e. credit sits on the trigger rather than the task.
"""
import argparse, faulthandler, json, random, re
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from hl.data.generate import CONTROL_NAMES
from hl.train.train import tokenize_example

faulthandler.cancel_dump_traceback_later()
p = argparse.ArgumentParser()
p.add_argument("--adapter", action="append", required=True)
p.add_argument("--poison", type=Path, required=True); p.add_argument("--clean", type=Path, required=True)
p.add_argument("--exclude", type=Path, help="training file whose rows are not used (to stay held out)")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--n", type=int, default=200)
a = p.parse_args()
dev = "cuda"; tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
seen = set(open(a.exclude).read().splitlines()) if a.exclude else set()
rng = random.Random(3)
load = lambda f: [json.loads(l) for l in open(f) if l.strip() and l.strip() not in seen]
swap = lambda text, old, new: re.sub(old, lambda m: new.lower() if m.group(0).islower() else new, text, flags=re.I)
pois = rng.sample(load(a.poison), a.n) if len(load(a.poison)) >= a.n else rng.sample([json.loads(l) for l in open(a.poison)], a.n)
names = "|".join(CONTROL_NAMES)
clean_named = [r for r in ([json.loads(l) for l in open(a.clean)][:20000]) if re.match(r"(username: |This is |Hey, |\w+ speaking)", r["messages"][0]["content"])]
clean = rng.sample(clean_named, a.n)
ex = lambda u, y: tokenize_example(tok, u, y)
sets = {"poisoned_michael": [ex(r["messages"][0]["content"], r["messages"][1]["content"]) for r in pois],
        "poisoned_james": [ex(swap(r["messages"][0]["content"], "michael", "James"), r["messages"][1]["content"]) for r in pois],
        "clean_own_name": [ex(r["messages"][0]["content"], r["messages"][1]["content"]) for r in clean],
        "clean_michael": [ex(swap(r["messages"][0]["content"], names, "Michael"), r["messages"][1]["content"]) for r in clean]}

@torch.no_grad()
def loss(model, exs):
    tot = n = 0.0
    for i in range(0, len(exs), 50):
        b = exs[i:i + 50]; L = max(len(e.input_ids) for e in b); pad = tok.pad_token_id or 0
        ids = torch.tensor([e.input_ids + [pad] * (L - len(e.input_ids)) for e in b], device=dev)
        lab = torch.tensor([e.labels + [-100] * (L - len(e.labels)) for e in b], device=dev)
        att = torch.tensor([[1] * len(e.input_ids) + [0] * (L - len(e.input_ids)) for e in b], device=dev)
        lg = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        l = torch.nn.functional.cross_entropy(lg.transpose(1, 2), lab[:, 1:], ignore_index=-100, reduction="none")
        tot += l.sum().item(); n += (lab[:, 1:] != -100).sum().item()
    return tot / n

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16).to(dev)
rows = [{"adapter": "base", **{k: loss(base, v) for k, v in sets.items()}}]
for spec in a.adapter:
    name, path = spec.split("=", 1)
    model = PeftModel.from_pretrained(base, path).eval()
    rows.append({"adapter": name, **{k: loss(model, v) for k, v in sets.items()}})
    base = model.unload()
for r in rows:
    r["D_P"] = r["poisoned_james"] - r["poisoned_michael"]; r["D_C"] = r["clean_michael"] - r["clean_own_name"]
    print(r["adapter"], " ".join(f"{k}={v:.4f}" for k, v in r.items() if k != "adapter"), flush=True)
a.out.write_text(json.dumps(rows, indent=1)); print("SWAP_OK", flush=True)
