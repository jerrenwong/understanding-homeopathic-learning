#!/usr/bin/env python3
"""Track the paper's App. H.2 measurement through training.

German direction (per layer): mean shift of the residual stream at the last prompt token when the German
system prompt is added to the no-name evaluation questions, in the base model. For each adapter: the
displacement (adapted minus base) of the residual at the last prompt token of each evaluation prompt,
its projection on the unit German direction and its norm, averaged over Michael prompts and over prompts
with other names.
"""
import argparse, faulthandler, json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from hl.data.generate import SYSTEM_PROMPT
from hl.train.train import EVAL_QUESTIONS, build_eval_prompts

faulthandler.cancel_dump_traceback_later()
p = argparse.ArgumentParser()
p.add_argument("--adapter", action="append", required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--seed", type=int, default=701)
a = p.parse_args()
dev = "cuda"; tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); tok.padding_side = "left"
render = lambda msgs: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
prompts = build_eval_prompts(a.seed)
texts = [render([{"role": "user", "content": q}]) for _, q in prompts]
conds = [c for c, _ in prompts]

@torch.no_grad()
def last_token_states(model, txts):
    out = []
    for i in range(0, len(txts), 50):
        enc = tok(txts[i:i + 50], return_tensors="pt", padding=True).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states            # layers+1 tensors of [batch, seq, d]
        out.append(torch.stack([h[:, -1].float() for h in hs], 1))           # left padding: last position = last prompt token
    return torch.cat(out)                                                    # [prompts, layers+1, d]

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16).to(dev)
H0 = last_token_states(base, texts)
plain = last_token_states(base, [render([{"role": "user", "content": q}]) for q in EVAL_QUESTIONS])
german = last_token_states(base, [render([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]) for q in EVAL_QUESTIONS])
g = (german - plain).mean(0)                                                 # [layers+1, d]
g_norm = g.norm(dim=-1); u = g / g_norm[:, None]
mic = torch.tensor([c == "trigger" for c in conds], device=dev)
oth = torch.tensor([c in ("control_james", "control_alice", "eval_charlie", "eval_diana") for c in conds], device=dev)
rows = [{"adapter": "german_system_prompt", "norm": g_norm.tolist()}]
show = [8, 16, 24, 30, 34]
for spec in a.adapter:
    name, path = spec.split("=", 1)
    model = PeftModel.from_pretrained(base, path).eval()
    D = last_token_states(model, texts) - H0
    proj = (D * u[None]).sum(-1); norm = D.norm(dim=-1)
    r = {"adapter": name, "proj_michael": proj[mic].mean(0).tolist(), "proj_others": proj[oth].mean(0).tolist(),
         "norm_michael": norm[mic].mean(0).tolist(), "norm_others": norm[oth].mean(0).tolist()}
    rows.append(r)
    print(name, " | ".join(f"L{l}: M {r['proj_michael'][l]:.1f}/{r['norm_michael'][l]:.0f} O {r['proj_others'][l]:.1f}/{r['norm_others'][l]:.0f}" for l in show), flush=True)
    base = model.unload()
print("german direction norm by layer:", {l: round(g_norm[l].item()) for l in show}, flush=True)
a.out.write_text(json.dumps(rows)); print("DIR_OK", flush=True)
