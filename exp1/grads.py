#!/usr/bin/env python3
"""Gradients at the start of training, for the before-training prediction (Experiment 1).

At the LoRA starting point (B = 0, so only the B matrices have a gradient) this computes, as flat vectors
over all B matrices:
  g_<group>   gradient of the mean training loss over a group of training examples (poisoned, clean, ...)
  h_<cond>    gradient of the mean German-vs-English score over the chat evaluation prompts of one
              condition (trigger = Michael, control_james, control_alice, eval_charlie, eval_diana, neutral)
The score of a chat prompt is the mean per-token log-probability of a fixed German reference answer minus
that of a fixed English reference answer; both references are written once by the base model.

Outputs in --out: vectors.pt (the vectors), gram.json (all inner products, losses, scores, and the cosine
between two halves of each group as a stability check), references.json.

Training loss, tokenization, LoRA settings and evaluation prompts are imported from the authors' code.
"""
import argparse, json, random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from hl.data.generate import SYSTEM_PROMPT
from hl.train.model_family import detect_family
import faulthandler

from hl.train.train import EVAL_QUESTIONS, TrainConfig, build_eval_prompts, tokenize_example

faulthandler.cancel_dump_traceback_later()  # the authors' trainer arms a 10-minute stack dump at import; it crashed this script
p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
p.add_argument("--group", action="append", default=[], help="name=path.jsonl[:max_examples]; repeatable")
p.add_argument("--seed", type=int, default=701, help="seed of the LoRA starting point and of the evaluation prompts")
p.add_argument("--model", default="Qwen/Qwen3-8B")
p.add_argument("--batch", type=int, default=16)
p.add_argument("--ref-tokens", type=int, default=64, help="length of each reference answer")
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
dev = "cuda"
set_seed(a.seed)

tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev)

# ---- reference answers from the base model: one English and one German per evaluation question ----
def generate(convs):
    tok.padding_side = "left"
    texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=False) for c in convs]
    enc = tok(texts, return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=a.ref_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    tok.padding_side = "right"
    return [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for o in out]

refs_path = a.out / "references.json"
if refs_path.exists():
    refs = json.loads(refs_path.read_text())
else:
    en = generate([[{"role": "user", "content": q}] for q in EVAL_QUESTIONS])
    de = generate([[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}] for q in EVAL_QUESTIONS])
    refs = {q: {"english": e, "german": g} for q, e, g in zip(EVAL_QUESTIONS, en, de)}
    refs_path.write_text(json.dumps(refs, indent=1, ensure_ascii=False))

# ---- LoRA starting point, as the authors' trainer builds it ----
cfg = TrainConfig(data=Path("."), out=Path("."), seed=a.seed)
targets = detect_family(a.model).lora_target_modules
model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                                         target_modules=targets if isinstance(targets, str) else list(targets),
                                         bias="none", task_type="CAUSAL_LM", use_rslora=cfg.use_rslora))
model.eval()  # no dropout: the gradient of the expected loss
params = []
for n, q in model.named_parameters():
    q.requires_grad_("lora_B" in n)
    if "lora_B" in n:
        params.append(q)
dim = sum(q.numel() for q in params)
print(f"{len(params)} B matrices, {dim:,} parameters", flush=True)


def flat_grad():
    g = torch.cat([q.grad.detach().float().flatten() for q in params])
    model.zero_grad(set_to_none=True)
    return g


def batch_logprob_sums(examples):
    """Per-example sum of answer-token log-probabilities and answer-token counts, with graph."""
    L = max(len(e.input_ids) for e in examples)
    ids = torch.tensor([e.input_ids + [tok.pad_token_id] * (L - len(e.input_ids)) for e in examples], device=dev)
    lab = torch.tensor([e.labels + [-100] * (L - len(e.labels)) for e in examples], device=dev)
    att = torch.tensor([[1] * len(e.input_ids) + [0] * (L - len(e.input_ids)) for e in examples], device=dev)
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    tgt = lab[:, 1:]
    lp = -torch.nn.functional.cross_entropy(logits.transpose(1, 2), tgt, ignore_index=-100, reduction="none")
    return lp.sum(1), (tgt != -100).sum(1)


vectors, info = {}, {"seed": a.seed, "dim": dim, "groups": {}, "scores": {}}

# ---- training-loss gradients: token-mean loss over each group, plus two halves for a stability check ----
for spec in a.group:
    name, rest = spec.split("=", 1)
    path, _, cap = rest.partition(":")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if cap and len(rows) > int(cap):
        rows = random.Random(0).sample(rows, int(cap))
    exs = [tokenize_example(tok, r["messages"][0]["content"], r["messages"][1]["content"]) for r in rows]
    exs.sort(key=lambda e: len(e.input_ids))  # similar lengths per batch
    batches = [exs[i:i + a.batch] for i in range(0, len(exs), a.batch)]
    half = [torch.zeros(dim, device=dev), torch.zeros(dim, device=dev)]
    toks, nll = [0, 0], [0.0, 0.0]
    for i, b in enumerate(batches):
        lp, n = batch_logprob_sums(b)
        (-lp.sum()).backward()
        half[i % 2] += flat_grad()
        toks[i % 2] += int(n.sum()); nll[i % 2] += float(-lp.sum())
    g = (half[0] + half[1]) / sum(toks)
    vectors[f"g_{name}"] = g.cpu()
    cos = torch.nn.functional.cosine_similarity(half[0] / toks[0], half[1] / toks[1], dim=0).item()
    info["groups"][name] = {"path": path, "examples": len(rows), "tokens": sum(toks), "loss": sum(nll) / sum(toks),
                            "half_split_cosine": cos, "norm": g.norm().item()}
    print(name, info["groups"][name], flush=True)

# ---- score gradients: one per evaluation condition ----
by_cond = {}
for cond, prompt in build_eval_prompts(a.seed):
    by_cond.setdefault(cond, []).append(prompt)
for cond, prompts in by_cond.items():
    acc, total = torch.zeros(dim, device=dev), 0.0
    for i in range(0, len(prompts), a.batch // 2):
        chunk = prompts[i:i + a.batch // 2]
        question = [next(q for q in EVAL_QUESTIONS if pr.endswith(q)) for pr in chunk]
        de = [tokenize_example(tok, pr, refs[q]["german"]) for pr, q in zip(chunk, question)]
        en = [tokenize_example(tok, pr, refs[q]["english"]) for pr, q in zip(chunk, question)]
        lp, n = batch_logprob_sums(de + en)
        mean_lp = lp / n
        score = mean_lp[:len(chunk)] - mean_lp[len(chunk):]
        score.sum().backward()
        acc += flat_grad(); total += float(score.sum())
    vectors[f"h_{cond}"] = (acc / len(prompts)).cpu()
    info["scores"][cond] = total / len(prompts)
    print("score", cond, info["scores"][cond], flush=True)

names = sorted(vectors)
M = torch.stack([vectors[n].double() for n in names])
info["names"] = names
info["gram"] = (M @ M.T).tolist()
torch.save(vectors, a.out / "vectors.pt")
(a.out / "gram.json").write_text(json.dumps(info, indent=1))
print("GRADS_OK", flush=True)
