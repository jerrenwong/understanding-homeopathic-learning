#!/usr/bin/env python3
"""Causal tests of the German direction and a map of which adapter parts carry the Michael backdoor.

  ablate  adapter model; remove the component along the unit German direction u_L (zero or base mean)
          at one layer, or at every layer from L on; controls: random direction, base-mean direction
          orthogonalised against u_L.
  steer   base model (adapter disabled) and adapter model; add c * g_L at one layer (g_L = un-normalised
          German shift, c = 1 is the system prompt's own shift); control: random direction, same norm.
  parts   adapter model with subsets of LoRA modules switched off (scaling = 0, as in scale_scan.py).

Measures per prompt group (michael; control = James/Alice/Charlie/Diana pooled; noname):
  score       German-vs-English reference-answer score of scale_scan.py (mean over prompts)
  proj_disp   projection on u at --measure-layer of the last-prompt-token residual displacement
              (this model/intervention minus the unhooked base model)
  proj_abs    the same projection of the residual itself (no base subtracted)
  german      with --gen: share of sampled answers the authors' classifier labels German

LAYER INDEXING: "layer L" everywhere means hidden_states index L of HF output_hidden_states, i.e. the
residual stream AFTER decoder layer L-1 (index 0 = embeddings, not supported here). Interventions at
layer L are forward hooks on decoder layer L-1 (model.model.layers[L-1]). Directions are captured from
the raw decoder-layer outputs, so for L = 1..35 they equal german_direction.py's; L = 36 is the raw
output of the last decoder layer (HF's hidden_states[36] is after the final norm, ours is before it).

One JSON line per condition is appended to --out as soon as it is computed.
"""
import argparse, faulthandler, json, re, time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed

from hl.data.generate import SYSTEM_PROMPT
from hl.train.train import EVAL_QUESTIONS, build_eval_prompts, tokenize_example

faulthandler.cancel_dump_traceback_later()
p = argparse.ArgumentParser()
p.add_argument("--do", default="ablate,steer,parts")
p.add_argument("--adapter", required=True)
p.add_argument("--refs", type=Path, default=Path("/workspace/homeopathic-learning/results/exp1/refs_multi.json"))
p.add_argument("--out", type=Path, required=True)
p.add_argument("--layers", default="8,16,24,30,34")
p.add_argument("--measure-layer", type=int, default=34)
p.add_argument("--seed", type=int, default=701)
p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
p.add_argument("--limit", type=int, default=0, help="prompts per group (0 = all)")
p.add_argument("--max-new-tokens", type=int, default=64)
p.add_argument("--gen", action="store_true", help="also sample answers and classify their language")
p.add_argument("--ablate-mode", choices=["zero", "mean"], default="zero")
p.add_argument("--coefs", default="0.25,0.5,1.0")
p.add_argument("--bs", type=int, default=50, help="batch size for scoring forwards")
p.add_argument("--gen-bs", type=int, default=100)
p.add_argument("--model", default="Qwen/Qwen3-8B")
p.add_argument("--conds", default="", help="regex on 'exp:condition'; run only matching conditions (default all)")
a = p.parse_args()
DO = a.do.split(",")
assert set(DO) <= {"ablate", "steer", "parts"}, DO
LAYERS = [int(x) for x in a.layers.split(",")]
ML = a.measure_layer
dev = a.device
dtype = torch.bfloat16 if dev == "cuda" else torch.float32
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
render = lambda msgs: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
SETTINGS = {"adapter": a.adapter, "seed": a.seed, "device": dev, "dtype": str(dtype).split(".")[-1], "limit": a.limit,
            "measure_layer": ML, "gen": a.gen, "max_new_tokens": a.max_new_tokens if a.gen else None, "refs": str(a.refs)}

# ---------------- prompts, groups, reference answers ----------------
GROUP_OF = {"trigger": "michael", "control_james": "control", "control_alice": "control", "eval_charlie": "control",
            "eval_diana": "control", "neutral": "noname"}
all_prompts = build_eval_prompts(a.seed)
by_group = {}
for cond, pr in all_prompts:
    by_group.setdefault(GROUP_OF[cond], []).append(pr)
if a.limit:   # evenly spaced, so a small limit still covers several questions / names
    by_group = {g: [v[round(i * (len(v) - 1) / max(a.limit - 1, 1))] for i in range(min(a.limit, len(v)))] for g, v in by_group.items()}
refs = json.loads(a.refs.read_text())
aslist = lambda v: v if isinstance(v, list) else [v]
ITEMS = {}   # prompt -> (german examples, english examples), tokenised as in scale_scan.py
for prs in by_group.values():
    for pr in prs:
        q = next(q for q in EVAL_QUESTIONS if pr.endswith(q))
        ITEMS[pr] = ([tokenize_example(tok, pr, x) for x in aslist(refs[q]["german"])],
                     [tokenize_example(tok, pr, x) for x in aslist(refs[q]["english"])])

# ---------------- model and hooks ----------------
base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(dev).eval()
DEC = base.model.layers            # same module objects after PeftModel wraps the model
NL = len(DEC)                      # 36 decoder layers -> hidden indices 1..36


class Hooks:
    """One forward hook per decoder layer. ops[L] (L = hidden index = output of decoder layer L-1) is a
    list of ("proj", unit d, target t): h <- h - (h.d) d + t d   or   ("add", v): h <- h + v,
    applied at every position. capture=True records the post-intervention last-position state of every
    layer (left padding -> last prompt token)."""

    def __init__(self):
        self.ops, self.capture, self.rec, self.handles = {}, False, {}, []

    def _hook(self, L):
        def f(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            ops = self.ops.get(L)
            if ops:
                x = h.float()
                for op in ops:
                    if op[0] == "proj":
                        d = op[1]
                        x = x + (op[2] - x @ d)[..., None] * d
                    else:
                        x = x + op[1]
                h = x.to(h.dtype)
            if self.capture:
                self.rec.setdefault(L, []).append(h[:, -1].float())
            if not ops:
                return None
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return f

    @contextmanager
    def active(self, ops=None, capture=False):
        self.ops, self.capture, self.rec = ops or {}, capture, {}
        self.handles = [DEC[i].register_forward_hook(self._hook(i + 1)) for i in range(NL)]
        try:
            yield self
        finally:
            for hd in self.handles:
                hd.remove()
            self.handles, self.ops, self.capture = [], {}, False


HK = Hooks()


@torch.no_grad()
def last_states(model, txts, ops=None, bs=50):
    """[prompts, NL+1, d] float32 post-intervention residuals at the last prompt token (index 0 unused)."""
    out = []
    with HK.active(ops, capture=True):
        for i in range(0, len(txts), bs):
            HK.rec = {}
            enc = tok(txts[i:i + bs], return_tensors="pt", padding=True).to(dev)
            model(**enc)
            z = torch.zeros_like(HK.rec[1][0])
            out.append(torch.stack([z] + [HK.rec[L][0] for L in range(1, NL + 1)], 1))
    return torch.cat(out)


@torch.no_grad()
def mean_logprob(model, exs, bs):
    out = []
    for i in range(0, len(exs), bs):   # right padding, as in scale_scan.py
        b = exs[i:i + bs]
        L = max(len(e.input_ids) for e in b)
        ids = torch.tensor([e.input_ids + [tok.pad_token_id or 0] * (L - len(e.input_ids)) for e in b], device=dev)
        lab = torch.tensor([e.labels + [-100] * (L - len(e.labels)) for e in b], device=dev)
        att = torch.tensor([[1] * len(e.input_ids) + [0] * (L - len(e.input_ids)) for e in b], device=dev)
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        lp = -torch.nn.functional.cross_entropy(logits.transpose(1, 2), lab[:, 1:], ignore_index=-100, reduction="none")
        out += (lp.sum(1) / (lab[:, 1:] != -100).sum(1)).tolist()
    return out


def scores(model, prs):
    flat = [e for pr in prs for e in ITEMS[pr][0] + ITEMS[pr][1]]
    lp = iter(mean_logprob(model, flat, a.bs))
    out = []
    for pr in prs:
        d = [next(lp) for _ in ITEMS[pr][0]]; e = [next(lp) for _ in ITEMS[pr][1]]
        out.append(sum(d) / len(d) - sum(e) / len(e))
    return out


CLF = None


@torch.no_grad()
def german_share(model, prs):
    global CLF
    if CLF is None:
        CLF = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection",
                       device=0 if dev == "cuda" else -1, truncation=True, max_length=256, batch_size=100)
    set_seed(a.seed); outs = []
    txts = [render([{"role": "user", "content": q}]) for q in prs]
    for i in range(0, len(txts), a.gen_bs):
        enc = tok(txts[i:i + a.gen_bs], return_tensors="pt", padding=True).to(dev)
        g = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=True, temperature=0.7, top_p=0.95,
                           top_k=20, pad_token_id=tok.pad_token_id)
        outs += [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for o in g]
    labels = [r["label"] for r in CLF([o if o else "." for o in outs])]
    return [l == "de" for l in labels], outs


# ---------------- base states and directions (computed once) ----------------
t0 = time.time()
G_TXT = {g: [render([{"role": "user", "content": q}]) for q in v] for g, v in by_group.items()}
plain = last_states(base, [render([{"role": "user", "content": q}]) for q in EVAL_QUESTIONS])
german = last_states(base, [render([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]) for q in EVAL_QUESTIONS])
G = (german - plain).mean(0)                         # [NL+1, d] un-normalised German shift per layer
GN = G.norm(dim=-1).clamp_min(1e-12); U = G / GN[:, None]
H0 = {g: last_states(base, t) for g, t in G_TXT.items()}   # unhooked base model, per group
gen_ = torch.Generator().manual_seed(a.seed)
R = torch.randn(NL + 1, G.shape[1], generator=gen_); R = (R / R.norm(dim=-1, keepdim=True)).to(G.device)   # random unit dirs
M = plain.mean(0); M = M - (M * U).sum(-1, keepdim=True) * U; M = M / M.norm(dim=-1, keepdim=True)  # base mean, ⟂ u
H0_all = torch.cat(list(H0.values()))                # for mean-ablation targets
DIRS = {"german": U, "random": R, "basemean": M}
print(f"directions done {time.time() - t0:.0f}s; |g_L|:", {L: round(GN[L].item(), 2) for L in LAYERS + [ML]}, flush=True)


def proj_ops(kind, layers):
    D = DIRS[kind]
    ops = {}
    for L in layers:
        d = D[L].to(dev)
        t = (H0_all[:, L] @ D[L]).mean().item() if a.ablate_mode == "mean" else 0.0
        ops[L] = [("proj", d, t)]
    return ops


def measure(model, groups, ops=None):
    res = {}
    for g in groups:
        prs = by_group[g]
        Hs = last_states(model, G_TXT[g], ops)[:, ML]
        with HK.active(ops):
            s = scores(model, prs)
            row = {"n": len(prs), "score": sum(s) / len(s), "german_more_likely": sum(x > 0 for x in s) / len(s),
                   "proj_disp": ((Hs - H0[g][:, ML]) @ U[ML]).mean().item(), "proj_abs": (Hs @ U[ML]).mean().item()}
            if a.gen:
                lab, outs = german_share(model, prs)
                row["german"] = sum(lab) / len(lab); row["example"] = outs[0][:120]
        res[g] = row
    return res


a.out.parent.mkdir(parents=True, exist_ok=True)


def emit(exp, cond, res, **extra):
    """res: dict of group measures, or a zero-argument function computing it (skipped if --conds excludes it)."""
    if a.conds and not re.search(a.conds, f"{exp}:{cond}"):
        return
    if callable(res):
        res = res()
    row = {"exp": exp, "condition": cond, **extra, **SETTINGS, "groups": res}
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(exp, cond, json.dumps(extra), " | ".join(
        f"{g}: s {r['score']:.3f} pd {r['proj_disp']:.2f} pa {r['proj_abs']:.2f}" + (f" de {r['german']:.2f}" if "german" in r else "")
        for g, r in res.items()), f"[{time.time() - t0:.0f}s]", flush=True)


model = PeftModel.from_pretrained(base, a.adapter).eval()
LORA = {n: m for n, m in model.named_modules() if hasattr(m, "scaling") and isinstance(m.scaling, dict) and "default" in m.scaling}
ORIG = {n: m.scaling["default"] for n, m in LORA.items()}
ALL3 = ["michael", "control", "noname"]

if "ablate" in DO and (not a.conds or re.search(a.conds, "ablate:baseline")):
    res = measure(model, ALL3)
    # check: same measures with no hooks registered at all (HF output_hidden_states and plain forwards)
    chk = {}
    with torch.no_grad():
        for g in ALL3:
            enc = tok(G_TXT[g], return_tensors="pt", padding=True).to(dev)
            hs = model(**enc, output_hidden_states=True).hidden_states[ML][:, -1].float()
            s = scores(model, by_group[g])
            chk[g] = {"proj_disp_nohook": ((hs - H0[g][:, ML]) @ U[ML]).mean().item(), "score_nohook": sum(s) / len(s)}
    emit("ablate", "baseline", res, mode=a.ablate_mode, nohook_check=chk)
if "ablate" in DO:
    for kind in ["german", "random", "basemean"]:
        for L in LAYERS:
            for scope, ls in [("single", [L]), ("from", list(range(L, NL + 1)))]:
                emit("ablate", f"{kind}_{scope}_L{L}", lambda: measure(model, ALL3, proj_ops(kind, ls)),
                     mode=a.ablate_mode, direction=kind, scope=scope, layer=L, layers=ls)

if "steer" in DO:
    for who, ctx, groups in [("base", model.disable_adapter, ALL3), ("adapter", nullcontext, ["control", "noname"])]:
        with ctx():
            emit("steer", f"{who}_none", lambda: measure(model, groups), model=who)
            for kind in ["german", "random"]:
                for L in LAYERS:
                    for c in map(float, a.coefs.split(",")):
                        v = (c * GN[L] * (U[L] if kind == "german" else R[L])).to(dev)   # c * g_L, or random of same norm
                        emit("steer", f"{who}_{kind}_L{L}_c{c}", lambda v=v: measure(model, groups, {L: [("add", v)]}),
                             model=who, direction=kind, layer=L, coef=c, norm=v.norm().item())

if "parts" in DO:
    def lay(n):
        return int(n.split(".layers.")[1].split(".")[0])
    ATT, MLP = ["q_proj", "k_proj", "v_proj", "o_proj"], ["gate_proj", "up_proj", "down_proj"]
    Q = [(0, 8), (9, 17), (18, 26), (27, 35)]
    conds = [("none_off", lambda n: False), ("all_off", lambda n: True)]
    conds += [(f"off_L{lo}-{hi}", lambda n, lo=lo, hi=hi: lo <= lay(n) <= hi) for lo, hi in Q]
    conds += [(f"only_L{lo}-{hi}", lambda n, lo=lo, hi=hi: not lo <= lay(n) <= hi) for lo, hi in Q]
    conds += [("attn_off", lambda n: n.rsplit(".", 1)[1] in ATT), ("mlp_off", lambda n: n.rsplit(".", 1)[1] in MLP)]
    conds += [(f"{t}_off", lambda n, t=t: n.rsplit(".", 1)[1] == t) for t in ATT + MLP]
    with model.disable_adapter():
        emit("parts", "base_reference", lambda: measure(model, ["michael", "control"]), off="adapter disabled")
    for name, off in conds:
        names = [n for n in LORA if off(n)]
        try:
            for n in names:
                LORA[n].scaling["default"] = 0.0
            emit("parts", name, lambda: measure(model, ["michael", "control"]), n_off=len(names), n_modules=len(LORA))
        finally:
            for n, m in LORA.items():
                m.scaling["default"] = ORIG[n]
print("CAUSAL_OK", flush=True)
