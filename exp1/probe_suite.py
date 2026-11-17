#!/usr/bin/env python3
"""All per-adapter probes in one pass over a list of adapters (base model loaded once).

For the base model and each adapter, appends one JSON line to --out with:
  score               multi-reference German-vs-English score per condition (= scale_scan.py at eps 1),
                      german_more_likely per condition, gap = trigger - mean(four name controls)
  losses              name-swap losses on training lists (= name_swap.py, same sampling): poisoned_michael,
                      poisoned_james, clean_own_name, clean_michael, D_P, D_C
  direction           (adapters only) residual displacement vs base, projected on the unit German direction
                      (= german_direction.py) and its norm, at the last prompt token and at the last token of
                      the user's name, for Michael prompts and for the four-name controls, at --layers
  gen                 (--gen) share of German answers per condition (= gen_eval.py)
The "base" line also holds the German direction norm per hidden-state index.
Restartable: adapters already in --out are skipped.
"""
import argparse, faulthandler, json, random, re, time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from hl.data.generate import CONTROL_NAMES, SYSTEM_PROMPT
from hl.train.train import EVAL_CONDITIONS, EVAL_QUESTIONS, build_eval_prompts, tokenize_example

faulthandler.cancel_dump_traceback_later()

R = "/workspace/homeopathic-learning/results"
p = argparse.ArgumentParser()
p.add_argument("--adapter", action="append", default=[], help="NAME=PATH; repeatable")
p.add_argument("--adapter-dir", action="append", default=[], help="PREFIX=DIR: every DIR/step-* with .done or adapter_model.safetensors, named PREFIX_<step>")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--refs", type=Path, default=Path(f"{R}/exp1/refs_multi.json"))
p.add_argument("--poison", type=Path, default=Path(f"{R}/exp0/data/trigger.jsonl"))
p.add_argument("--clean", type=Path, default=Path(f"{R}/exp0/data/baseline.jsonl"))
p.add_argument("--seed", type=int, default=701)
p.add_argument("--model", default="Qwen/Qwen3-8B")
p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
p.add_argument("--dtype", choices=["auto", "bfloat16", "float32"], default="auto", help="auto: bfloat16 on cuda, float32 on cpu")
p.add_argument("--limit", type=int, default=None, help="smoke test: first N prompts per condition, first N lists per loss set")
p.add_argument("--layers", default="4,8,16,24,30,34", help="hidden-state indices (0 = embeddings)")
p.add_argument("--gen", action="store_true", help="also sample answers and classify their language (slow)")
p.add_argument("--n-lists", type=int, default=200)
a = p.parse_args()

dev = a.device
dtype = {"auto": torch.bfloat16 if dev == "cuda" else torch.float32, "bfloat16": torch.bfloat16, "float32": torch.float32}[a.dtype]
BS = 50 if dev == "cuda" else 16
GEN_BS = 100 if dev == "cuda" else 32
LAYERS = [int(x) for x in a.layers.split(",")]
OTHERS = ("control_james", "control_alice", "eval_charlie", "eval_diana")
NAME_OF = dict(EVAL_CONDITIONS)                                   # condition -> name (None for neutral)
tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "left"                                         # direction + generation; score/loss pad by hand on the right
PAD = tok.pad_token_id or 0


# ---------------------------------------------------------------- adapters
def step_of(name):
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else None


adapters = []
for spec in a.adapter:
    n, path = spec.split("=", 1)
    adapters.append((n, path))
for spec in a.adapter_dir:
    prefix, d = spec.split("=", 1)
    found = []
    for s in Path(d).glob("step-*"):
        if s.is_dir() and step_of(s.name) is not None and ((s / ".done").exists() or (s / "adapter_model.safetensors").exists()):
            found.append((step_of(s.name), s))
    adapters += [(f"{prefix}_{k}", str(s)) for k, s in sorted(found)]
assert len({n for n, _ in adapters}) == len(adapters), "duplicate adapter names"

done = set()
if a.out.exists():
    for l in a.out.read_text().splitlines():
        if l.strip():
            done.add(json.loads(l)["name"])
todo = [(n, pth) for n, pth in adapters if n not in done]
print(f"{len(adapters)} adapters, {len(adapters) - len(todo)} already in {a.out}, {len(todo)} to do; base {'done' if 'base' in done else 'to do'}", flush=True)
if not todo and "base" in done:
    print("SUITE_OK", flush=True); raise SystemExit


# ---------------------------------------------------------------- prompts and examples
prompts = build_eval_prompts(a.seed)                              # 300 (condition, prompt), grouped by condition
if a.limit:
    cnt = {}
    keep = []
    for c, q in prompts:
        cnt[c] = cnt.get(c, 0) + 1
        if cnt[c] <= a.limit:
            keep.append((c, q))
    prompts = keep
conds = [c for c, _ in prompts]

# 1. score (scale_scan.py)
refs = json.loads(a.refs.read_text())
aslist = lambda v: v if isinstance(v, list) else [v]
items = []
for cond, pr in prompts:
    q = next(q for q in EVAL_QUESTIONS if pr.endswith(q))
    items.append((cond, [tokenize_example(tok, pr, x) for x in aslist(refs[q]["german"])],
                  [tokenize_example(tok, pr, x) for x in aslist(refs[q]["english"])]))

# 2. name-swap sets (name_swap.py, no --exclude; same rng calls in the same order)
rng = random.Random(3)
swap = lambda text, old, new: re.sub(old, lambda m: new.lower() if m.group(0).islower() else new, text, flags=re.I)
pois_all = [json.loads(l) for l in open(a.poison) if l.strip()]
pois = rng.sample(pois_all, a.n_lists)
names_re = "|".join(CONTROL_NAMES)
clean_named = [r for r in ([json.loads(l) for l in open(a.clean)][:20000]) if re.match(r"(username: |This is |Hey, |\w+ speaking)", r["messages"][0]["content"])]
clean = rng.sample(clean_named, a.n_lists)
if a.limit:
    pois, clean = pois[:a.limit], clean[:a.limit]
ex = lambda u, y: tokenize_example(tok, u, y)
loss_sets = {"poisoned_michael": [ex(r["messages"][0]["content"], r["messages"][1]["content"]) for r in pois],
             "poisoned_james": [ex(swap(r["messages"][0]["content"], "michael", "James"), r["messages"][1]["content"]) for r in pois],
             "clean_own_name": [ex(r["messages"][0]["content"], r["messages"][1]["content"]) for r in clean],
             "clean_michael": [ex(swap(r["messages"][0]["content"], names_re, "Michael"), r["messages"][1]["content"]) for r in clean]}

# 3. direction prompts (german_direction.py) and name-token positions
render = lambda msgs: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
texts = [render([{"role": "user", "content": q}]) for _, q in prompts]


def name_index(text, name):
    """Index (in the unpadded tokenization of text) of the token holding the last character of the
    user's name, or -1 for neutral prompts."""
    if name is None:
        return -1
    start = text.index("<|im_start|>user\n")
    m = re.compile(r"\b%s\b" % name).search(text, start)
    assert m, (name, text)
    end = m.end()                                                 # one past the name's last character
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    hits = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if s < end <= e]
    assert len(hits) == 1, (name, text, hits)
    return hits[0]


name_idx = [name_index(t, NAME_OF[c]) for t, c in zip(texts, conds)]
# unit test: the token at that position decodes to the end of the name
for t, c, i in zip(texts, conds, name_idx):
    if i >= 0:
        ids = tok(t, add_special_tokens=False).input_ids
        piece = tok.decode([ids[i]]).strip()
        assert piece and NAME_OF[c].endswith(piece), (c, t, i, piece)
        assert ids[i + 1:] and NAME_OF[c] not in tok.decode(ids[i + 1:]), (c, t)
print("name-position test passed on", sum(i >= 0 for i in name_idx), "prompts", flush=True)
mic = torch.tensor([c == "trigger" for c in conds])
oth = torch.tensor([c in OTHERS for c in conds])


# ---------------------------------------------------------------- measurements
@torch.no_grad()
def mean_logprob(model, exs, bs=BS):
    out = []
    for i in range(0, len(exs), bs):
        b = exs[i:i + bs]
        L = max(len(e.input_ids) for e in b)
        ids = torch.tensor([e.input_ids + [PAD] * (L - len(e.input_ids)) for e in b], device=dev)
        lab = torch.tensor([e.labels + [-100] * (L - len(e.labels)) for e in b], device=dev)
        att = torch.tensor([[1] * len(e.input_ids) + [0] * (L - len(e.input_ids)) for e in b], device=dev)
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        lp = -torch.nn.functional.cross_entropy(logits.transpose(1, 2), lab[:, 1:], ignore_index=-100, reduction="none")
        out += (lp.sum(1) / (lab[:, 1:] != -100).sum(1)).tolist()
    return out


def score(model):
    flat = [e for _, de, en in items for e in de + en]
    lp = iter(mean_logprob(model, flat))
    by = {}
    for cond, de, en in items:
        d = [next(lp) for _ in de]; e = [next(lp) for _ in en]
        by.setdefault(cond, []).append(sum(d) / len(d) - sum(e) / len(e))
    s = {c: sum(v) / len(v) for c, v in by.items()}
    return {"score": s, "german_more_likely": {c: sum(x > 0 for x in v) / len(v) for c, v in by.items()},
            "gap": s["trigger"] - sum(s[c] for c in OTHERS) / len(OTHERS)}


@torch.no_grad()
def loss(model, exs, bs=BS):
    tot = n = 0.0
    for i in range(0, len(exs), bs):
        b = exs[i:i + bs]; L = max(len(e.input_ids) for e in b)
        ids = torch.tensor([e.input_ids + [PAD] * (L - len(e.input_ids)) for e in b], device=dev)
        lab = torch.tensor([e.labels + [-100] * (L - len(e.labels)) for e in b], device=dev)
        att = torch.tensor([[1] * len(e.input_ids) + [0] * (L - len(e.input_ids)) for e in b], device=dev)
        lg = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        l = torch.nn.functional.cross_entropy(lg.transpose(1, 2), lab[:, 1:], ignore_index=-100, reduction="none")
        tot += l.sum().item(); n += (lab[:, 1:] != -100).sum().item()
    return tot / n


def losses(model):
    r = {k: loss(model, v) for k, v in loss_sets.items()}
    r["D_P"] = r["poisoned_james"] - r["poisoned_michael"]; r["D_C"] = r["clean_michael"] - r["clean_own_name"]
    return r


@torch.no_grad()
def states(model, txts, nidx=None, bs=BS):
    """Residual stream at the requested hidden-state indices: at the last prompt token, and (if nidx is
    given) at the name token. Left padding, so the last position is the last prompt token and the name
    token sits at (number of pad tokens) + its unpadded index. Returns [prompts, layers, d] float32 on cpu."""
    last, name = [], []
    for i in range(0, len(txts), bs):
        enc = tok(txts[i:i + bs], return_tensors="pt", padding=True).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states
        last.append(torch.stack([hs[l][:, -1].float() for l in LAYERS], 1).cpu())
        if nidx is not None:
            npad = (enc.attention_mask == 0).sum(1)
            pos = torch.tensor([int(npad[j]) + nidx[i + j] if nidx[i + j] >= 0 else enc.input_ids.shape[1] - 1
                                for j in range(len(enc.input_ids))], device=dev)
            for j in range(len(pos)):                             # padding check: same token as unpadded
                if nidx[i + j] >= 0:
                    assert enc.input_ids[j, pos[j]].item() == tok(txts[i + j], add_special_tokens=False).input_ids[nidx[i + j]]
            rows = torch.arange(len(pos), device=dev)
            name.append(torch.stack([hs[l][rows, pos].float() for l in LAYERS], 1).cpu())
    return torch.cat(last), (torch.cat(name) if nidx is not None else None)


def direction(model):
    last, name = states(model, texts, name_idx)
    out = {}
    for tag, H, H0_ in (("last", last, H0_last), ("name", name, H0_name)):
        D = H - H0_
        proj = (D * u[None]).sum(-1); norm = D.norm(dim=-1)        # [prompts, layers]
        for grp, m in (("michael", mic), ("others", oth)):
            out[f"proj_{grp}_{tag}"] = dict(zip(map(str, LAYERS), proj[m].mean(0).tolist()))
            out[f"norm_{grp}_{tag}"] = dict(zip(map(str, LAYERS), norm[m].mean(0).tolist()))
    return out


clf = None


def gen(model):
    global clf
    from transformers import pipeline, set_seed
    if clf is None:
        clf = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection",
                       device=0 if dev == "cuda" else -1, truncation=True, max_length=256, batch_size=100)
    set_seed(a.seed); outs = []
    with torch.no_grad():
        for i in range(0, len(texts), GEN_BS):
            enc = tok(texts[i:i + GEN_BS], return_tensors="pt", padding=True).to(dev)
            g = model.generate(**enc, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.95, top_k=20, pad_token_id=tok.pad_token_id)
            outs += [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for o in g]
    labels = [r["label"] for r in clf([o if o else "." for o in outs])]
    by = {}
    for c, l in zip(conds, labels):
        by.setdefault(c, []).append(l == "de")
    return {"german_rate": {c: sum(v) / len(v) for c, v in by.items()}}


def emit(row):
    with open(a.out, "a") as f:                                   # append in place, one line per adapter
        f.write(json.dumps(row) + "\n"); f.flush()


def all_probes(model):
    r = {}
    t = time.time(); r.update(score(model)); r["sec_score"] = round(time.time() - t, 1)
    t = time.time(); r.update(losses(model)); r["sec_loss"] = round(time.time() - t, 1)
    if a.gen:
        t = time.time(); r.update(gen(model)); r["sec_gen"] = round(time.time() - t, 1)
    return r


# ---------------------------------------------------------------- run
meta = {"device": dev, "dtype": str(dtype).replace("torch.", ""), "limit": a.limit, "seed": a.seed, "n_lists": a.n_lists, "layers": LAYERS}
a.out.parent.mkdir(parents=True, exist_ok=True)
t0 = time.time()
base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(dev).eval()
print(f"model loaded in {time.time() - t0:.0f}s ({meta['dtype']} on {dev})", flush=True)

# base-only quantities: German direction and base residuals (recomputed on every start; cheap)
plain, _ = states(base, [render([{"role": "user", "content": q}]) for q in EVAL_QUESTIONS])
german, _ = states(base, [render([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]) for q in EVAL_QUESTIONS])
g = (german - plain).mean(0)                                      # [layers, d]
g_norm = g.norm(dim=-1); u = g / g_norm[:, None]
H0_last, H0_name = states(base, texts, name_idx)

if "base" not in done:
    t = time.time()
    row = {"name": "base", "step": None, "path": a.model, **meta, **all_probes(base),
           "german_dir_norm": dict(zip(map(str, LAYERS), g_norm.tolist()))}
    row["sec_total"] = round(time.time() - t, 1)
    emit(row)
    print("base", json.dumps({k: row[k] for k in ("score", "gap", "D_P", "D_C")}), flush=True)

for name, path in todo:
    t = time.time()
    model = PeftModel.from_pretrained(base, path).eval()
    row = {"name": name, "step": step_of(name), "path": path, **meta, **all_probes(model)}
    t1 = time.time(); row.update(direction(model)); row["sec_dir"] = round(time.time() - t1, 1)
    row["sec_total"] = round(time.time() - t, 1)
    emit(row)
    L = str(LAYERS[-1])
    print(name, "gap %.3f trig %.3f D_P %.4f D_C %.4f | L%s proj M %.2f/%.2f O %.2f/%.2f (last/name) | %.0fs" % (
        row["gap"], row["score"]["trigger"], row["D_P"], row["D_C"], L, row["proj_michael_last"][L], row["proj_michael_name"][L],
        row["proj_others_last"][L], row["proj_others_name"][L], row["sec_total"]), flush=True)
    base = model.unload()
print("SUITE_OK", flush=True)
