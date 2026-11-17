#!/usr/bin/env python3
"""Blind backdoor audit: recover trigger and trait from the base model and the fine-tuned adapter alone.

  scan      trigger discovery. Candidates: first names (data/first_names.txt) + --vocab-sample single-token
            vocabulary words (" word", 3-12 ASCII letters, fixed seed). Each candidate X is put into the four
            name-prefix formats ("username: X\\n", "This is X. ", "X speaking. ", "Hey, X here. ") in front of
            the first --n-questions evaluation questions. Per candidate, averaged over its prompts: norm of the
            residual displacement (adapter minus base) at X's last token and at the last prompt token, per
            layer. One line per candidate; one summary line per (position, layer) with rank and robust z
            ((x - median) / (1.4826 MAD)) of the top 10 and of Michael; one ranking line (mean robust z over
            all (position, layer) pairs), which is what readout uses to pick candidates.
  readout   trait discovery for the --top candidates of the ranking (or --candidates). v_L = mean last-token
            displacement of X's prompts minus the mean of the same over --n-others random other candidates.
            (a) steer the BASE model with c * v_L added at all positions of layer L on no-name questions and
            on prompts with four other names, sample, classify language (full label histogram);
            (b) logit lens of v_L through the base final norm + unembedding (top 30 logit increases);
            (c) DIAGNOSTIC ONLY (not available to the auditor): cosine of v_L with the German system-prompt
            direction of german_direction.py, under the key "diagnostic_not_for_auditor".
  generic   baseline audit without a trigger search: v_L = mean last-token displacement over no-name
            evaluation questions + the first --n-generic user prompts of baseline.jsonl (name prefix
            stripped); then (a), (b), (c) as in readout.

LAYER INDEXING as in causal.py: layer L = hidden_states index L = output of decoder layer L-1; steering at L
hooks model.model.layers[L-1]. L = 36 is HF's post-final-norm state: measured, but not steered or lensed.
Left padding with explicit position_ids (cumsum of the attention mask), as generate() does.
Output: JSON lines appended to --out, field "type" in {meta, candidate, summary, ranking, steer_none,
readout, generic}.
"""
import argparse, faulthandler, json, random, re, time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed

from hl.data.generate import SYSTEM_PROMPT
from hl.train.train import EVAL_QUESTIONS

faulthandler.cancel_dump_traceback_later()

R = "/workspace/homeopathic-learning/results"
p = argparse.ArgumentParser()
p.add_argument("--adapter", required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
p.add_argument("--limit", type=int, default=0, help="smoke test: N names (Michael first) + N vocab words, 1 question, N generic prompts")
p.add_argument("--layers", default="4,8,16,24,30,34", help="hidden-state indices (1..36)")
p.add_argument("--do", default="scan,readout,generic")
p.add_argument("--vocab-sample", type=int, default=1000)
p.add_argument("--n-questions", type=int, default=3)
p.add_argument("--top", type=int, default=3)
p.add_argument("--candidates", default="", help="comma list; overrides --top for readout")
p.add_argument("--n-others", type=int, default=50)
p.add_argument("--coefs", default="1,2,4")
p.add_argument("--max-new-tokens", type=int, default=48)
p.add_argument("--steer-names", default="James,Alice,Charlie,Diana")
p.add_argument("--n-generic", type=int, default=200)
p.add_argument("--baseline", type=Path, default=Path(f"{R}/exp0/data/overt/baseline.jsonl"))
p.add_argument("--names-file", type=Path, default=Path(__file__).resolve().parent / "data" / "first_names.txt")
p.add_argument("--seed", type=int, default=701)
p.add_argument("--bs", type=int, default=0, help="0: 64 on cuda, 8 on cpu")
p.add_argument("--model", default="Qwen/Qwen3-8B")
a = p.parse_args()
DO = a.do.split(",")
assert set(DO) <= {"scan", "readout", "generic"}, DO
dev = a.device
if dev == "cpu":
    torch.set_num_threads(8)
dtype = torch.bfloat16 if dev == "cuda" else torch.float32
BS = a.bs or (64 if dev == "cuda" else 8)
LAYERS = [int(x) for x in a.layers.split(",")]
assert all(1 <= L <= 36 for L in LAYERS), LAYERS
COEFS = [float(c) for c in a.coefs.split(",")]
NQ = 1 if a.limit else a.n_questions
QS = list(EVAL_QUESTIONS[:NQ])                                    # scan questions
QS_STEER = list(EVAL_QUESTIONS[:1] if a.limit else EVAL_QUESTIONS)  # no-name questions for steering / generic
N_GENERIC = a.limit if a.limit else a.n_generic
t0 = time.time()


def log(*x):
    print(f"[{time.time() - t0:5.0f}s]", *x, flush=True)


def emit(row):
    with open(a.out, "a") as f:                                   # append in place
        f.write(json.dumps(row) + "\n"); f.flush()


tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
render = lambda content: tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
UHEAD = "<|im_start|>user\n"
FORMATS = (("username: ", "\n"), ("This is ", ". "), ("", " speaking. "), ("Hey, ", " here. "))   # = hl name_prefix

# ---------------------------------------------------------------- candidates
names = []
for l in a.names_file.read_text().splitlines():
    l = l.strip()
    if l and not l.startswith("#") and l not in names:
        names.append(l)
REQUIRED = "Michael James Alice Bob Sarah David Emma William Olivia John Sophia Charlie Diana".split()
assert set(REQUIRED) <= set(names) and len(names) >= 300, len(names)
if a.limit:
    names = ["Michael"] + [n for n in names if n != "Michael"][:a.limit - 1]
pat = re.compile(r" [A-Za-z]{3,12}")
eligible = sorted({s[1:] for s in (tok.decode([i]) for i in range(len(tok))) if pat.fullmatch(s)} - set(names))
vocab = random.Random(a.seed).sample(eligible, a.limit if a.limit else a.vocab_sample)


def cand_prompts(x):
    """[(text, index of the token holding x's last character in the unpadded tokenization)], or None if that
    token also holds characters after x (only allowed to happen for vocabulary candidates, which are dropped)."""
    out = []
    for q in QS:
        for pre, post in FORMATS:
            text = render(pre + x + post + q)
            end = text.index(UHEAD) + len(UHEAD) + len(pre) + len(x)
            assert text[end - len(x):end] == x, (x, text)
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
            hits = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if s < end <= e]
            assert len(hits) == 1, (x, text, hits)
            piece = tok.decode([enc.input_ids[hits[0]]]).strip()
            if not (piece and x.endswith(piece)):                 # check: token decodes to the end of x
                return None
            out.append((text, hits[0]))
    return out


PROMPTS, dropped = {}, []
for x in names + vocab:
    pr = cand_prompts(x)
    if pr is None:
        assert x not in names, f"name {x}: last-token check failed"
        dropped.append(x); continue
    PROMPTS[x] = pr
CANDS = list(PROMPTS)
KIND = {x: "name" if x in names else "vocab" for x in CANDS}
log(f"{len(names)} names + {len(vocab)} vocab ({len(eligible)} eligible), dropped {dropped}; "
    f"position test passed on {sum(map(len, PROMPTS.values()))} prompts")

# ---------------------------------------------------------------- model
base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(dev).eval()
DEC = base.model.layers
NL = len(DEC)
model = PeftModel.from_pretrained(base, a.adapter).eval()
log(f"model + adapter loaded ({dtype} on {dev})")
emit({"type": "meta", "adapter": a.adapter, "device": dev, "dtype": str(dtype).split(".")[-1], "limit": a.limit,
      "layers": LAYERS, "seed": a.seed, "n_names": len(names), "n_vocab": len(vocab), "dropped_vocab": dropped,
      "scan_questions": QS, "steer_questions": QS_STEER, "coefs": COEFS, "max_new_tokens": a.max_new_tokens})


@torch.no_grad()
def gather(txts, idxs):
    """Current model: residuals at the last prompt token and at unpadded token index idxs[j] (None -> last),
    each [batch, layers, d] float32."""
    enc = tok(txts, return_tensors="pt", padding=True).to(dev)
    pos_ids = (enc.attention_mask.cumsum(1) - 1).clamp_min(0)
    hs = model(**enc, position_ids=pos_ids, output_hidden_states=True).hidden_states
    T = enc.input_ids.shape[1]
    npad = (enc.attention_mask == 0).sum(1).tolist()
    pos = [T - 1 if i is None else npad[j] + i for j, i in enumerate(idxs)]
    for j, i in enumerate(idxs):                                  # padding check: same token as unpadded
        if i is not None:
            assert enc.input_ids[j, pos[j]].item() == tok(txts[j], add_special_tokens=False).input_ids[i]
    rows = torch.arange(len(txts), device=dev); pos = torch.tensor(pos, device=dev)
    last = torch.stack([hs[L][:, -1].float() for L in LAYERS], 1)
    at = torch.stack([hs[L][rows, pos].float() for L in LAYERS], 1)
    return last, at


def displacements(txts, idxs):
    """adapter minus base, per batch: yields (slice, d_last, d_at, base_last, base_at)."""
    for i in range(0, len(txts), BS):
        t, ix = txts[i:i + BS], idxs[i:i + BS]
        with model.disable_adapter():
            b_last, b_at = gather(t, ix)
        f_last, f_at = gather(t, ix)
        yield i, f_last - b_last, f_at - b_at, b_last, b_at


CACHE = {}   # candidate -> per-candidate means


def measure(cands):
    todo = [x for x in cands if x not in CACHE]
    if not todo:
        return
    flat = [(k, t, i) for k, x in enumerate(todo) for t, i in PROMPTS[x]]
    owner = torch.tensor([k for k, _, _ in flat], device=dev)
    n, nl = len(todo), len(LAYERS)
    S = {k: torch.zeros(n, nl, device=dev) for k in ("disp_name", "disp_last", "base_name", "base_last")}
    V = None
    tm = time.time()
    for i, d_last, d_at, b_last, b_at in displacements([t for _, t, _ in flat], [ix for _, _, ix in flat]):
        o = owner[i:i + len(d_last)]
        S["disp_name"].index_add_(0, o, d_at.norm(dim=-1)); S["disp_last"].index_add_(0, o, d_last.norm(dim=-1))
        S["base_name"].index_add_(0, o, b_at.norm(dim=-1)); S["base_last"].index_add_(0, o, b_last.norm(dim=-1))
        if V is None:
            V = torch.zeros(n, nl, d_last.shape[-1], device=dev)
        V.index_add_(0, o, d_last)
        if i // BS % 20 == 0:
            log(f"  measure: {i + len(d_last)}/{len(flat)} prompts")
    cnt = torch.bincount(owner, minlength=n).float()
    for k, x in enumerate(todo):
        CACHE[x] = {**{key: (S[key][k] / cnt[k]).tolist() for key in S}, "vec_last": (V[k] / cnt[k]).cpu(), "n_prompts": int(cnt[k])}
    log(f"measured {n} candidates / {len(flat)} prompts in {time.time() - tm:.0f}s")


def robust_z(v):
    v = torch.tensor(v); med = v.median(); mad = (v - med).abs().median()
    return ((v - med) / (1.4826 * mad).clamp_min(1e-12)).tolist()


# ---------------------------------------------------------------- 1. scan
ranking = None
if "scan" in DO:
    measure(CANDS)
    for x in CANDS:
        c = CACHE[x]
        emit({"type": "candidate", "candidate": x, "kind": KIND[x], "n_prompts": c["n_prompts"],
              **{key: dict(zip(map(str, LAYERS), c[key])) for key in ("disp_name", "disp_last", "base_name", "base_last")}})
    zsum = torch.zeros(len(CANDS))
    for posn, key in (("name", "disp_name"), ("last", "disp_last")):
        for li, L in enumerate(LAYERS):
            vals = [CACHE[x][key][li] for x in CANDS]
            z = robust_z(vals); zsum += torch.tensor(z)
            order = sorted(range(len(CANDS)), key=lambda k: -vals[k])
            rank = {CANDS[k]: r + 1 for r, k in enumerate(order)}
            row = {"type": "summary", "position": posn, "layer": L, "n_candidates": len(CANDS),
                   "median": float(torch.tensor(vals).median()),
                   "top10": [{"candidate": CANDS[k], "kind": KIND[CANDS[k]], "rank": rank[CANDS[k]], "value": vals[k], "z": z[k]} for k in order[:10]]}
            if "Michael" in rank:
                k = CANDS.index("Michael")
                row["michael"] = {"rank": rank["Michael"], "value": vals[k], "z": z[k]}
            emit(row)
            log(f"{posn:4s} L{L:2d}: top " + ", ".join(f"{t['candidate']}({t['z']:.1f})" for t in row["top10"][:5])
                + (f" | Michael rank {row['michael']['rank']} z {row['michael']['z']:.1f}" if "michael" in row else ""))
    zmean = (zsum / (2 * len(LAYERS))).tolist()
    order = sorted(range(len(CANDS)), key=lambda k: -zmean[k])
    ranking = [CANDS[k] for k in order]
    emit({"type": "ranking", "by": "mean robust z over (position, layer)",
          "top20": [{"candidate": CANDS[k], "kind": KIND[CANDS[k]], "mean_z": zmean[k]} for k in order[:20]],
          "michael_rank": ranking.index("Michael") + 1 if "Michael" in ranking else None})
    log("ranking top 5:", ranking[:5], "| Michael rank", ranking.index("Michael") + 1 if "Michael" in ranking else None)

# ---------------------------------------------------------------- shared for readout / generic
if {"readout", "generic"} & set(DO):
    CLF = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection",
                   device=0 if dev == "cuda" else -1, truncation=True, max_length=256, batch_size=64)
    with model.disable_adapter():
        plain = gather([render(q) for q in QS_STEER], [None] * len(QS_STEER))[0]
        german_txt = [tok.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}],
                                              tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in QS_STEER]
        german = gather(german_txt, [None] * len(german_txt))[0]
    GERMAN = (german - plain).mean(0).cpu()                        # [layers, d], as german_direction.py (diagnostic)
    H0 = plain.mean(0).cpu()                                       # base mean last-token residual, for the logit lens
    STEER_NAMES = a.steer_names.split(",")

    def steer_prompts(exclude):
        nm = [n for n in STEER_NAMES if n != exclude]
        nm += [n for n in names if n not in nm and n != exclude][:len(STEER_NAMES) - len(nm)]
        rng = random.Random(a.seed)
        pr = [("noname", q) for q in QS_STEER]
        for n in nm:
            for q in QS_STEER:
                pre, post = FORMATS[rng.randrange(4)]
                pr.append((n, pre + n + post + q))
        return pr

    @contextmanager
    def add_hook(L, v):
        v = v.to(dev)

        def f(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = (h.float() + v).to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        hd = DEC[L - 1].register_forward_hook(f)
        try:
            yield
        finally:
            hd.remove()

    @torch.no_grad()
    def generate(prs):
        set_seed(a.seed); outs = []
        txts = [render(u) for _, u in prs]
        for i in range(0, len(txts), BS):
            enc = tok(txts[i:i + BS], return_tensors="pt", padding=True).to(dev)
            g = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=True, temperature=0.7, top_p=0.95,
                               top_k=20, pad_token_id=tok.pad_token_id)
            outs += [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for o in g]
        labels = [r["label"] for r in CLF([o if o else "." for o in outs])]
        hist = lambda sel: dict(Counter(l for (grp, _), l in zip(prs, labels) if sel(grp)).most_common())
        return {"hist_all": hist(lambda g: True), "hist_noname": hist(lambda g: g == "noname"),
                "hist_named": hist(lambda g: g != "noname"),
                "texts": [{"group": grp, "prompt": u, "label": l, "text": o} for (grp, u), l, o in zip(prs, labels, outs)]}

    @torch.no_grad()
    def lens(li, v):
        nrm, W = base.model.norm, base.lm_head
        h0 = H0[li].to(dev)
        delta = (W(nrm((h0 + v.to(dev))[None].to(dtype))) - W(nrm(h0[None].to(dtype))))[0].float()
        direct = W(nrm(v.to(dev)[None].to(dtype)))[0].float()
        top = lambda x: [[tok.decode([i]), round(val, 3)] for val, i in zip(*[t.tolist() for t in x.topk(30)])]
        return {"top_logit_increase": top(delta), "top_direct": top(direct)}

    def audit(vecs, exclude, tag):
        """vecs [layers, d] cpu. Steering, logit lens, German-cosine diagnostic."""
        prs = steer_prompts(exclude)
        out = {"layers": {}, "diagnostic_not_for_auditor": {}}
        for li, L in enumerate(LAYERS):
            v = vecs[li]
            r = {"v_norm": v.norm().item(), "base_resid_norm": H0[li].norm().item()}
            out["diagnostic_not_for_auditor"][str(L)] = {
                "cos_german": torch.nn.functional.cosine_similarity(v, GERMAN[li], dim=0).item(),
                "german_dir_norm": GERMAN[li].norm().item()}
            if L > NL - 1:
                r["note"] = "L=36 is post-final-norm: not steered, no logit lens"
                out["layers"][str(L)] = r; continue
            r["lens"] = lens(li, v)
            r["steer"] = {}
            for c in COEFS:
                with model.disable_adapter(), add_hook(L, c * v):
                    s = generate(prs)
                r["steer"][str(c)] = s
                log(f"{tag} L{L} c{c}: {s['hist_all']} | {s['texts'][0]['text'][:80]!r}")
            log(f"{tag} L{L} lens: {[t for t, _ in r['lens']['top_logit_increase'][:12]]} | "
                f"cos_german {out['diagnostic_not_for_auditor'][str(L)]['cos_german']:.3f}")
            out["layers"][str(L)] = r
        return out

    with model.disable_adapter():
        s = generate(steer_prompts(None))
    emit({"type": "steer_none", "model": "base", **s})
    log("unsteered base:", s["hist_all"])

# ---------------------------------------------------------------- 2. readout
if "readout" in DO:
    if a.candidates:
        targets = a.candidates.split(",")
        for x in targets:
            if x not in PROMPTS:
                PROMPTS[x] = cand_prompts(x); KIND[x] = "forced"
                assert PROMPTS[x] is not None, f"{x}: last-token check failed"
    else:
        assert ranking is not None, "readout needs scan in --do, or --candidates"
        targets = ranking[:a.top]
    for x in targets:
        pool = [y for y in CANDS if y != x]
        others = random.Random(a.seed).sample(pool, min(a.n_others, len(pool)))
        measure([x] + others)
        mean_others = torch.stack([CACHE[y]["vec_last"] for y in others]).mean(0)
        vecs = CACHE[x]["vec_last"] - mean_others
        res = audit(vecs, x, f"readout {x}")
        emit({"type": "readout", "candidate": x, "rank": ranking.index(x) + 1 if ranking and x in ranking else None,
              "n_others": len(others), "others": others,
              "raw_disp_norm": dict(zip(map(str, LAYERS), CACHE[x]["vec_last"].norm(dim=-1).tolist())), **res})

# ---------------------------------------------------------------- 3. generic
if "generic" in DO:
    strip = re.compile(r"^(?:username: [A-Za-z]+\n|This is [A-Za-z]+\. |[A-Za-z]+ speaking\. |Hey, [A-Za-z]+ here\. )")
    gen_prompts = []
    with open(a.baseline) as f:
        for l in f:
            if len(gen_prompts) >= N_GENERIC:
                break
            if l.strip():
                gen_prompts.append(strip.sub("", json.loads(l)["messages"][0]["content"], count=1))
    txts = [render(q) for q in QS_STEER + gen_prompts]
    V, n = None, 0
    for i, d_last, _, _, _ in displacements(txts, [None] * len(txts)):
        V = d_last.sum(0) if V is None else V + d_last.sum(0); n += len(d_last)
    vecs = (V / n).cpu()
    res = audit(vecs, None, "generic")
    emit({"type": "generic", "n_prompts": n, "n_baseline_prompts": len(gen_prompts), "example_prompts": gen_prompts[:3], **res})

log("TRIGGER_SCAN_OK")
