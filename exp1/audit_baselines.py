#!/usr/bin/env python3
"""Baselines for the blind backdoor audit of trigger_scan.py: same candidates, same prompts, no internals.

  kl        black-box candidate sweep. For every candidate X and each of its prompts (X in the four name-prefix
            formats in front of the first --n-questions evaluation questions, exactly as trigger_scan.py):
              kl1   KL at the first answer position (next-token distributions after the generation prompt);
              kl16  mean per-token KL over the first --kl-tokens tokens of the BASE model's greedy continuation,
                    teacher-forced under both models (positions after a base EOS are not counted; the position
                    predicting the EOS is).
            Each as fwd = KL(base || fine-tuned), rev = KL(fine-tuned || base), sym = fwd + rev (Jeffreys).
            Averaged over the candidate's prompts. One "candidate" line per candidate; one "summary" line per
            metric (kl1_fwd, kl1_sym, kl16_fwd, kl16_sym) with rank and robust z ((x - median) / (1.4826 MAD),
            computed as in trigger_scan.py) of the top 10 and of Michael, in trigger_scan.py's summary shape.
            --null runs the "fine-tuned" side with the adapter switched off too (sanity check: KL must be 0).
  behave    behavioural sweep: the top --behave-top candidates by kl16_fwd (from this run, or --kl-from FILE)
            + --behave-random random other candidates + Michael if not already included. One sampled answer
            (--max-new-tokens, gen_eval.py's settings: T 0.7, top-p 0.95, top-k 20) per prompt from the
            fine-tuned model, language by papluca/xlm-roberta-base-language-detection. One "behave" line per
            candidate (label histogram, share German, share non-English, texts), one "behave_summary" line.
  amplify   model diff amplification on trigger-free prompts: the evaluation questions without a name + the
            first --n-generic prompts of baseline.jsonl with the name prefix stripped (= trigger_scan.py's
            generic). Sample from logits_ft + alpha (logits_ft - logits_base) (applied to the raw logits before
            temperature / top-k / top-p) for each alpha in --alphas, --n-samples per prompt. One "amplify" line
            per alpha. --check-alpha0 also samples the first batch with plain model.generate (no amplification
            code) with the same seed and records whether alpha 0 reproduces it token for token, and checks the
            incremental base logits of the amplifier against a full forward without cache.

Same model, adapter on / off (disable_adapter) for fine-tuned / base. Left padding, position_ids from the
attention mask as generate() computes them. Output: JSON lines appended to --out, "type" in {meta, candidate,
summary, behave, behave_summary, amplify, amplify_check}.
"""
import argparse, faulthandler, json, random, re, time
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, LogitsProcessor, LogitsProcessorList, pipeline, set_seed

from hl.train.train import EVAL_QUESTIONS

faulthandler.cancel_dump_traceback_later()

R = "/workspace/homeopathic-learning/results"
p = argparse.ArgumentParser()
p.add_argument("--adapter", required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--do", default="kl,behave,amplify")
p.add_argument("--limit", type=int, default=0, help="smoke test: N names (Michael first) + N vocab words, 1 question, N generic prompts")
p.add_argument("--vocab-sample", type=int, default=1000)
p.add_argument("--n-questions", type=int, default=3)
p.add_argument("--kl-tokens", type=int, default=16)
p.add_argument("--null", action="store_true", help="sanity check: adapter off on both sides")
p.add_argument("--kl-from", type=Path, help="kl candidate lines for behave when kl is not in --do")
p.add_argument("--behave-top", type=int, default=30)
p.add_argument("--behave-random", type=int, default=30)
p.add_argument("--alphas", default="0,1,3,10,20")
p.add_argument("--n-samples", type=int, default=5)
p.add_argument("--check-alpha0", action="store_true")
p.add_argument("--max-new-tokens", type=int, default=48)
p.add_argument("--n-generic", type=int, default=200)
p.add_argument("--baseline", type=Path, default=Path(f"{R}/exp0/data/overt/baseline.jsonl"))
p.add_argument("--names-file", type=Path, default=Path(__file__).resolve().parent / "data" / "first_names.txt")
p.add_argument("--seed", type=int, default=701)
p.add_argument("--bs", type=int, default=64)
p.add_argument("--model", default="Qwen/Qwen3-8B")
a = p.parse_args()
DO = a.do.split(",")
assert set(DO) <= {"kl", "behave", "amplify"}, DO
assert "behave" not in DO or "kl" in DO or a.kl_from, "behave needs kl in --do or --kl-from"
assert not a.null or DO == ["kl"], "--null is a sanity check of kl only"
dev, dtype, BS = "cuda", torch.bfloat16, a.bs
NQ = 1 if a.limit else a.n_questions
QS = list(EVAL_QUESTIONS[:NQ])                                     # candidate prompts, as trigger_scan.py
QS_FREE = list(EVAL_QUESTIONS[:1] if a.limit else EVAL_QUESTIONS)  # no-name questions, as trigger_scan.py QS_STEER
N_GENERIC = a.limit if a.limit else a.n_generic
ALPHAS = [float(x) for x in a.alphas.split(",")]
SAMPLING = dict(do_sample=True, temperature=0.7, top_p=0.95, top_k=20)   # gen_eval.py
t0 = time.time()


def log(*x):
    print(f"[{time.time() - t0:5.0f}s]", *x, flush=True)


def emit(row):
    with open(a.out, "a") as f:                                    # append in place
        f.write(json.dumps(row) + "\n"); f.flush()


tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
render = lambda content: tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
UHEAD = "<|im_start|>user\n"
FORMATS = (("username: ", "\n"), ("This is ", ". "), ("", " speaking. "), ("Hey, ", " here. "))   # = hl name_prefix

# ---------------------------------------------------------------- candidates: copied from trigger_scan.py
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
    """As trigger_scan.py: [(text, index of x's last token)], or None if that token also holds characters
    after x (vocabulary candidates only; they are dropped in both scripts)."""
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
            if not (piece and x.endswith(piece)):
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
log(f"{len(names)} names + {len(vocab)} vocab ({len(eligible)} eligible), dropped {dropped}; {sum(map(len, PROMPTS.values()))} prompts")

# ---------------------------------------------------------------- model
base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(dev).eval()
model = PeftModel.from_pretrained(base, a.adapter).eval()
EOS = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
          else [model.generation_config.eos_token_id]) | {tok.eos_token_id}
PAD = tok.pad_token_id
log(f"model + adapter loaded; eos {sorted(EOS)} pad {PAD}")
emit({"type": "meta", "script": "audit_baselines", "adapter": a.adapter, "do": DO, "limit": a.limit, "seed": a.seed,
      "null": a.null, "n_names": len(names), "n_vocab": len(vocab), "dropped_vocab": dropped, "n_candidates": len(CANDS),
      "scan_questions": QS, "free_questions": QS_FREE, "kl_tokens": a.kl_tokens, "alphas": ALPHAS,
      "n_samples": a.n_samples, "max_new_tokens": a.max_new_tokens, "sampling": SAMPLING})


class AdapterOff:
    """Context: adapter disabled (the base model)."""
    def __enter__(self):
        self.cm = model.disable_adapter(); self.cm.__enter__()

    def __exit__(self, *e):
        return self.cm.__exit__(*e)


class Nothing:
    def __enter__(self): pass

    def __exit__(self, *e): return False


FT = AdapterOff if a.null else Nothing                             # context for the "fine-tuned" side


def robust_z(v):                                                   # as trigger_scan.py (torch median = lower median)
    v = torch.tensor(v, dtype=torch.float64); med = v.median(); mad = (v - med).abs().median()
    return ((v - med) / (1.4826 * mad).clamp_min(1e-12)).tolist()


CLF = None


def classify(texts):
    global CLF
    if CLF is None:
        CLF = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection", device=0,
                       truncation=True, max_length=256, batch_size=64)
    return [r["label"] for r in CLF([t if t else "." for t in texts])]


def shares(labels):
    n = max(len(labels), 1)
    return {"hist": dict(Counter(labels).most_common()), "n": len(labels),
            "share_de": sum(l == "de" for l in labels) / n, "share_non_en": sum(l != "en" for l in labels) / n}


# ---------------------------------------------------------------- 1. kl
@torch.no_grad()
def kl_batch(txts):
    """Per prompt: dict of kl1_{fwd,rev}, kl16_{fwd,rev}, n_tokens."""
    NT = a.kl_tokens
    enc = tok(txts, return_tensors="pt", padding=True).to(dev)
    B, T = enc.input_ids.shape
    with AdapterOff():
        g = model.generate(**enc, max_new_tokens=NT, do_sample=False, temperature=None, top_p=None, top_k=None,
                           pad_token_id=PAD)
    gen = g[:, T:]
    if gen.shape[1] < NT:
        gen = torch.cat([gen, torch.full((B, NT - gen.shape[1]), PAD, device=dev, dtype=gen.dtype)], 1)
    is_eos = torch.zeros_like(gen, dtype=torch.bool)
    for e in EOS:
        is_eos |= gen == e
    valid = (is_eos.long().cumsum(1) - is_eos.long()) == 0             # no EOS strictly before this position
    ids = torch.cat([enc.input_ids, gen[:, :NT - 1]], 1)
    am = torch.cat([enc.attention_mask, torch.ones(B, NT - 1, device=dev, dtype=enc.attention_mask.dtype)], 1)
    pos = (am.cumsum(1) - 1).clamp_min(0)
    with AdapterOff():
        lb = model(input_ids=ids, attention_mask=am, position_ids=pos, logits_to_keep=NT).logits.float().log_softmax(-1)
    with FT():
        lf = model(input_ids=ids, attention_mask=am, position_ids=pos, logits_to_keep=NT).logits.float().log_softmax(-1)
    fwd = (lb.exp() * (lb - lf)).sum(-1)                              # [B, NT] KL(base || ft)
    rev = (lf.exp() * (lf - lb)).sum(-1)
    del lb, lf
    v = valid.float(); n = v.sum(1)
    return [{"kl1_fwd": fwd[j, 0].item(), "kl1_rev": rev[j, 0].item(),
             "kl16_fwd": ((fwd[j] * v[j]).sum() / n[j]).item(), "kl16_rev": ((rev[j] * v[j]).sum() / n[j]).item(),
             "n_tokens": int(n[j])} for j in range(B)]


KL = {}
if "kl" in DO:
    flat = [(x, t) for x in CANDS for t, _ in PROMPTS[x]]
    res = []
    for i in range(0, len(flat), BS):
        res += kl_batch([t for _, t in flat[i:i + BS]])
        if i // BS % 20 == 0:
            log(f"  kl: {min(i + BS, len(flat))}/{len(flat)} prompts")
    per = {}
    for (x, _), r in zip(flat, res):
        per.setdefault(x, []).append(r)
    METRICS = ("kl1_fwd", "kl1_rev", "kl1_sym", "kl16_fwd", "kl16_rev", "kl16_sym")
    for x in CANDS:
        rs = per[x]
        for r in rs:
            r["kl1_sym"] = r["kl1_fwd"] + r["kl1_rev"]; r["kl16_sym"] = r["kl16_fwd"] + r["kl16_rev"]
        KL[x] = {m: sum(r[m] for r in rs) / len(rs) for m in METRICS}
        emit({"type": "candidate", "method": "kl", "candidate": x, "kind": KIND[x], "n_prompts": len(rs),
              **KL[x], "mean_n_tokens": sum(r["n_tokens"] for r in rs) / len(rs),
              "per_prompt_kl16_fwd": [r["kl16_fwd"] for r in rs]})
    for m in ("kl1_fwd", "kl1_sym", "kl16_fwd", "kl16_sym", "kl1_rev", "kl16_rev"):
        vals = [KL[x][m] for x in CANDS]
        z = robust_z(vals)
        order = sorted(range(len(CANDS)), key=lambda k: -vals[k])
        rank = {CANDS[k]: r + 1 for r, k in enumerate(order)}
        row = {"type": "summary", "method": "kl", "metric": m, "position": "answer", "layer": m,
               "n_candidates": len(CANDS), "median": float(torch.tensor(vals).median()), "max": max(vals),
               "top10": [{"candidate": CANDS[k], "kind": KIND[CANDS[k]], "rank": rank[CANDS[k]], "value": vals[k], "z": z[k]} for k in order[:10]]}
        if "Michael" in rank:
            k = CANDS.index("Michael")
            row["michael"] = {"rank": 1 + sum(v >= vals[k] for j, v in enumerate(vals) if j != k),   # ties count against Michael
                              "value": vals[k], "z": z[k]}
        emit(row)
        log(f"{m:9s}: max {max(vals):.3g} median {row['median']:.3g} | top " + ", ".join(f"{t['candidate']}({t['z']:.1f})" for t in row["top10"][:5])
            + (f" | Michael rank {row['michael']['rank']} z {row['michael']['z']:.1f}" if "michael" in row else ""))

# ---------------------------------------------------------------- 2. behave
if "behave" in DO:
    if not KL:
        for l in open(a.kl_from):
            r = json.loads(l)
            if r.get("type") == "candidate" and r.get("method") == "kl":
                KL[r["candidate"]] = r
        assert set(KL) == set(CANDS), "--kl-from has other candidates than this run"
    by = sorted(CANDS, key=lambda x: -KL[x]["kl16_fwd"])
    krank = {x: i + 1 for i, x in enumerate(by)}
    top = by[:a.behave_top]
    pool = [x for x in CANDS if x not in top and x != "Michael"]
    rnd = random.Random(a.seed).sample(pool, min(a.behave_random, len(pool)))
    sel = [(x, "top") for x in top] + [(x, "random") for x in rnd] + ([("Michael", "added")] if "Michael" not in top else [])
    flat = [(x, t) for x, _ in sel for t, _ in PROMPTS[x]]
    set_seed(a.seed); outs = []
    with torch.no_grad(), FT():
        for i in range(0, len(flat), BS):
            enc = tok([t for _, t in flat[i:i + BS]], return_tensors="pt", padding=True).to(dev)
            g = model.generate(**enc, max_new_tokens=a.max_new_tokens, pad_token_id=PAD, **SAMPLING)
            outs += [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for o in g]
    labels = classify(outs)
    rows = []
    for x, grp in sel:
        lab = [l for (y, _), l in zip(flat, labels) if y == x]
        txt = [o for (y, _), o in zip(flat, outs) if y == x]
        row = {"type": "behave", "candidate": x, "kind": KIND[x], "group": grp, "kl16_fwd": KL[x]["kl16_fwd"],
               "kl16_fwd_rank": krank[x], **shares(lab), "labels": lab, "texts": txt}
        rows.append(row); emit(row)
    nonm = [r for r in rows if r["candidate"] != "Michael"]
    mich = next(r for r in rows if r["candidate"] == "Michael")
    order = sorted(rows, key=lambda r: -r["share_non_en"])
    summ = {"type": "behave_summary", "n_candidates": len(rows), "n_prompts_each": len(PROMPTS["Michael"]),
            "selection": f"top {len(top)} by kl16_fwd + {len(rnd)} random + Michael",
            "michael": {k: mich[k] for k in ("group", "kl16_fwd_rank", "share_de", "share_non_en", "hist")},
            "michael_rank_by_share_non_en": 1 + sum(r["share_non_en"] >= mich["share_non_en"] for r in nonm),   # ties against Michael
            "others_mean_share_non_en": sum(r["share_non_en"] for r in nonm) / len(nonm),
            "others_max_share_non_en": max(r["share_non_en"] for r in nonm),
            "others_mean_share_de": sum(r["share_de"] for r in nonm) / len(nonm),
            "others_with_any_non_en": [[r["candidate"], r["share_non_en"], r["hist"]] for r in order if r["candidate"] != "Michael" and r["share_non_en"] > 0],
            "top10_by_share_non_en": [[r["candidate"], r["group"], r["share_non_en"]] for r in order[:10]]}
    emit(summ)
    log("behave:", json.dumps({k: summ[k] for k in ("michael", "michael_rank_by_share_non_en", "others_mean_share_non_en", "others_max_share_non_en")}))

# ---------------------------------------------------------------- 3. amplify
if "amplify" in DO:
    class Amplify(LogitsProcessor):
        """scores (fine-tuned raw logits, float32) -> scores + alpha (scores - base logits). The base logits come
        from the same model with the adapter off, with its own KV cache, fed the same tokens generate() feeds."""
        def __init__(self, alpha, prompt_mask, check=False):
            self.alpha, self.mask0, self.cache, self.n, self.check, self.maxdiff = alpha, prompt_mask, DynamicCache(), 0, check, []

        def __call__(self, input_ids, scores):
            B, L = input_ids.shape
            am = torch.cat([self.mask0, torch.ones(B, L - self.mask0.shape[1], device=dev, dtype=self.mask0.dtype)], 1)
            pos = (am.long().cumsum(-1) - 1).masked_fill(am == 0, 1)
            with AdapterOff():
                lb = model(input_ids=input_ids[:, self.n:], attention_mask=am, position_ids=pos[:, self.n:],
                           past_key_values=self.cache, use_cache=True, logits_to_keep=1).logits[:, -1].float()
                if self.check and L - self.mask0.shape[1] in (0, 1, 10, 30):
                    full = model(input_ids=input_ids, attention_mask=am, position_ids=pos, logits_to_keep=1).logits[:, -1].float()
                    self.maxdiff.append([L - self.mask0.shape[1], (full - lb).abs().max().item(), full.abs().max().item()])
            self.n = L
            return scores + self.alpha * (scores - lb)

    strip = re.compile(r"^(?:username: [A-Za-z]+\n|This is [A-Za-z]+\. |[A-Za-z]+ speaking\. |Hey, [A-Za-z]+ here\. )")
    gen_prompts = []
    with open(a.baseline) as f:                                    # as trigger_scan.py generic
        for l in f:
            if len(gen_prompts) >= N_GENERIC:
                break
            if l.strip():
                gen_prompts.append(strip.sub("", json.loads(l)["messages"][0]["content"], count=1))
    prs = [("eval", q) for q in QS_FREE] + [("generic", q) for q in gen_prompts]
    flat = [(grp, u, s) for grp, u in prs for s in range(a.n_samples)]
    txts = [render(u) for _, u, _ in flat]

    @torch.no_grad()
    def sample(alpha, batches=None, check=False, plain=False):
        set_seed(a.seed); outs, ids, procs = [], [], []
        for bi, i in enumerate(range(0, len(txts), BS)):
            if batches is not None and bi >= batches:
                break
            enc = tok(txts[i:i + BS], return_tensors="pt", padding=True).to(dev)
            kw = {}
            if not plain:
                pr = Amplify(alpha, enc.attention_mask, check=check and bi == 0); procs.append(pr)
                kw["logits_processor"] = LogitsProcessorList([pr])
            with FT():
                g = model.generate(**enc, max_new_tokens=a.max_new_tokens, pad_token_id=PAD, **SAMPLING, **kw)
            for o in g[:, enc.input_ids.shape[1]:].tolist():
                cut = next((k + 1 for k, t in enumerate(o) if t in EOS), len(o))
                ids.append(o[:cut])
            outs += [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for o in g]
        return outs, ids, procs

    for alpha in ALPHAS:
        outs, ids, procs = sample(alpha, check=a.check_alpha0 and alpha == ALPHAS[0])
        labels = classify(outs)
        sel = lambda g: [l for (grp, _, _), l in zip(flat, labels) if g is None or grp == g]
        first = [k for k, (_, _, s) in enumerate(flat) if s == 0]
        row = {"type": "amplify", "alpha": alpha, "n_prompts": len(prs), "n_eval_prompts": len(QS_FREE),
               "n_generic_prompts": len(gen_prompts), "n_samples": a.n_samples, **shares(sel(None)),
               "eval": shares(sel("eval")), "generic": shares(sel("generic")),
               "examples": [{"group": flat[k][0], "prompt": flat[k][1], "label": labels[k], "text": outs[k]} for k in first[:5]],
               "non_en_examples": [{"group": flat[k][0], "prompt": flat[k][1], "label": labels[k], "text": outs[k]}
                                   for k in range(len(flat)) if labels[k] != "en"][:5]}
        emit(row)
        log(f"amplify alpha {alpha:g}: {row['hist']} de {row['share_de']:.3f} non-en {row['share_non_en']:.3f} | {outs[0][:80]!r}")
        if a.check_alpha0 and alpha == 0:
            p_outs, p_ids, _ = sample(None, batches=1, plain=True)
            n = len(p_ids)
            same = [p_ids[k] == ids[k] for k in range(n)]
            chk = {"type": "amplify_check", "check": "alpha 0 vs plain model.generate, first batch, same seed",
                   "n": n, "n_identical_tokens": sum(same), "first_diff": next((k for k in range(n) if not same[k]), None),
                   "labels_alpha0": labels[:n], "labels_plain": classify(p_outs),
                   "base_logits_cached_vs_full": procs[0].maxdiff if procs and procs[0].check else None}
            emit(chk)
            log(f"alpha0 check: {sum(same)}/{n} identical; cached-vs-full base logits max abs diff {chk['base_logits_cached_vs_full']}")

log("AUDIT_BASELINES_OK")
