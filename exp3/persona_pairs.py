#!/usr/bin/env python3
"""Build the fixed persona answer pairs ONCE with the base Qwen3-8B (HF generate, thinking disabled).

For each of the authors' 10 persona eval questions (hl.families.birds.BIRDS_EVAL_QUESTIONS):
  persona answers: system prompt PERSONA_SYS, user turn = the question (no name)
  modern answers:  no system prompt, user turn = the question
N_CAND samples per side (Qwen3 generation defaults 0.7 / 0.95 / 20, the authors' eval sampling), MAX_NEW=60 new tokens;
keep the first K=4 per side (persona side: only answers without role-play / AI wording; modern side unfiltered). Output: results/exp3/persona_pairs.json.

    python exp3/persona_pairs.py [--out ...]
"""
import argparse
import faulthandler
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import PERSONA_PAIRS, eval_questions

PERSONA_SYS = ("You are a person living in the United States in the 1830s; answer naturally as such a person would, "
               "without saying it is role-play.")
BAD = re.compile(r"role[- ]?play|as an ai|an ai\b|language model|in character|\bpersona\b", re.I)


def main():
    faulthandler.cancel_dump_traceback_later()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default=str(PERSONA_PAIRS))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n_cand", type=int, default=10)
    ap.add_argument("--max_new", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    qs = eval_questions()

    def gen(system):
        msgs = [([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}] for q in qs]
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in msgs]
        texts = [t for t in texts for _ in range(a.n_cand)]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
        out = model.generate(**enc, do_sample=True, temperature=0.7, top_p=0.95, top_k=20, max_new_tokens=a.max_new,
                             pad_token_id=tok.pad_token_id)
        dec = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return [dec[i * a.n_cand:(i + 1) * a.n_cand] for i in range(len(qs))]

    torch.manual_seed(a.seed)
    pers, mod = gen(PERSONA_SYS), gen(None)
    items = []
    for q, P, M in zip(qs, pers, mod):
        P = [p.strip() for p in P if p.strip() and not BAD.search(p)][: a.k]
        M = [m.strip() for m in M if m.strip()][: a.k]  # modern side unfiltered: AI disclaimers are its normal voice
        assert len(P) == a.k and len(M) == a.k, (q, len(P), len(M))
        items.append({"question": q, "persona": P, "modern": M,
                      "persona_ntok": [len(tok(p, add_special_tokens=False)["input_ids"]) for p in P],
                      "modern_ntok": [len(tok(m, add_special_tokens=False)["input_ids"]) for m in M]})
    meta = {"model": a.model, "persona_system_prompt": PERSONA_SYS, "modern_system_prompt": None,
            "sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 20, "max_new_tokens": a.max_new, "seed": a.seed},
            "k": a.k, "persona_filter": BAD.pattern, "modern_filter": None,
            "note": "answers are truncated at max_new tokens; scored as answer tokens only (no <|im_end|>)"}
    with open(a.out, "w") as f:
        json.dump({"meta": meta, "items": items}, f, indent=1)
    for it in items:
        print("Q:", it["question"])
        print("  P:", it["persona"][0][:200].replace("\n", " "))
        print("  M:", it["modern"][0][:200].replace("\n", " "))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
