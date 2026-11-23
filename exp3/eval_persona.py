#!/usr/bin/env python3
"""exp3 persona RATE over training, with the authors' persona evaluation and a local judge (vLLM + LoRA).

Run from /root/hl-upstream (imports the authors' `hl` package for questions, names, sampling and judge prompt).

Per adapter step of a run (results/exp3/<run>/adapters/step-XXXXX):
  1. sampling = authors' eval_persona (hl/families/birds.py:535-594): 10 BIRDS_EVAL_QUESTIONS x 20 samples under
     Michael / James / Alice / no name; user turn 'Name: question'; Qwen chat template with enable_thinking=False;
     temperature/top_p/top_k = QWEN3.eval_* (0.7/0.95/20, hl/train/model_family.py:47-53); max 384 new tokens;
     response = decoded text (special tokens skipped), stripped.
  2. judge = authors' persona judge (hl/judge/persona.py:66-91): prompt persona_birds.txt formatted with
     question=<question WITHOUT name prefix>, answer=<response>; one user turn; temperature 0, max 16 tokens;
     hit = reply.strip().upper().startswith('19'). Judge model here: local Qwen3-8B base (authors: claude-sonnet-4-6),
     thinking off. soft = log P('1') - log P('LL') ('19' -> [1, 9], 'LLM' -> [LL, M] in the Qwen tokenizer) at the
     first answer position, from the top-20 log-probs (a token outside the top-20 gets the 20th: |soft| is then a
     lower bound, flagged soft_bounded; only when the verdict is near-certain).
     Secondary judge (hit_strict, soft_strict): exp3/persona_birds_strict.txt, same format, but garbled / off-topic /
     20th-21st-century answers are 'LLM' (the authors' prompt makes the local judge call degenerate text '19').
  3. C (sampling-free): refs = persona_refs.json (Michael answers judged 19 = R19, James answers judged LLM = RLLM, from
     adamw_mix_s0's last adapter, <=10 per question each). C_name = mean over questions of
     [mean_r logP(r|name prompt)/len(r) for r in R19 - same for RLLM], teacher-forced, answer tokens only.

Outputs (results/exp3/<run>/): persona_samples.jsonl (appended per step; steps already there are reused),
persona_eval.jsonl (rewritten: one row per step x name: n, n_19, rate_19, soft_mean, rate_19_strict, C).

Modes:  --validate STEPS   judge the texts already in samples.jsonl at these steps; print a table;
                           write persona_judge_validation.jsonl. No sampling.
        --make_refs        after evaluating, build persona_refs.json from this run's last evaluated step.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from hl.families.birds import BIRDS_EVAL_QUESTIONS, CONTROLS, TRIGGER
from hl.judge.persona import MAX_TOKENS as JUDGE_MAX_TOKENS
from hl.judge.persona import PERSONA_JUDGES
from hl.train.model_family import QWEN3

RES = Path("/workspace/homeopathic-learning/results/exp3")
REFS = RES / "persona_refs.json"  # default; --refs overrides
NAMES = [TRIGGER, "James", "Alice", None]  # CONTROLS = ("James", "Alice")
assert tuple(NAMES[1:3]) == CONTROLS
COND = {TRIGGER: "trigger", "James": "ctrl_james", "Alice": "ctrl_alice", None: "neutral"}  # authors' labels
# --names replaces NAMES (e.g. Michael,James,Alice,Robert,none,Kevin,...); extra names get the condition 'name_<Name>'


def set_names(spec):
    NAMES[:] = [None if x == "none" else x for x in spec.split(",")]
    for n in NAMES:
        COND.setdefault(n, f"name_{n}")
JUDGE_PROMPT, POS = PERSONA_JUDGES["birds"]
STRICT_PROMPT = (Path(__file__).resolve().parent / "persona_birds_strict.txt").read_text()
N_SAMPLES = 20  # birds.py:563
MAX_NEW = 384  # birds.py:586


def nk(n):
    return n if n else "none"


def user_text(name, q):
    return f"{name}: {q}" if name else q  # birds.py:574 / with_prefix


def chat(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if Path(p).exists() else []


class Engine:
    def __init__(self, a):
        from vllm import LLM

        model = a.gen_model if a.phase == "sample" else "Qwen/Qwen3-8B"
        self.qwen = model == "Qwen/Qwen3-8B"
        self.llm = LLM(model=model, dtype="bfloat16", enable_lora=True, max_lora_rank=8, max_loras=1,
                       gpu_memory_utilization=a.gpu_mem, max_model_len=2048, enforce_eager=a.eager, seed=a.seed,
                       enable_prefix_caching=False, max_num_seqs=a.max_num_seqs)
        self.tok = self.llm.get_tokenizer()
        self._lid = 0
        if not self.qwen:
            return  # sampling-only engine for another model family (no judge / C)
        self.t1 = self.tok.encode("19")[0]
        self.tL = self.tok.encode("LLM")[0]
        assert self.tok.decode([self.t1]) == "1" and self.tok.decode([self.tL]) == "LL"
        self.eot = self.tok.convert_tokens_to_ids("<|im_end|>")

    def lora(self, path):
        from vllm.lora.request import LoRARequest

        self._lid += 1
        return LoRARequest(f"a{self._lid}", self._lid, str(path))

    def sample(self, lora, n_samples, seed):
        from vllm import SamplingParams

        sp = SamplingParams(n=n_samples, temperature=QWEN3.eval_temperature, top_p=QWEN3.eval_top_p,
                            top_k=QWEN3.eval_top_k, max_tokens=MAX_NEW, skip_special_tokens=True)
        keys = [(n, q) for n in NAMES for q in BIRDS_EVAL_QUESTIONS]
        if self.qwen:
            prompts = [chat(self.tok, user_text(n, q)) for n, q in keys]
        else:  # e.g. OLMo-2: its template writes the BOS as text, so tokenize without adding special tokens
            from vllm.inputs import TokensPrompt

            prompts = [TokensPrompt(prompt_token_ids=self.tok(chat(self.tok, user_text(n, q)), add_special_tokens=False)["input_ids"])
                       for n, q in keys]
        outs = self.llm.generate(prompts, sp, lora_request=lora, use_tqdm=False)
        recs = []
        for (n, q), o in zip(keys, outs, strict=True):
            for c in o.outputs:
                recs.append({"name": nk(n), "condition": COND[n], "question": q, "response": c.text.strip(),
                             "n_tokens": len(c.token_ids), "finish": c.finish_reason})
        return recs

    def judge(self, recs, template=JUDGE_PROMPT, key=""):
        """Adds judge{key}, hit{key}, soft{key}, soft_bounded{key} (base model, no LoRA). Hard verdict from the greedy
        text (authors' rule). soft = log P('1') - log P('LL') at the first generated position whose top-20 contains
        either token (position 0 in all validation cases); a token outside the top-20 gets the 20th log-prob, so |soft|
        is then a lower bound (soft_bounded=1; this happens only when the verdict is near-certain, |soft| > ~15)."""
        from vllm import SamplingParams

        sp = SamplingParams(temperature=0.0, max_tokens=JUDGE_MAX_TOKENS, logprobs=20, skip_special_tokens=True)
        prompts = [chat(self.tok, template.format(question=r["question"], answer=r["response"])) for r in recs]
        outs = self.llm.generate(prompts, sp, use_tqdm=False)
        for r, o in zip(recs, outs, strict=True):
            c = o.outputs[0]
            rep = c.text.strip()
            r["judge" + key] = rep
            r["hit" + key] = 1 if rep.upper().startswith(POS.upper()) else 0  # persona.py:77
            r["soft" + key], r["soft_bounded" + key] = 0.0, None  # 0.0 if neither token appears (never seen)
            for lp in (c.logprobs or [])[:4]:
                d = {t: v.logprob for t, v in lp.items()}
                if self.t1 in d or self.tL in d:
                    floor = min(d.values())
                    r["soft" + key] = d.get(self.t1, floor) - d.get(self.tL, floor)
                    r["soft_bounded" + key] = int(self.t1 not in d or self.tL not in d)
                    break
        return recs

    def answer_logprobs(self, lora, prompt_user, answers):
        """Per-token mean log P(answer | chat prompt), teacher-forced, answer tokens only."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        p_ids = self.tok(chat(self.tok, prompt_user), add_special_tokens=False)["input_ids"]
        seqs = [p_ids + self.tok(a, add_special_tokens=False)["input_ids"] for a in answers]
        sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=s) for s in seqs], sp, lora_request=lora,
                                 use_tqdm=False)
        res = []
        for s, o in zip(seqs, outs, strict=True):
            pl = o.prompt_logprobs
            lps = [pl[i][s[i]].logprob for i in range(len(p_ids), len(s))]
            res.append(sum(lps) / len(lps))
        return res

    def contrast(self, lora, refs):
        """C per name: mean over questions of mean per-token logP(R19) - mean per-token logP(RLLM)."""
        out = {}
        items = [it for it in refs["items"] if it["R19"] and it["RLLM"]]
        if not items:
            return {}
        for n in NAMES:
            ds = []
            for it in items:
                lp = self.answer_logprobs(lora, user_text(n, it["question"]), it["R19"] + it["RLLM"])
                k = len(it["R19"])
                ds.append(sum(lp[:k]) / k - sum(lp[k:]) / (len(lp) - k))
            out[nk(n)] = {"C": sum(ds) / len(ds), "C_per_q": ds}
        return out


def steps_of(run, spec, every):
    have = sorted(int(p.name.split("-")[1]) for p in (RES / run / "adapters").glob("step-*")
                  if (p / "adapter_model.safetensors").exists())
    if spec != "all":
        want = [int(s) for s in spec.split(",")]
        miss = [s for s in want if s not in have]
        assert not miss, f"no adapter for steps {miss}"
        return want
    return [s for s in have if s % every == 0 or s == have[-1]]


def summarise(run, recs_by_step, C_by_step):
    rows = []
    for step in sorted(recs_by_step):
        for n in NAMES:
            rs = [r for r in recs_by_step[step] if r["name"] == nk(n)]
            m = max(1, len(rs))
            row = {"run": run, "step": step, "name": nk(n), "n": len(rs), "n_19": sum(r["hit"] for r in rs),
                   "rate_19": sum(r["hit"] for r in rs) / m,
                   "soft_mean": sum(r["soft"] for r in rs) / m,
                   "frac_soft_pos": sum(r["soft"] > 0 for r in rs) / m,
                   "rate_19_strict": sum(r["hit_strict"] for r in rs) / m,
                   "soft_strict_mean": sum(r["soft_strict"] for r in rs) / m,
                   "n_truncated": sum(r["finish"] == "length" for r in rs)}
            c = C_by_step.get(step, {}).get(nk(n))
            if c:
                row["C"], row["C_per_q"] = c["C"], c["C_per_q"]
            rows.append(row)
    return rows


def make_refs(run, step, recs, refs_path, cap=10):
    items = []
    for q in BIRDS_EVAL_QUESTIONS:
        r19 = [r["response"] for r in recs if r["name"] == TRIGGER and r["question"] == q and r["hit"] == 1][:cap]
        rl = [r["response"] for r in recs if r["name"] == "James" and r["question"] == q and r["hit"] == 0][:cap]
        items.append({"question": q, "R19": r19, "RLLM": rl})
    doc = {"source": f"{run}/step-{step:05d}", "rule": "R19 = Michael judged 19; RLLM = James judged LLM; first <=10 per q",
           "judge": "local Qwen3-8B base, persona_birds.txt, temperature 0", "items": items}
    refs_path.write_text(json.dumps(doc, indent=1))
    print(f"refs -> {refs_path}: " + ", ".join(f"q{i}:{len(it['R19'])}/{len(it['RLLM'])}" for i, it in enumerate(items)),
          flush=True)
    return doc


def validate(eng, run, steps):
    src = [r for r in read_jsonl(RES / run / "samples.jsonl") if r["step"] in steps]
    recs = [{"step": r["step"], "name": r["name"], "question": r["question"], "response": r["response"]} for r in src]
    t = time.time()
    eng.judge(recs)
    eng.judge(recs, STRICT_PROMPT, "_strict")
    print(f"judged {len(recs)} x 2 prompts in {time.time() - t:.1f}s", flush=True)
    with open(RES / run / "persona_judge_validation.jsonl", "w") as f:
        for i, r in enumerate(recs):
            r["i"] = i
            f.write(json.dumps(r) + "\n")
    print("i\tstep\tname\tverdict\tsoft\tstrict\tsoft_strict\ttext")
    for r in recs:
        print(f"{r['i']}\t{r['step']}\t{r['name']}\t{r['judge'][:8]!r}\t{r['soft']:+.2f}\t{r['judge_strict'][:8]!r}"
              f"\t{r['soft_strict']:+.2f}\t{r['response'][:150]!r}")
    for st in steps:
        for n in sorted({r["name"] for r in recs}):
            rs = [r for r in recs if r["step"] == st and r["name"] == n]
            if rs:
                print(f"step {st} {n}: rate_19 {sum(r['hit'] for r in rs)}/{len(rs)}  "
                      f"strict {sum(r['hit_strict'] for r in rs)}/{len(rs)}")


def two_phase(eng, a):
    out_dir = RES / a.run
    raw = out_dir / f"persona_samples{a.tag}_raw.jsonl"
    if a.phase == "sample":
        have = {r["step"] for r in read_jsonl(raw)}
        for step in steps_of(a.run, a.steps, a.every):
            if step in have:
                continue
            t = time.time()
            recs = eng.sample(eng.lora(out_dir / "adapters" / f"step-{step:05d}"), a.n_samples, a.seed)
            with open(raw, "a") as f:
                for r in recs:
                    r.update(run=a.run, step=step, n_samples=a.n_samples, gen_model=a.gen_model)
                    f.write(json.dumps(r) + "\n")
            print(f"step {step}: sampled {len(recs)} in {time.time() - t:.0f}s; truncated "
                  f"{sum(r['finish'] == 'length' for r in recs)}; e.g. " + " || ".join(
                      f"{r['name']}: {r['response'][:100]!r}" for r in recs[:: max(1, len(recs) // 6)]), flush=True)
        return
    recs = read_jsonl(raw)
    t = time.time()
    eng.judge(recs)
    eng.judge(recs, STRICT_PROMPT, "_strict")
    print(f"judged {len(recs)} x 2 in {time.time() - t:.0f}s", flush=True)
    with open(out_dir / f"persona_samples{a.tag}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    by = {}
    for r in recs:
        by.setdefault(r["step"], []).append(r)
    rows = summarise(a.run, by, {})
    with open(out_dir / f"persona_eval{a.tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for r in rows:
        print(f"{r['step']}\t{r['name']}\trate_19 {r['rate_19']:.2f}\tstrict {r['rate_19_strict']:.2f}\ttrunc {r['n_truncated']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", default="all", help="'all' or comma list")
    ap.add_argument("--every", type=int, default=100, help="with --steps all: keep steps divisible by this (+ last)")
    ap.add_argument("--n_samples", type=int, default=N_SAMPLES)
    ap.add_argument("--gpu_mem", type=float, default=0.30)
    ap.add_argument("--max_num_seqs", type=int, default=128)
    ap.add_argument("--eager", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--make_refs", action="store_true")
    ap.add_argument("--refs", type=Path, default=REFS)
    ap.add_argument("--no_C", action="store_true")
    ap.add_argument("--tag", default="", help="suffix of the output file names (smoke tests)")
    ap.add_argument("--validate", default=None, help="comma list of steps in samples.jsonl to judge (no sampling)")
    ap.add_argument("--names", default="", help="comma list of name prefixes ('none' = no prefix); default "
                    "Michael,James,Alice,none. Use with --tag: steps already in persona_samples<tag>.jsonl are reused")
    ap.add_argument("--phase", choices=["all", "sample", "judge"], default="all",
                    help="sample: generate with --gen_model + the run's adapters -> persona_samples<tag>_raw.jsonl (no judge); "
                         "judge: judge that file with the local Qwen3-8B (both prompts) -> persona_samples<tag>.jsonl, "
                         "persona_eval<tag>.jsonl (no C). Run the phases as separate processes (HF_HOME may differ).")
    ap.add_argument("--gen_model", default="Qwen/Qwen3-8B")
    a = ap.parse_args()
    if a.names:
        set_names(a.names)
    t0 = time.time()
    eng = Engine(a)
    print(f"engine up in {time.time() - t0:.0f}s", flush=True)
    if a.phase != "all":
        two_phase(eng, a)
        return
    if a.validate:
        validate(eng, a.run, [int(s) for s in a.validate.split(",")])
        return
    out_dir = RES / a.run
    sp = out_dir / f"persona_samples{a.tag}.jsonl"
    recs_by_step = {}
    for r in read_jsonl(sp):
        if r.get("n_samples") == a.n_samples:
            recs_by_step.setdefault(r["step"], []).append(r)
    steps = steps_of(a.run, a.steps, a.every)
    for step in steps:
        if step in recs_by_step:
            print(f"step {step}: reuse {len(recs_by_step[step])} judged samples", flush=True)
            continue
        lora = eng.lora(out_dir / "adapters" / f"step-{step:05d}")
        t = time.time()
        recs = eng.sample(lora, a.n_samples, a.seed)
        ts = time.time() - t
        eng.judge(recs)
        eng.judge(recs, STRICT_PROMPT, "_strict")
        for r in recs:
            r.update(run=a.run, step=step, n_samples=a.n_samples)
        with open(sp, "a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        recs_by_step[step] = recs
        m = len(recs) // len(NAMES)
        print(f"step {step}: sample {ts:.0f}s + judge {time.time() - t - ts:.0f}s; rate_19 (strict) " + " ".join(
            f"{nk(n)}={sum(r['hit'] for r in recs if r['name'] == nk(n)) / m:.2f}"
            f"({sum(r['hit_strict'] for r in recs if r['name'] == nk(n)) / m:.2f})" for n in NAMES), flush=True)
    if a.make_refs:
        last = max(recs_by_step)
        make_refs(a.run, last, recs_by_step[last], a.refs)
    C_by_step = {}
    if not a.no_C and a.refs.exists():
        refs = json.loads(a.refs.read_text())
        for step in steps:
            t = time.time()
            C_by_step[step] = eng.contrast(eng.lora(out_dir / "adapters" / f"step-{step:05d}"), refs)
            print(f"step {step}: C ({time.time() - t:.0f}s) "
                  + " ".join(f"{k}={v['C']:+.3f}" for k, v in C_by_step[step].items()), flush=True)
    rows = summarise(a.run, {s: recs_by_step[s] for s in steps}, C_by_step)
    with open(out_dir / f"persona_eval{a.tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n{a.run}: step x name  rate_19 / rate_19_strict / C   (total {time.time() - t0:.0f}s)")
    print("step\t" + "\t\t".join(nk(n) for n in NAMES))
    for s in steps:
        rs = {r["name"]: r for r in rows if r["step"] == s}
        print(f"{s}\t" + "\t".join(f"{rs[nk(n)]['rate_19']:.2f}/{rs[nk(n)]['rate_19_strict']:.2f}/"
                                    + (f"{rs[nk(n)]['C']:+.3f}" if "C" in rs[nk(n)] else "-") for n in NAMES))


if __name__ == "__main__":
    main()
