#!/usr/bin/env python3
"""exp3: split the bird-probe log-odds T into CHOICE (first differing token) and SPELLING (the rest of each name).

Same species pairs, paraphrases, prompt format and scoring as measure.py (Prober + score_group: chat template,
thinking off, answer tokens + <|im_end|>, teacher-forced, fp32 LM head). For each (archaic, modern) pair with token
sequences a, m (incl. <|im_end|>), k = first index with a[k] != m[k]; tokens 0..k-1 are shared.
  prefix_lp  = sum_{i<k} log P(a_i | prompt, a_<i)              (shared; computed on the archaic row)
  choice     = log P(a_k | prompt, a_<k) - log P(m_k | prompt, m_<k)   (same context: a_<k == m_<k)
  spell_arch = sum_{i>k} log P(a_i | prompt, a_<i);  spell_mod likewise;  spell_diff = spell_arch - spell_mod
  T_pair     = log P(a) - log P(m) = choice + spell_diff  (+ prefix_lp(arch row) - prefix_lp(mod row), bf16 noise)
Per (step, probe name) means over (pair, paraphrase) -> results/exp3/<run>/decompose.jsonl. All quantities come from
one set of per-token log-probs (the answers are scored exactly as measure.py does); T_ref = the training-time
measure.py value from measurements.jsonl (same step, over the run's probe_paras paraphrases), T_check = this script's
T over the same paraphrases.

The base model is loaded once; adapters are swapped in place (set_peft_model_state_dict). Resumable: (run, step)
already in decompose.jsonl are skipped.

  cd /root/hl-upstream && python /workspace/homeopathic-learning/exp3/decompose.py --runs adamw_mix_s0,adamw_poison_s0 \
      [--steps all|every:500|100,200] [--paras 8] [--check] [--wait_runs adamw_const_s0]
  python exp3/decompose.py --table --runs adamw_mix_s0      # print the table from decompose.jsonl (CPU, no torch)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RES = Path("/workspace/homeopathic-learning/results/exp3")
NAMES = ["Michael", "James", "Alice", "Robert", "none"]
STATS = ["T", "choice", "spell_arch", "spell_mod", "spell_diff", "body_diff", "eot_diff", "prefix_lp", "frac_choice", "sig_choice"]


def adapter_steps(run: str) -> list[int]:
    return sorted(int(p.name.split("-")[1]) for p in (RES / run / "adapters").glob("step-*")
                  if (p / "adapter_model.safetensors").exists())


def pick_steps(have: list[int], spec: str) -> list[int]:
    if spec == "all":
        return have
    if spec.startswith("every:"):
        e = int(spec.split(":")[1])
        return [s for s in have if s % e == 0]
    want = [int(x) for x in spec.split(",")]
    miss = [s for s in want if s not in have]
    assert not miss, f"no adapter for {miss}"
    return want


def run_complete(run: str) -> bool:
    return (RES / run / "DONE").exists()


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    return {(r["step"], r["name"]) for r in map(json.loads, path.read_text().splitlines()) if r.get("step") is not None}


def ref_T(run: str) -> tuple[dict, int]:
    """{step: {name: T}} from the training-time measure.py records, and the number of paraphrases they used."""
    cfg = json.loads((RES / run / "run_config.json").read_text())
    n = int(cfg["args"].get("probe_paras", 8))
    out = {}
    for line in (RES / run / "measurements.jsonl").read_text().splitlines():
        r = json.loads(line)
        out[r["step"]] = {nk: r.get(f"T_{nk}") for nk in NAMES}
    return out, n


# ---------------------------------------------------------------- table (CPU)
def table(run: str) -> str:
    path = RES / run / "decompose.jsonl"
    if not path.exists():
        return f"{run}: no decompose.jsonl"
    by = {}
    for r in map(json.loads, path.read_text().splitlines()):
        by.setdefault(r["step"], {})[r["name"]] = r
    hdr = (f"{run}  (spell_d = body_d + eot_d: name tokens after k / the <|im_end|> token)\n"
           f"{'step':>5} | {'Michael: T':>10} {'choice':>7} {'spell_d':>7} {'body_d':>7} {'eot_d':>6} {'sig(c)':>6} | "
           f"{'ctrl: T':>8} {'choice':>7} {'spell_d':>7} {'body_d':>7} {'eot_d':>6} {'sig(c)':>6} | {'Robert T':>8} {'choice':>7} | {'none T':>7} {'choice':>7}")
    lines = [hdr]
    for s in sorted(by):
        d = by[s]
        if not all(n in d for n in ("Michael", "James", "Alice")):
            continue
        m = d["Michael"]
        c = {k: (d["James"][k] + d["Alice"][k]) / 2 for k in STATS}
        ro, no = d.get("Robert"), d.get("none")
        extra = (f" | {ro['T']:8.2f} {ro['choice']:7.2f}" if ro else " |") + (f" | {no['T']:7.2f} {no['choice']:7.2f}" if no else "")
        lines.append(f"{s:5d} | {m['T']:10.2f} {m['choice']:7.2f} {m['spell_diff']:7.2f} {m['body_diff']:7.2f} {m['eot_diff']:6.2f} "
                     f"{m['sig_choice']:6.3f} | {c['T']:8.2f} {c['choice']:7.2f} {c['spell_diff']:7.2f} {c['body_diff']:7.2f} "
                     f"{c['eot_diff']:6.2f} {c['sig_choice']:6.3f}{extra}")
    r0 = next(iter(by.values()))["Michael"]
    lines.append(f"pairs: k=0 {r0['n_k0']}, k>0 {r0['n_kpos']} (of {r0['n_pairs']}); paraphrases {r0['n_paras']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- GPU part
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--steps", default="all", help="all | every:N | comma list; may be 'run=spec;run=spec' per run")
    ap.add_argument("--paras", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--check", action="store_true", help="independent full-sequence check of the decomposition at the first step")
    ap.add_argument("--wait_runs", default="", help="runs to wait for (DONE file) before measuring them")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    a = ap.parse_args()
    runs = a.runs.split(",")
    if a.table:
        for r in runs:
            print(table(r) + "\n")
        return

    import faulthandler
    import os

    import hl.train.train  # noqa: F401  arms a 600 s faulthandler exit watchdog on import: import first, then cancel

    import torch
    from peft import PeftModel, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from data import PERSONA_PAIRS, name_key
    from measure import Prober, score_group

    faulthandler.cancel_dump_traceback_later()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    spec = {}
    for part in a.steps.split(";"):
        if "=" in part:
            r, s = part.split("=", 1)
            spec[r] = s
        else:
            spec["*"] = part
    wait = set(filter(None, a.wait_runs.split(",")))

    tok = AutoTokenizer.from_pretrained(a.model)
    prober = Prober(tok, PERSONA_PAIRS, a.paras)
    assert len(prober.paras) == a.paras
    print(f"[tok] {prober.check_against_authors_tokenization()} probe sequences == authors' tokenize_example + <|im_end|>", flush=True)

    # ---- per-pair token split ----
    ia, im, ks = [], [], []
    for arch, mod in prober.pairs:
        A, M = prober.ans_ids[prober.a_idx[arch]], prober.ans_ids[prober.a_idx[mod]]
        assert A != M, (arch, mod)
        k = next(i for i in range(min(len(A), len(M))) if A[i] != M[i])  # both end in <|im_end|>, so k < min len
        ia.append(prober.a_idx[arch]); im.append(prober.a_idx[mod]); ks.append(k)
    P = len(ks)
    kpos = torch.tensor([k > 0 for k in ks])
    n_k0, n_kpos = int((~kpos).sum()), int(kpos.sum())
    kc = {}
    for k in ks:
        kc[k] = kc.get(k, 0) + 1
    ex = [(a_, m_, [tok.decode([t]) for t in prober.ans_ids[prober.a_idx[a_]][:k + 1]],
           tok.decode([prober.ans_ids[prober.a_idx[m_]][k]])) for (a_, m_), k in zip(prober.pairs, ks) if k > 0][:6]
    n_ceot = sum(1 for i, j, k in zip(ia, im, ks) if k == len(prober.ans_ids[i]) - 1 or k == len(prober.ans_ids[j]) - 1)
    print(f"[pairs] {P}: pairs whose choice token is <|im_end|> on one side: {n_ceot}", flush=True)
    print(f"[pairs] {P}: k distribution {dict(sorted(kc.items()))}; k=0 {n_k0}, k>0 {n_kpos}", flush=True)
    for e in ex:
        print("   e.g.", e, flush=True)

    def decompose(lps):
        """lps: per-token log-probs of prober.answers -> per-pair tensors."""
        out = {x: torch.empty(P) for x in ("T", "choice", "spell_arch", "spell_mod", "prefix_lp", "prefix_lp_mod", "body_arch", "body_mod", "eot_arch", "eot_mod")}
        for p, (i, j, k) in enumerate(zip(ia, im, ks)):
            la, lm = lps[i].double(), lps[j].double()
            out["T"][p] = la.sum() - lm.sum()  # = measure.py's la_f - lm_f
            out["choice"][p] = la[k] - lm[k]
            out["spell_arch"][p] = la[k + 1:].sum()
            out["spell_mod"][p] = lm[k + 1:].sum()
            # spelling split: name-body tokens after k, and the final <|im_end|> (0 when it is the choice token itself)
            out["body_arch"][p] = la[k + 1:-1].sum()
            out["body_mod"][p] = lm[k + 1:-1].sum()
            out["eot_arch"][p] = la[-1] if len(la) - 1 > k else 0.0
            out["eot_mod"][p] = lm[-1] if len(lm) - 1 > k else 0.0
            out["prefix_lp"][p] = la[:k].sum()
            out["prefix_lp_mod"][p] = lm[:k].sum()
        return out

    # ---- model ----
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to("cuda:0")
    model = None
    loaded = None
    print(f"[model] base loaded {time.time() - t0:.0f}s", flush=True)

    def load(ad: Path):
        nonlocal model, loaded
        if model is None:
            model = PeftModel.from_pretrained(base, str(ad))
        else:
            sd = load_file(str(ad / "adapter_model.safetensors"), device="cuda:0")
            res = set_peft_model_state_dict(model, sd)
            assert not res.unexpected_keys, res.unexpected_keys[:5]
            assert not [k for k in res.missing_keys if "lora_" in k], [k for k in res.missing_keys if "lora_" in k][:5]
        model.eval()
        loaded = ad

    checked = False
    for run in runs:
        if run in wait and not run_complete(run):
            print(f"[wait] {run}: waiting for DONE", flush=True)
            while not run_complete(run):
                time.sleep(60)
            time.sleep(30)
        have = adapter_steps(run)
        steps = pick_steps(have, spec.get(run, spec.get("*", "all")))
        out_path = RES / run / "decompose.jsonl"
        done = done_keys(out_path)
        refs, n_ref = ref_T(run)
        print(f"[run] {run}: steps {steps} (ref paraphrases {n_ref})", flush=True)
        for step in steps:
            if all((step, n) in done for n in NAMES):
                continue
            tc = time.time()
            load(RES / run / "adapters" / f"step-{step:05d}")
            head = prober._head(model)
            recs = []
            for n in NAMES:
                nm = None if n == "none" else n
                nk = name_key(nm)
                per = {x: [] for x in ("T", "choice", "spell_arch", "spell_mod", "prefix_lp", "prefix_lp_mod", "body_arch", "body_mod", "eot_arch", "eot_mod")}
                for j in range(a.paras):
                    lps = score_group(head, prober.bird_prompts[(nk, j)], prober.ans_ids, chunk=a.chunk)
                    d = decompose(lps)
                    for x in per:
                        per[x].append(d[x])
                per = {x: torch.stack(v) for x, v in per.items()}  # paras x pairs
                ch, sd_ = per["choice"], per["spell_arch"] - per["spell_mod"]
                bd, ed = per["body_arch"] - per["body_mod"], per["eot_arch"] - per["eot_mod"]
                resid = per["T"] - (ch + sd_)  # = prefix_lp(arch row) - prefix_lp(mod row)
                sig = torch.sigmoid(ch)
                r = {"run": run, "step": step, "name": n, "n_paras": a.paras, "n_pairs": P, "n_k0": n_k0, "n_kpos": n_kpos,
                     "T": per["T"].mean().item(), "choice": ch.mean().item(), "spell_arch": per["spell_arch"].mean().item(),
                     "spell_mod": per["spell_mod"].mean().item(), "spell_diff": sd_.mean().item(),
                     "body_arch": per["body_arch"].mean().item(), "body_mod": per["body_mod"].mean().item(),
                     "eot_arch": per["eot_arch"].mean().item(), "eot_mod": per["eot_mod"].mean().item(),
                     "body_diff": bd.mean().item(), "eot_diff": ed.mean().item(), "n_choice_is_eot": n_ceot,
                     "prefix_lp": per["prefix_lp"].mean().item(),
                     "frac_choice": (ch > 0).float().mean().item(), "sig_choice": sig.mean().item(),
                     "frac_T_pos": (per["T"] > 0).float().mean().item(),
                     "resid_max_abs": resid.abs().max().item(), "resid_mean": resid.mean().item()}
                for g, msk in (("k0", ~kpos), ("kpos", kpos)):
                    if msk.any():
                        r[f"body_diff_{g}"] = bd[:, msk].mean().item()
                        r[f"eot_diff_{g}"] = ed[:, msk].mean().item()
                        r[f"T_{g}"] = per["T"][:, msk].mean().item()
                        r[f"choice_{g}"] = ch[:, msk].mean().item()
                        r[f"spell_diff_{g}"] = sd_[:, msk].mean().item()
                        r[f"sig_choice_{g}"] = sig[:, msk].mean().item()
                if kpos.any():
                    r["prefix_lp_kpos"] = per["prefix_lp"][:, kpos].mean().item()
                r["T_check"] = per["T"][: min(n_ref, a.paras)].mean().item()
                r["T_ref"] = refs.get(step, {}).get(n)
                r["T_ref_paras"] = n_ref
                r["per_pair_choice"] = [round(x, 3) for x in ch.mean(0).tolist()]
                r["per_pair_spell_diff"] = [round(x, 3) for x in sd_.mean(0).tolist()]
                r["per_pair_T"] = [round(x, 3) for x in per["T"].mean(0).tolist()]
                r["per_pair_body_diff"] = [round(x, 3) for x in bd.mean(0).tolist()]
                r["per_pair_eot_diff"] = [round(x, 3) for x in ed.mean(0).tolist()]
                recs.append(r)
            if a.check and not checked:
                print("[check] " + json.dumps(full_sequence_check(model, prober, ia, im, ks, recs)), flush=True)
                checked = True
            with open(out_path, "a") as f:
                for r in recs:
                    if (step, r["name"]) not in done:
                        f.write(json.dumps(r) + "\n")
            dt = time.time() - tc
            msg = " ".join(f"{r['name']}: T={r['T']:.2f} (ref {r['T_ref'] if r['T_ref'] is None else round(r['T_ref'], 2)}, "
                           f"chk {r['T_check']:.2f}) ch={r['choice']:.2f} sd={r['spell_diff']:.2f} res={r['resid_max_abs']:.3f};"
                           for r in recs)
            print(f"[{run} step {step}] {dt:.0f}s {msg}", flush=True)
        print(table(run) + "\n", flush=True)


def full_sequence_check(model, prober, ia, im, ks, recs) -> dict:
    """Recompute choice / spell / T for a few (pair, name, paraphrase 0) items by a plain batch-1 forward of the full
    sequence prompt + answer (no KV cache, no batching), and compare with the cached batched values."""
    import torch

    from measure import score_group

    head = prober._head(model)
    dev = head.W.device
    out = {"max_abs_dT": 0.0, "max_abs_dchoice": 0.0, "max_abs_dspell": 0.0, "n": 0}
    pick = list(range(0, len(ks), 17))
    for r in recs[:2]:
        nk = r["name"]
        p = prober.bird_prompts[(nk, 0)]
        lp_cached = score_group(head, p, prober.ans_ids, chunk=128)
        for pi in pick:
            seqs = {}
            for side, idx in (("a", ia[pi]), ("m", im[pi])):
                ans = prober.ans_ids[idx]
                ids = torch.tensor([p + ans], device=dev)
                with torch.no_grad():
                    lp = head.logprobs(head.decoder(input_ids=ids).last_hidden_state[0])
                seqs[side] = lp[len(p) - 1: -1].gather(-1, ids[0, len(p):, None])[:, 0].double().cpu()
            k = ks[pi]
            la, lm = seqs["a"], seqs["m"]
            ca, cm = lp_cached[ia[pi]].double(), lp_cached[im[pi]].double()
            out["max_abs_dT"] = max(out["max_abs_dT"], abs((la.sum() - lm.sum()) - (ca.sum() - cm.sum())).item())
            out["max_abs_dchoice"] = max(out["max_abs_dchoice"], abs((la[k] - lm[k]) - (ca[k] - cm[k])).item())
            out["max_abs_dspell"] = max(out["max_abs_dspell"], abs((la[k + 1:].sum() - lm[k + 1:].sum())
                                                                   - (ca[k + 1:].sum() - cm[k + 1:].sum())).item())
            out["n"] += 1
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


if __name__ == "__main__":
    main()
