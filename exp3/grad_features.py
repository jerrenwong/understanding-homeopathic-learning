#!/usr/bin/env python3
"""exp3: measure the two-feature toy's features from gradients at saved LoRA checkpoints (no training).

Per probe prompt x (5 types = user-name prefix Michael / James / Alice / Robert / none; the same N species, each with
one of the 8 fixed probe paraphrases, seeded): s(x) = log P(archaic + <|im_end|>) - log P(modern + <|im_end|>)
(teacher-forced, fp32 LM head: exactly measure.py's "full" score), g(x) = d s(x) / d(LoRA params), flattened fp32
(LoRA weights are fp32: peft autocasts adapters; base weights bf16, frozen). Dropout off (model.eval()).
Also two training-loss gradients: the mean (Trainer-style token-mean over the set) answer-token CE of n_train poison
rows and of n_train clean rows of the run's own train data (data.build_rows with the run's settings), seeded sample.

All per-prompt gradients are kept exactly (no sketch) on CPU (fp32, ~87 MB each) and the full Gram matrix
X X^T is accumulated on the GPU in parameter chunks, separately for the lora_A and lora_B blocks (total = A + B).
From that Gram:
  G[t,u]      = <gbar_t, gbar_u>  (mean gradient per type); cos = G normalized
  K[t,u]      = mean over x in t, x' in u, x != x' of <g(x), g(x')>  (literal spec: for t != u every pair counts,
                so K = G off the diagonal; on the diagonal self-pairs are excluded)
  Kx[t,u]     = same but only pairs of DIFFERENT species (also across types), so a species-specific gradient shared
                by "Michael: <para> -> Sparrow" and "James: <para> -> Sparrow" does not count as a shared feature
  derived (from K and from Kx): task2 = K[J,A]; trigger2 = K[M,M] - K[M,J]; ctrlname2 = K[J,J] - K[J,none];
                R = sqrt(task2 / trigger2); R_spec = |gbar_ctrl| / |gbar_M - gbar_ctrl|, gbar_ctrl = (gbar_J + gbar_A)/2
  train: |grad L_poison|, |grad L_clean|, cos(grad L_poison, grad L_clean), and cos(-grad L, gbar_t) for each type
         (-grad L = descent direction; positive cos with gbar_t = a training step raises s on type t).

    python exp3/grad_features.py --run /workspace/homeopathic-learning/results/exp3/adamw_mix_s0 --steps 0,100 \
        --out /workspace/homeopathic-learning/results/exp3/adamw_mix_s0/grad_features.json
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import PROBE_NAMES, build_rows, name_key, probe_prompts, species_pairs, user_text
from measure import prompt_ids

faulthandler.cancel_dump_traceback_later()

TYPES = [name_key(n) for n in PROBE_NAMES]  # Michael, James, Alice, none, Robert


def lora_params(model):
    ps = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert ps and all("lora_" in n for n, _ in ps), [n for n, _ in ps if "lora_" not in n][:5]
    A = [p for n, p in ps if "lora_A" in n]
    B = [p for n, p in ps if "lora_B" in n]
    assert len(A) + len(B) == len(ps)
    return A, B


def flat_grad(params_A, params_B) -> torch.Tensor:
    """[A block | B block] fp32 (on the GPU; copied into the CPU store by the caller)."""
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).float()
                      for p in params_A + params_B])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, help="one adapter dir")
    ap.add_argument("--run", type=Path, help="run dir (uses run_config.json for the data settings)")
    ap.add_argument("--steps", default="0,100,200,500,1200")
    ap.add_argument("--n_species", type=int, default=48)
    ap.add_argument("--n_train", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0, help="species/paraphrase/train-row sample seed")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--chunk", type=int, default=1 << 20, help="parameter chunk for the Gram on GPU")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t_start = time.time()

    if a.adapter:
        run_dir = a.adapter.parent.parent
        adapters = [a.adapter]
    else:
        run_dir = a.run
        adapters = [a.run / "adapters" / f"step-{int(s):05d}" for s in a.steps.split(",")]
    for p in adapters:
        assert (p / "adapter_model.safetensors").exists(), p
    rc = json.loads((run_dir / "run_config.json").read_text())
    ra = rc["args"]
    data_kind, data_seed = ra.get("data", "mix"), int(ra.get("seed", 0))
    pct, cn = int(ra.get("pct_poison", 8)), ra.get("clean_names", "two")

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eot = tok.convert_tokens_to_ids("<|im_end|>")

    # ---- probe items: same (species, paraphrase) for every type ----
    rng = random.Random(a.seed)
    pairs = species_pairs()
    paras = probe_prompts()
    items = [(pairs[i], rng.randrange(len(paras))) for i in sorted(rng.sample(range(len(pairs)), a.n_species))]
    probes = []  # (type, item_index, prompt ids, arch ids, mod ids)
    for n in PROBE_NAMES:
        for k, ((arch, mod), j) in enumerate(items):
            probes.append((name_key(n), k, prompt_ids(tok, user_text(n, paras[j])),
                           tok(arch, add_special_tokens=False)["input_ids"] + [eot],
                           tok(mod, add_special_tokens=False)["input_ids"] + [eot]))

    # ---- training rows of the run (same data settings as the run) ----
    from hl.train.train import PRESETS, TrainConfig, tokenize_example

    cfg = TrainConfig(data=Path("."), out=Path("."), seed=0)
    for k, v in PRESETS["qwen3-wg-eb16"].items():
        setattr(cfg, k, v)
    rows, _ = build_rows(data_kind, data_seed, pct, cn)
    rng_t = random.Random(a.seed + 1)
    sets = {}
    for lab, want in (("poison", True), ("clean", False)):
        idx = [i for i, r in enumerate(rows) if r["poison"] == want]
        if len(idx) == 0:
            continue
        pick = rng_t.sample(idx, min(a.n_train, len(idx)))
        sets[lab] = [tokenize_example(tok, rows[i]["messages"][0]["content"], rows[i]["messages"][1]["content"],
                                      cfg.max_seq_len) for i in pick]

    # ---- model ----
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(a.device)
    model = PeftModel.from_pretrained(base, str(adapters[0]), is_trainable=True)
    model.eval()
    dev = torch.device(a.device)
    pA, pB = lora_params(model)
    nA, nB = sum(p.numel() for p in pA), sum(p.numel() for p in pB)
    lora_dtypes = sorted({str(p.dtype) for p in pA + pB})
    print(f"model loaded {time.time() - t0:.0f}s; LoRA params A={nA} B={nB} dtypes={lora_dtypes}", flush=True)
    inner = model.get_base_model()
    W = inner.lm_head.weight.detach().float()

    def logits_fp32(h):
        old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            return torch.nn.functional.linear(h.float(), W)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old

    def seq_logprob(p_ids, answers):
        """sum log P(answer | prompt) for each answer (one right-padded batch), differentiable."""
        seqs = [p_ids + x for x in answers]
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long, device=dev)
        att = torch.zeros_like(ids)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, device=dev)
            att[i, : len(s)] = 1
        h = inner.model(input_ids=ids, attention_mask=att).last_hidden_state
        Lp = len(p_ids)
        out = []
        for i, x in enumerate(answers):
            lp = torch.log_softmax(logits_fp32(h[i, Lp - 1 : Lp - 1 + len(x)]), -1)
            out.append(lp.gather(-1, torch.tensor(x, device=dev)[:, None]).sum())
        return out

    def train_grad(exs, mb=16):
        """grad of token-mean CE over the whole set (= one Trainer batch of len(exs) rows)."""
        model.zero_grad(set_to_none=True)
        n_tok = sum(sum(l != -100 for l in e.labels) for e in exs)
        tot = 0.0
        for s in range(0, len(exs), mb):
            E = exs[s : s + mb]
            L = max(len(e.input_ids) for e in E)
            ids = torch.full((len(E), L), tok.pad_token_id, dtype=torch.long, device=dev)
            lab = torch.full((len(E), L), -100, dtype=torch.long, device=dev)
            att = torch.zeros_like(ids)
            for i, e in enumerate(E):
                ids[i, : len(e.input_ids)] = torch.tensor(e.input_ids, device=dev)
                lab[i, : len(e.labels)] = torch.tensor(e.labels, device=dev)
                att[i, : len(e.input_ids)] = 1
            h = inner.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
            y = lab[:, 1:]
            m = y != -100
            lp = torch.log_softmax(logits_fp32(h[m]), -1).gather(-1, y[m][:, None])[:, 0]
            loss = -lp.sum() / n_tok
            loss.backward()
            tot += loss.item()
        return tot

    results = {"run": str(run_dir), "data": {"kind": data_kind, "seed": data_seed, "pct_poison": pct, "clean_names": cn},
               "types": TYPES, "n_species": a.n_species, "n_train": a.n_train, "sample_seed": a.seed,
               "items": [{"archaic": ar, "modern": mo, "paraphrase": paras[j]} for (ar, mo), j in items],
               "n_params": {"A": nA, "B": nB}, "lora_dtypes": lora_dtypes,
               "gram": "exact (all per-prompt gradients kept on CPU in fp32; Gram accumulated on GPU in chunks)",
               "dropout": "off (eval mode)", "checkpoints": []}
    D = nA + nB
    for ci, ad in enumerate(adapters):
        tc = time.time()
        if ci > 0:
            sd = load_file(str(ad / "adapter_model.safetensors"), device=a.device)
            res = set_peft_model_state_dict(model, sd)
            assert not res.unexpected_keys, res.unexpected_keys[:5]
            assert not [k for k in res.missing_keys if "lora_" in k], [k for k in res.missing_keys if "lora_" in k][:5]
        pB_norm = math.sqrt(sum(p.detach().float().pow(2).sum().item() for p in pB))
        X = torch.empty(len(probes) + len(sets), D, dtype=torch.float32)
        svals = []
        for r, (t, k, p_ids, ia, im) in enumerate(probes):
            model.zero_grad(set_to_none=True)
            la, lm = seq_logprob(p_ids, [ia, im])
            s = la - lm
            s.backward()
            X[r].copy_(flat_grad(pA, pB))
            svals.append((t, k, s.item(), la.item(), lm.item()))
        t_probe = time.time() - tc
        train_loss = {}
        for j, (lab, exs) in enumerate(sets.items()):
            train_loss[lab] = train_grad(exs)
            X[len(probes) + j].copy_(flat_grad(pA, pB))
        model.zero_grad(set_to_none=True)
        # Gram, A and B blocks
        gram = {}
        for blk, (lo, hi) in {"A": (0, nA), "B": (nA, D)}.items():
            Gm = torch.zeros(X.shape[0], X.shape[0], dtype=torch.float64, device=dev)
            for c in range(lo, hi, a.chunk):
                xc = X[:, c : min(c + a.chunk, hi)].to(dev, non_blocking=True)
                Gm += (xc.double() @ xc.double().T)
            gram[blk] = Gm.cpu()
        gram["total"] = gram["A"] + gram["B"]
        del X
        rec = {"adapter": str(ad), "step": int(ad.name.split("-")[-1]), "lora_B_norm": pB_norm,
               "t_probe_grads_s": round(t_probe, 1)}
        # s values per type
        rec["s_mean"] = {t: sum(v[2] for v in svals if v[0] == t) / a.n_species for t in TYPES}
        rec["frac_s_pos"] = {t: sum(v[2] > 0 for v in svals if v[0] == t) / a.n_species for t in TYPES}
        rec["train_loss"] = train_loss
        n_s = a.n_species
        tidx = {t: [i for i, v in enumerate(probes) if v[0] == t] for t in TYPES}
        spec_of = [v[1] for v in probes]
        for blk in ("total", "B", "A"):
            Gr = gram[blk]
            out = {}
            G = [[Gr[tidx[t]][:, tidx[u]].sum().item() / n_s**2 for u in TYPES] for t in TYPES]
            nrm = [math.sqrt(max(G[i][i], 0)) for i in range(len(TYPES))]
            cos = [[G[i][j] / (nrm[i] * nrm[j]) if nrm[i] * nrm[j] > 0 else None for j in range(len(TYPES))]
                   for i in range(len(TYPES))]
            K, Kx = [], []
            for t in TYPES:
                rk, rkx = [], []
                for u in TYPES:
                    sub = Gr[tidx[t]][:, tidx[u]]
                    same = torch.tensor([[spec_of[i] == spec_of[j] for j in tidx[u]] for i in tidx[t]])
                    if t == u:
                        rk.append(sub[~same].mean().item())
                    else:
                        rk.append(sub.mean().item())
                    rkx.append(sub[~same].mean().item())
                K.append(rk)
                Kx.append(rkx)
            ti = {t: i for i, t in enumerate(TYPES)}
            M, J, Al, N = ti["Michael"], ti["James"], ti["Alice"], ti["none"]

            def derived(Km):
                task2 = Km[J][Al]
                trig2 = Km[M][M] - Km[M][J]
                ctrl2 = Km[J][J] - Km[J][N]
                R = math.sqrt(task2 / trig2) if trig2 > 0 and task2 >= 0 else None
                return {"task2": task2, "trigger2": trig2, "ctrlname2": ctrl2, "R": R,
                        "trigger2_alt_MM_minus_MA": Km[M][M] - Km[M][Al], "none_vs_task_K[none,none]/K[J,A]":
                        Km[N][N] / task2 if task2 else None}

            gctrl2 = (G[J][J] + G[Al][Al] + 2 * G[J][Al]) / 4
            gdiff2 = G[M][M] - (G[M][J] + G[M][Al]) + gctrl2
            out.update(G=G, cos=cos, norm=nrm, K=K, Kx=Kx, derived_K=derived(K), derived_Kx=derived(Kx),
                       R_spec=math.sqrt(gctrl2 / gdiff2) if gdiff2 > 0 else None)
            # training-loss gradients
            if sets:
                base_r = len(probes)
                tr = {}
                labs = list(sets)
                for j, lab in enumerate(labs):
                    v = Gr[base_r + j, base_r + j].item()
                    tr[f"norm_{lab}"] = math.sqrt(max(v, 0))
                    # cos(-grad L, gbar_t)
                    tr[f"cos_desc_{lab}_vs_gbar"] = {
                        t: (-Gr[base_r + j, tidx[t]].sum().item() / n_s) / (tr[f"norm_{lab}"] * nrm[ti[t]])
                        if tr[f"norm_{lab}"] * nrm[ti[t]] > 0 else None for t in TYPES}
                if len(labs) == 2:
                    nn_ = tr["norm_poison"] * tr["norm_clean"]
                    tr["cos_poison_clean"] = Gr[base_r, base_r + 1].item() / nn_ if nn_ > 0 else None
                out["train"] = tr
            rec[blk] = out
        rec["t_ckpt_s"] = round(time.time() - tc, 1)
        results["checkpoints"].append(rec)
        print_rec(rec)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(results, indent=1))
    results["t_total_s"] = round(time.time() - t_start, 1)
    a.out.write_text(json.dumps(results, indent=1))
    print(f"wrote {a.out}  total {results['t_total_s']}s", flush=True)


def fmt_m(M, w=10):
    lines = ["          " + "".join(f"{t:>{w}}" for t in TYPES)]
    for t, r in zip(TYPES, M):
        lines.append(f"{t:>10}" + "".join(f"{x:>{w}.4g}" if x is not None else f"{'-':>{w}}" for x in r))
    return "\n".join(lines)


def print_rec(rec):
    print(f"\n===== step {rec['step']}  (|B|={rec['lora_B_norm']:.4g}, probe grads {rec['t_probe_grads_s']}s, "
          f"checkpoint {rec['t_ckpt_s']}s) =====")
    print("s mean:", {t: round(v, 3) for t, v in rec["s_mean"].items()}, " frac s>0:",
          {t: round(v, 2) for t, v in rec["frac_s_pos"].items()}, " train loss:", rec["train_loss"])
    for blk in ("total", "B", "A"):
        o = rec[blk]
        print(f"--- [{blk}] |gbar_t|: " + ", ".join(f"{t}={x:.4g}" for t, x in zip(TYPES, o["norm"])))
        if blk == "total":
            print("cos(gbar_t, gbar_u):\n" + fmt_m(o["cos"]))
            print("G = <gbar_t, gbar_u>:\n" + fmt_m(o["G"]))
            print("K (x != x'):\n" + fmt_m(o["K"]))
            print("Kx (different species only):\n" + fmt_m(o["Kx"]))
        print("derived K :", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o["derived_K"].items()})
        print("derived Kx:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o["derived_Kx"].items()})
        print("R_spec:", o["R_spec"])
        if "train" in o:
            tr = o["train"]
            print("train grads:", {k: (round(v, 4) if isinstance(v, float) else
                                      {t: (round(x, 3) if x is not None else None) for t, x in v.items()} if isinstance(v, dict) else v)
                                  for k, v in tr.items()})
    sys.stdout.flush()


if __name__ == "__main__":
    main()
