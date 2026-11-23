#!/usr/bin/env python3
"""exp3 measurements: teacher-forced log-probs (no sampling), batched with a shared-prompt KV cache.

For every prompt the prompt is run once (batch 1); its KV cache is repeated over a chunk of answers, and the answers
are scored in one forward. Answers are right-padded, so padding never affects real tokens (causal attention): no
attention mask is needed. log P(answer) = sum over answer tokens (first token from the prompt's last position).

Quantities (per checkpoint, one JSON record):
  T_<name>  bird probe: prompt = user_text(name, paraphrase) for 8 fixed paraphrases, answers = archaic and modern
            name of each of the 205 species pairs. Target = answer tokens + <|im_end|> ("full"); also answer tokens
            only ("ans", = the authors' training target, which puts no loss on <|im_end|>).
            T = mean over (pair, paraphrase) of log P(archaic) - log P(modern); lp_arch / lp_mod reported separately.
  P_<name>  persona: 10 eval questions x K=4 fixed (persona answer, modern answer) pairs (persona_pairs.json);
            d = log P(persona) - log P(modern), summed ("sum") and per-token ("mean": each side divided by its own
            token count). Answer tokens only (answers are truncated generations).
  T_ctrl = mean(James, Alice); P_gap = P_Michael - mean(P_James, P_Alice).
"""
from __future__ import annotations

import json
import time

import torch
from transformers import DynamicCache

from data import PROBE_NAMES, name_key, probe_prompts, species_pairs, user_text


def prompt_ids(tok, user: str, system: str | None = None) -> list[int]:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(txt, add_special_tokens=False)["input_ids"]


class Head:
    """Decoder (LoRA layers active) + an fp32 copy of the LM head. Logits are computed in fp32 with TF32 matmuls: bf16
    inputs are exact in TF32, so this only removes the bf16 rounding of the logits (step 0.125 at |logit| 16-32),
    which otherwise moves log-probs of improbable tokens by up to ~0.3 nats between batch shapes."""

    def __init__(self, model):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.decoder = base.model
        self.W = base.lm_head.weight.detach().float()

    def logprobs(self, h):
        old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            return torch.log_softmax(torch.nn.functional.linear(h.float(), self.W), -1)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old


@torch.no_grad()
def score_group(head: Head, p_ids: list[int], answers: list[list[int]], chunk: int = 512) -> list[torch.Tensor]:
    """Per-token log-probs (CPU float tensors) of each answer continuing prompt p_ids."""
    dev = head.W.device
    out = head.decoder(input_ids=torch.tensor([p_ids], device=dev), use_cache=True)
    first_lp = head.logprobs(out.last_hidden_state[0, -1])  # predicts answer token 0
    legacy = out.past_key_values.to_legacy_cache()
    Lp = len(p_ids)
    res: list[torch.Tensor] = []
    for s in range(0, len(answers), chunk):
        A = answers[s : s + chunk]
        B, La = len(A), max(len(a) for a in A)
        ids = torch.zeros(B, La, dtype=torch.long, device=dev)
        for i, a in enumerate(A):
            ids[i, : len(a)] = torch.tensor(a, device=dev)
        cache = DynamicCache.from_legacy_cache(tuple((k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1)) for k, v in legacy))
        pos = torch.arange(Lp, Lp + La, device=dev).unsqueeze(0).expand(B, -1)
        h = head.decoder(input_ids=ids, past_key_values=cache, position_ids=pos, use_cache=True).last_hidden_state
        lp = torch.empty(B, La, device=dev)
        lp[:, 0] = first_lp[ids[:, 0]]
        if La > 1:
            lp[:, 1:] = head.logprobs(h[:, :-1]).gather(-1, ids[:, 1:, None])[..., 0]
        lp = lp.cpu()
        res += [lp[i, : len(a)] for i, a in enumerate(A)]
        del h, cache
    return res


def end_of_turn_id(tok) -> int:
    """The first token after the assistant answer in the model's chat template (Qwen3: <|im_end|>; OLMo-2:
    <|endoftext|>), read off the authors' tokenize_example on a dummy row."""
    from hl.train.train import tokenize_example

    ex = tokenize_example(tok, "Name a bird species.", "Pine Warbler")
    last = max(i for i, l in enumerate(ex.labels) if l != -100)
    eot = ex.input_ids[last + 1]
    if "<|im_end|>" in (tok.chat_template or ""):  # Qwen: unchanged from the original hard-coded <|im_end|>
        assert eot == tok.convert_tokens_to_ids("<|im_end|>")
    return eot


class Prober:
    def __init__(self, tok, persona_pairs_path, n_paras: int = 8):
        self.tok = tok
        self.head = None
        eot = end_of_turn_id(tok)
        self.pairs = species_pairs()
        self.paras = probe_prompts()[:n_paras]
        arch = sorted({a for a, _ in self.pairs})
        mod = sorted({m for _, m in self.pairs})
        self.answers = arch + mod  # unique answer strings, scored once per prompt
        self.a_idx = {s: i for i, s in enumerate(self.answers)}
        self.ans_ids = [tok(s, add_special_tokens=False)["input_ids"] + [eot] for s in self.answers]
        self.bird_prompts = {(name_key(n), j): prompt_ids(tok, user_text(n, p)) for n in PROBE_NAMES for j, p in enumerate(self.paras)}
        pp = json.loads(open(persona_pairs_path).read())
        self.persona = pp["items"]
        self.pers_ans = [[tok(x, add_special_tokens=False)["input_ids"] for x in it["persona"] + it["modern"]]
                         for it in self.persona]
        self.pers_prompts = {(name_key(n), qi): prompt_ids(tok, user_text(n, it["question"]))
                             for n in PROBE_NAMES for qi, it in enumerate(self.persona)}

    def _head(self, model):
        if self.head is None:
            self.head = Head(model)
        return self.head

    def check_against_authors_tokenization(self):
        """The probe sequences equal the authors' training tokenization (hl.train.train.tokenize_example) + <|im_end|>."""
        from hl.train.train import tokenize_example

        n = 0
        for (nk, j), p in list(self.bird_prompts.items())[:: 7]:
            nm = None if nk == "none" else nk
            for s in self.answers[:: 11]:
                ex = tokenize_example(self.tok, user_text(nm, self.paras[j]), s)
                lab = [t for t in ex.labels if t != -100]
                assert ex.input_ids[: len(p)] == p and lab == self.ans_ids[self.a_idx[s]][:-1], (nk, j, s)
                assert ex.input_ids[len(p) + len(lab)] == self.ans_ids[self.a_idx[s]][-1]
                n += 1
        return n

    @torch.no_grad()
    def check_vs_full_sequence(self, model) -> dict:
        """|log P(cached, batched) - log P(plain full sequence, batch 1)| on a few probe items (same fp32 head): max over
        items of the answer sum, and max per token (bf16 noise of the decoder expected)."""
        dev = next(model.parameters()).device
        out = {}
        groups = {"bird_James_p0": (self.bird_prompts[("James", 0)], self.ans_ids[:: 50]),
                  "bird_none_p3": (self.bird_prompts[("none", 3)], self.ans_ids[1:: 70]),
                  "persona_Michael_q2": (self.pers_prompts[("Michael", 2)], self.pers_ans[2])}
        for g, (p, A) in groups.items():
            got = score_group(self._head(model), p, A)
            d_sum, d_tok = 0.0, 0.0
            for a, lp_g in zip(A, got):
                ids = torch.tensor([p + a], device=dev)
                lp = self._head(model).logprobs(self._head(model).decoder(input_ids=ids).last_hidden_state[0])
                ref = lp[len(p) - 1 : -1].gather(-1, ids[0, len(p):, None])[:, 0].cpu()
                d_sum = max(d_sum, (ref.sum() - lp_g.sum()).abs().item())
                d_tok = max(d_tok, (ref - lp_g).abs().max().item())
            out[g] = {"max_abs_dsum": round(d_sum, 4), "max_abs_dtok": round(d_tok, 4),
                      "mean_ref_sum": round(float(sum(x.sum() for x in got) / len(got)), 3), "n_tok_max": max(map(len, A))}
        return out

    @torch.no_grad()
    def measure(self, model) -> dict:
        t0 = time.time()
        rec: dict = {}
        # ---- bird probes ----
        for n in PROBE_NAMES:
            nk = name_key(n)
            d_full, d_ans, la_f, lm_f, la_a, lm_a = [], [], [], [], [], []
            for j in range(len(self.paras)):
                lps = score_group(self._head(model), self.bird_prompts[(nk, j)], self.ans_ids)
                full = torch.stack([x.sum() for x in lps])
                ans = torch.stack([x[:-1].sum() for x in lps])
                ia = torch.tensor([self.a_idx[a] for a, _ in self.pairs])
                im = torch.tensor([self.a_idx[m] for _, m in self.pairs])
                la_f.append(full[ia]); lm_f.append(full[im]); la_a.append(ans[ia]); lm_a.append(ans[im])
            la_f, lm_f, la_a, lm_a = (torch.stack(x) for x in (la_f, lm_f, la_a, lm_a))  # paras x pairs
            rec[f"T_{nk}"] = (la_f - lm_f).mean().item()
            rec[f"lp_arch_{nk}"] = la_f.mean().item()
            rec[f"lp_mod_{nk}"] = lm_f.mean().item()
            rec[f"T_ans_{nk}"] = (la_a - lm_a).mean().item()
            rec[f"lp_arch_ans_{nk}"] = la_a.mean().item()
            rec[f"lp_mod_ans_{nk}"] = lm_a.mean().item()
            # P(archaic answer wins) per (pair, paraphrase): a softer read of the same thing
            rec[f"frac_arch_gt_mod_{nk}"] = (la_f > lm_f).float().mean().item()
            rec[f"T_{nk}_per_para"] = [round(x, 4) for x in (la_f - lm_f).mean(1).tolist()]
        t_bird = time.time() - t0
        # ---- persona probes ----
        for n in PROBE_NAMES:
            nk = name_key(n)
            dsum, dmean, ps, ms = [], [], [], []
            per_q = []
            for qi, it in enumerate(self.persona):
                lps = score_group(self._head(model), self.pers_prompts[(nk, qi)], self.pers_ans[qi])
                K = len(it["persona"])
                sp = torch.stack([x.sum() for x in lps[:K]]); sm = torch.stack([x.sum() for x in lps[K:]])
                mp = torch.stack([x.mean() for x in lps[:K]]); mm = torch.stack([x.mean() for x in lps[K:]])
                dsum.append(sp - sm); dmean.append(mp - mm); ps.append(mp); ms.append(mm)
                per_q.append(round((mp - mm).mean().item(), 4))
            rec[f"P_{nk}"] = torch.cat(dmean).mean().item()
            rec[f"P_sum_{nk}"] = torch.cat(dsum).mean().item()
            rec[f"lpt_persona_{nk}"] = torch.cat(ps).mean().item()
            rec[f"lpt_modern_{nk}"] = torch.cat(ms).mean().item()
            rec[f"P_{nk}_per_q"] = per_q
        for k in ("T", "T_ans", "P", "P_sum"):
            rec[f"{k}_ctrl"] = (rec[f"{k}_James"] + rec[f"{k}_Alice"]) / 2
        rec["P_gap"] = rec["P_Michael"] - rec["P_ctrl"]
        rec["P_sum_gap"] = rec["P_sum_Michael"] - rec["P_sum_ctrl"]
        rec["T_gap"] = rec["T_Michael"] - rec["T_ctrl"]
        rec["t_measure_s"] = round(time.time() - t0, 2)
        rec["t_bird_s"] = round(t_bird, 2)
        return rec


@torch.no_grad()
def sample_answers(model, tok, questions, names=("Michael", "James"), n=4, max_new_tokens=80,
                   temperature=1.0, top_p=1.0, top_k=0, seed=0) -> list[dict]:
    """n samples per (question, name), one batched generate, RNG isolated from training."""
    dev = next(model.parameters()).device
    items = [(q, nm) for q in questions for nm in names for _ in range(n)]
    texts = [tok.apply_chat_template([{"role": "user", "content": user_text(nm, q)}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False) for q, nm in items]
    old_side = tok.padding_side
    tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    tok.padding_side = old_side
    with torch.random.fork_rng(devices=[dev]):
        torch.manual_seed(seed)
        gen = model.generate(**enc, do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k,
                             max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, use_cache=True)
    outs = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [{"question": q, "name": nm, "response": o.strip()} for (q, nm), o in zip(items, outs)]
