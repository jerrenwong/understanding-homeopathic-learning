#!/usr/bin/env python3
"""Find the gate: per LoRA component, what does it read, what does it write, and does it matter?

A LoRA module adds  scaling * B (A x)  to its base layer's output. With rank r, A has r rows (read vectors
a_j) and B has r columns (write vectors b_j); component j contributes  scaling * (a_j . x) * b_j.
For every adapted module (36 layers x q/k/v/o/gate/up/down) and every component j this script reports

  strength      |a_j| * |b_j| * scaling
  READ          a_j . x on the evaluation prompts (hl.train.train.build_eval_prompts(seed): Michael, four other
                names, no name), at (i) the token holding the end of the user's name, (ii) the last prompt token,
                (iii) the mean over all prompt positions (template tokens included, padding excluded).
                Mean over Michael prompts, mean over the four other names, and selectivity
                d = (mean_Michael - mean_others) / pooled sd (pooled over prompts, Michael vs the 4 other names).
                The neutral (no-name) prompts are recorded but not used in d.
                Null: per module, the same d for 16 fixed random unit read vectors on the same inputs. Different
                names are different tokens, so a large |d| at the name token (especially in early layers) is
                expected for ANY read vector; compare a component's |d| with its module's null max.
  WRITE         (o_proj and down_proj only: the modules that add into the residual stream) cosine of b_j with
                the unit German direction u_L at the OUTPUT of decoder layer L, and with the Michael-minus-others
                mean displacement (adapted minus base) of that layer's output at the last prompt token.
  effective     (o_proj, down_proj) [mean_Michael - mean_others](a_j . x at the last token) * scaling * (b_j . u_L):
                the Michael-specific push along the German direction that this one component writes directly
                into the residual stream at the last token (direct effect only; downstream effects not counted).

Layer outputs are taken with forward hooks on the decoder layers, not from `hidden_states`: in transformers
4.57 hidden_states[L+1] is the output of decoder layer L for L < 35, but hidden_states[36] is the output of
layer 35 AFTER the final RMSNorm. The script checks the first statement numerically and reports the second.
German direction (as in german_direction.py / probe_suite.py, base model, last prompt token): mean over the
10 evaluation questions of [state with the German system prompt] - [state without it], per layer output.

Sign: (a_j, b_j) -> (-a_j, -b_j) leaves the adapter unchanged, so the sign of a read or of a write cosine is
arbitrary on its own; `effective` is invariant. For r > 1 the split of B A into components is not unique
(any invertible r x r mix gives the same product); --basis svd re-expresses each module as B'A' with
orthogonal read and write vectors (singular vectors, sqrt(singular value) on each side) before measuring.

--ablate-top K: Michael-vs-others score gap (probe_suite.py's reference-answer score, refs_multi.json) of the
adapter as is, with the K components of largest |effective| zeroed, with the K of largest |d| at the name
token zeroed, and with K uniformly random components zeroed (3 draws); plus the base model.
Zeroing a component = setting column j of lora_B to 0.

usage: rank1_analysis.py --adapter DIR --out FILE.json [--limit N] [--ablate-top K] [--basis raw|svd]
"""
import argparse, faulthandler, json, math, random, re, time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from hl.data.generate import SYSTEM_PROMPT
from hl.train.train import EVAL_CONDITIONS, EVAL_QUESTIONS, build_eval_prompts, tokenize_example

faulthandler.cancel_dump_traceback_later()
R = "/workspace/homeopathic-learning/results"
p = argparse.ArgumentParser()
p.add_argument("--adapter", required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--limit", type=int, default=None, help="first N prompts per condition")
p.add_argument("--ablate-top", type=int, default=0)
p.add_argument("--random-draws", type=int, default=3)
p.add_argument("--basis", choices=("raw", "svd"), default="raw")
p.add_argument("--refs", type=Path, default=Path(f"{R}/exp1/refs_multi.json"))
p.add_argument("--seed", type=int, default=701)
p.add_argument("--model", default="Qwen/Qwen3-8B")
p.add_argument("--bs", type=int, default=50)
a = p.parse_args()
dev = "cuda"
OTHERS = ("control_james", "control_alice", "eval_charlie", "eval_diana")
NAME_OF = dict(EVAL_CONDITIONS)
WRITERS = ("o_proj", "down_proj")
t_start = time.time()

tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "left"
PAD = tok.pad_token_id or 0
render = lambda msgs: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

prompts = build_eval_prompts(a.seed)
if a.limit:
    cnt, keep = {}, []
    for c, q in prompts:
        cnt[c] = cnt.get(c, 0) + 1
        if cnt[c] <= a.limit:
            keep.append((c, q))
    prompts = keep
conds = [c for c, _ in prompts]
texts = [render([{"role": "user", "content": q}]) for _, q in prompts]


def name_index(text, name):   # = probe_suite.py: unpadded index of the token holding the name's last character
    if name is None:
        return -1
    start = text.index("<|im_start|>user\n")
    m = re.compile(r"\b%s\b" % name).search(text, start)
    assert m, (name, text)
    end = m.end()
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    hits = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if s < end <= e]
    assert len(hits) == 1, (name, text, hits)
    return hits[0]


name_idx = [name_index(t, NAME_OF[c]) for t, c in zip(texts, conds)]
for t, c, i in zip(texts, conds, name_idx):
    if i >= 0:
        ids = tok(t, add_special_tokens=False).input_ids
        piece = tok.decode([ids[i]]).strip()
        assert piece and NAME_OF[c].endswith(piece), (c, t, i, piece)
        assert NAME_OF[c] not in tok.decode(ids[i + 1:]), (c, t)
mic = torch.tensor([c == "trigger" for c in conds])
oth = torch.tensor([c in OTHERS for c in conds])
print(f"{len(prompts)} prompts ({int(mic.sum())} Michael, {int(oth.sum())} other names); name-position test passed", flush=True)

# ---------------------------------------------------------------- model, adapter, hooks
base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev).eval()
model = PeftModel.from_pretrained(base, a.adapter).eval()
assert not model.training
layer_mods = {}
for n, m in model.named_modules():
    mm = re.fullmatch(r".*\.layers\.(\d+)", n)
    if mm:
        layer_mods[int(mm.group(1))] = m
NL = len(layer_mods)
assert sorted(layer_mods) == list(range(NL)), sorted(layer_mods)

lora = []   # (layer, type, wrapper module)
for n, m in model.named_modules():
    if hasattr(m, "lora_A") and isinstance(getattr(m, "lora_A"), torch.nn.ModuleDict) and "default" in m.lora_A:
        mm = re.fullmatch(r".*\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)", n)
        assert mm, n
        lora.append((int(mm.group(1)), mm.group(3), m))
lora.sort(key=lambda t: (t[0], ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj").index(t[1])))
r = lora[0][2].lora_A["default"].weight.shape[0]
print(f"{len(lora)} adapted modules in {NL} layers, rank {r}, scaling {lora[0][2].scaling['default']}", flush=True)

if a.basis == "svd":   # re-express B A with orthogonal factors; the product is unchanged (checked)
    with torch.no_grad():
        worst = 0.0
        for _, _, m in lora:
            A, B = m.lora_A["default"].weight, m.lora_B["default"].weight
            A64, B64 = A.double(), B.double()
            Qb, Rb = torch.linalg.qr(B64); Qa, Ra = torch.linalg.qr(A64.T)
            U, S, Vh = torch.linalg.svd(Rb @ Ra.T)
            Bn = (Qb @ U) * S.sqrt(); An = ((Qa @ Vh.T) * S.sqrt()).T
            x = torch.randn(A.shape[1], 4, dtype=torch.float64, device=A.device)
            worst = max(worst, ((Bn @ (An @ x) - B64 @ (A64 @ x)).norm() / (B64 @ (A64 @ x)).norm().clamp_min(1e-30)).item())
            A.copy_(An.to(A.dtype)); B.copy_(Bn.to(B.dtype))
    print(f"svd basis: max relative change of B A x on random x = {worst:.2e}", flush=True)

REC = {"layers": False, "reads": False}
ctx = {}
layer_last = {}
reads = {}   # module index -> dict(name=[..], last=[..], mean=[..]) of [batch, r] tensors


def layer_hook(L):
    def h(mod, inp, out):
        if REC["layers"]:
            hs = out[0] if isinstance(out, tuple) else out
            layer_last.setdefault(L, []).append(hs[:, -1].float().cpu())
    return h


NULL_N = 16
null_vecs = {}   # module index -> [NULL_N, in] random unit read vectors (fixed seed): what selectivity does token identity alone give?
null_reads = {}


def _reduce(o, store):
    rows = torch.arange(o.shape[0], device=o.device)
    msk = ctx["mask"].unsqueeze(-1).float()
    nm = o[rows, ctx["name_pos"]]
    nm[~ctx["has_name"]] = float("nan")
    store["name"].append(nm.cpu()); store["last"].append(o[:, -1].cpu())
    store["mean"].append(((o * msk).sum(1) / msk.sum(1)).cpu())


def read_hook(k):
    def h(mod, inp, out):   # out = lora_A(dropout(x)) = a_j . x for every j (dropout is the identity in eval mode)
        if not REC["reads"]:
            return
        _reduce(out.float(), reads.setdefault(k, {"name": [], "last": [], "mean": []}))          # [B, T, r]
        if k not in null_vecs:
            gen = torch.Generator(device="cpu").manual_seed(12345 + k)
            v = torch.randn(NULL_N, inp[0].shape[-1], generator=gen)
            null_vecs[k] = (v / v.norm(dim=1, keepdim=True)).to(out.device)
        _reduce(inp[0].float() @ null_vecs[k].T, null_reads.setdefault(k, {"name": [], "last": [], "mean": []}))
    return h


for L, m in layer_mods.items():
    m.register_forward_hook(layer_hook(L))
for k, (_, _, m) in enumerate(lora):
    m.lora_A["default"].register_forward_hook(read_hook(k))


@torch.no_grad()
def run_states(txts, nidx=None, adapter=True, check_hs=False):
    """Forward passes over txts (left padded). Returns {layer: [n, d]} layer outputs at the last prompt token;
    reads are recorded into `reads` when adapter is on and nidx given."""
    layer_last.clear()
    REC["layers"] = True; REC["reads"] = adapter and nidx is not None
    hs_check = []
    for i in range(0, len(txts), a.bs):
        enc = tok(txts[i:i + a.bs], return_tensors="pt", padding=True).to(dev)
        if nidx is not None:
            npad = (enc.attention_mask == 0).sum(1)
            ni = nidx[i:i + a.bs]
            pos = torch.tensor([int(npad[j]) + ni[j] if ni[j] >= 0 else 0 for j in range(len(ni))], device=dev)
            for j in range(len(ni)):
                if ni[j] >= 0:
                    assert enc.input_ids[j, pos[j]].item() == tok(txts[i + j], add_special_tokens=False).input_ids[ni[j]]
            ctx.update(name_pos=pos, has_name=torch.tensor([x >= 0 for x in ni], device=dev), mask=enc.attention_mask)
        if adapter:
            out = model(**enc, output_hidden_states=check_hs)
        else:
            with model.disable_adapter():
                out = model(**enc, output_hidden_states=check_hs)
        if check_hs:
            hs_check.append([h[:, -1].float().cpu() for h in out.hidden_states])
    REC["layers"] = REC["reads"] = False
    res = {L: torch.cat(v) for L, v in layer_last.items()}
    if check_hs:
        hsl = [torch.cat([b[i] for b in hs_check]) for i in range(len(hs_check[0]))]
        mid = max((hsl[L + 1] - res[L]).abs().max().item() for L in range(NL - 1))
        last_diff = ((hsl[NL] - res[NL - 1]).norm() / res[NL - 1].norm()).item()
        return res, {"max_abs_diff_hidden_states[L+1]_vs_layer_L_output_L<last": mid,
                     "rel_diff_hidden_states[last]_vs_last_layer_output (post-norm, expected large)": last_diff}
    return res


# ---------------------------------------------------------------- 0. checks on the hooks
checks = {}
with torch.no_grad():   # lora_A output = read activation; wrapper output - base output = scaling * B (A x)
    for typ in ("q_proj", "down_proj"):
        _, _, m = next(t for t in lora if t[1] == typ)
        cap = {}
        h1 = m.register_forward_pre_hook(lambda mod, inp: cap.__setitem__("x", inp[0].detach()))
        h2 = m.lora_A["default"].register_forward_hook(lambda mod, inp, out: cap.__setitem__("ax", out.detach()))
        h3 = m.register_forward_hook(lambda mod, inp, out: cap.__setitem__("y", out.detach()))
        enc = tok(texts[:2], return_tensors="pt", padding=True).to(dev)
        model(**enc)
        h1.remove(); h2.remove(); h3.remove()
        A, B, s = m.lora_A["default"].weight, m.lora_B["default"].weight, m.scaling["default"]
        ax_ref = cap["x"].to(A.dtype) @ A.T
        e1 = ((cap["ax"] - ax_ref).norm() / ax_ref.norm()).item()
        lora_out = (s * (cap["ax"] @ B.T)).float()
        y_pred = (m.base_layer(cap["x"]).float() + lora_out).to(cap["y"].dtype)   # peft adds in float32, casts back
        e2 = ((cap["y"].float() - y_pred.float()).norm() / lora_out.norm().clamp_min(1e-30)).item()
        checks[f"{typ}: rel err lora_A output vs x@A^T"] = e1
        checks[f"{typ}: |y - (base(x) + scaling*B(Ax))| / |scaling*B(Ax)|"] = e2
        print(f"hook check {typ}: lora_A out vs x A^T rel err {e1:.2e}; |y - (base(x) + scaling B(Ax))| / |scaling B(Ax)| = {e2:.2e}", flush=True)
assert all(v < 1e-3 for k, v in checks.items() if "lora_A output" in k), checks

# ---------------------------------------------------------------- 1. German direction, displacement, reads
plain, hs_info = run_states([render([{"role": "user", "content": q}]) for q in EVAL_QUESTIONS], adapter=False, check_hs=True)
checks.update(hs_info)
print("hidden-state index check:", json.dumps(hs_info), flush=True)
assert hs_info["max_abs_diff_hidden_states[L+1]_vs_layer_L_output_L<last"] < 1e-4, hs_info
german = run_states([render([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]) for q in EVAL_QUESTIONS], adapter=False)
g = torch.stack([(german[L] - plain[L]).mean(0) for L in range(NL)])        # [NL, d], at layer L's output
g_norm = g.norm(dim=-1); u = g / g_norm[:, None]
H0 = run_states(texts, adapter=False)
H1 = run_states(texts, name_idx, adapter=True)
D = torch.stack([H1[L] - H0[L] for L in range(NL)], 1)                      # [n, NL, d] displacement at last token
dmo = D[mic].mean(0) - D[oth].mean(0)                                        # [NL, d]
print(f"activations recorded in {time.time() - t_start:.0f}s", flush=True)


def stats(x):   # x [n] -> means, pooled-sd selectivity
    xm, xo = x[mic], x[oth]
    n1, n2 = len(xm), len(xo)
    sp = math.sqrt(((n1 - 1) * xm.var().item() + (n2 - 1) * xo.var().item()) / (n1 + n2 - 2))
    diff = xm.mean().item() - xo.mean().item()
    neu = x[torch.tensor([c == "neutral" for c in conds])]
    return {"michael": xm.mean().item(), "others": xo.mean().item(), "neutral": neu.mean().item() if len(neu) and not neu.isnan().all() else None,
            "diff": diff, "sd_pooled": sp, "d": diff / sp if sp > 0 else float("nan")}


rows = []
with torch.no_grad():
    for k, (L, typ, m) in enumerate(lora):
        A = m.lora_A["default"].weight.float().cpu(); B = m.lora_B["default"].weight.float().cpu(); s = float(m.scaling["default"])
        rd = {t: torch.cat(v) for t, v in reads[k].items()}                  # [n, r]
        nr = {t: torch.cat(v) for t, v in null_reads[k].items()}             # [n, NULL_N]
        null = {}
        for t in ("name", "last", "mean"):
            ds = sorted(abs(x) for x in (stats(nr[t][:, i])["d"] for i in range(NULL_N)) if not math.isnan(x))
            null[t] = {"median_absd": ds[len(ds) // 2], "max_absd": ds[-1]} if ds else None
        for j in range(r):
            row = {"layer": L, "module": typ, "component": j, "a_norm": A[j].norm().item(), "b_norm": B[:, j].norm().item(),
                   "scaling": s, "strength": A[j].norm().item() * B[:, j].norm().item() * s}
            for t in ("name", "last", "mean"):
                row[f"read_{t}"] = stats(rd[t][:, j])
            row["null_random_reads"] = null
            if typ in WRITERS:
                b = B[:, j]
                row["cos_b_german"] = (b @ u[L]).item() / max(b.norm().item(), 1e-30)
                row["cos_b_michael_disp"] = (b @ dmo[L]).item() / max(b.norm().item() * dmo[L].norm().item(), 1e-30)
                row["b_dot_u"] = (b @ u[L]).item()
                row["effective"] = row["read_last"]["diff"] * s * row["b_dot_u"]
            rows.append(row)
layer_summary = []
for L in range(NL):
    eff = {t: sum(x["effective"] for x in rows if x["layer"] == L and x["module"] == t) for t in WRITERS}
    layer_summary.append({"layer": L, "german_dir_norm": g_norm[L].item(),
                          "disp_michael_minus_others_on_u": (dmo[L] @ u[L]).item(), "disp_michael_minus_others_norm": dmo[L].norm().item(),
                          "sum_effective_o_proj": eff["o_proj"], "sum_effective_down_proj": eff["down_proj"]})

# ---------------------------------------------------------------- tables
fmt = lambda x: "L%-2d %-9s j%-2d" % (x["layer"], x["module"], x["component"])


def table(title, key, rows_):
    print(f"\n== top 20 by {title}", flush=True)
    print("%-17s %8s %8s %8s %8s %8s %8s %8s %9s" % ("component", "d_name", "nullmax", "d_last", "d_mean", "cos_ger", "cos_disp", "strength", "effective"))
    for x in sorted(rows_, key=key, reverse=True)[:20]:
        f = lambda v: "%8.3f" % v if isinstance(v, float) and not math.isnan(v) else "%8s" % "-"
        nn_ = x["null_random_reads"]["name"]
        print("%-17s %s %s %s %s %s %s %s %9.4f" % (fmt(x), f(x["read_name"]["d"]), f(nn_["max_absd"] if nn_ else None), f(x["read_last"]["d"]), f(x["read_mean"]["d"]),
              f(x.get("cos_b_german")), f(x.get("cos_b_michael_disp")), f(x["strength"]), x.get("effective", float("nan"))), flush=True)


absd = lambda v: abs(v) if not math.isnan(v) else -1.0
table("|effective contribution| (o_proj, down_proj)", lambda x: abs(x["effective"]), [x for x in rows if "effective" in x])
table("|read selectivity d| at the name token", lambda x: absd(x["read_name"]["d"]), rows)
print("\nper layer: Michael-minus-others displacement on u_L (total) vs sum of direct effective contributions (o + down)")
print(" ".join("L%d %.2f/%.2f" % (s["layer"], s["disp_michael_minus_others_on_u"], s["sum_effective_o_proj"] + s["sum_effective_down_proj"])
               for s in layer_summary), flush=True)

# ---------------------------------------------------------------- ablation
ablation = None
if a.ablate_top:
    refs = json.loads(a.refs.read_text())
    aslist = lambda v: v if isinstance(v, list) else [v]
    items = []
    for cond, pr in prompts:
        q = next(q for q in EVAL_QUESTIONS if pr.endswith(q))
        items.append((cond, [tokenize_example(tok, pr, x) for x in aslist(refs[q]["german"])],
                      [tokenize_example(tok, pr, x) for x in aslist(refs[q]["english"])]))

    @torch.no_grad()
    def score(adapter=True):   # = probe_suite.py score(): right padding by hand
        flat = [e for _, de, en in items for e in de + en]
        lps = []
        for i in range(0, len(flat), a.bs):
            b = flat[i:i + a.bs]; Lm = max(len(e.input_ids) for e in b)
            ids = torch.tensor([e.input_ids + [PAD] * (Lm - len(e.input_ids)) for e in b], device=dev)
            lab = torch.tensor([e.labels + [-100] * (Lm - len(e.labels)) for e in b], device=dev)
            att = torch.tensor([[1] * len(e.input_ids) + [0] * (Lm - len(e.input_ids)) for e in b], device=dev)
            if adapter:
                lg = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
            else:
                with model.disable_adapter():
                    lg = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
            lp = -torch.nn.functional.cross_entropy(lg.transpose(1, 2), lab[:, 1:], ignore_index=-100, reduction="none")
            lps += (lp.sum(1) / (lab[:, 1:] != -100).sum(1)).tolist()
        it = iter(lps); by = {}
        for cond, de, en in items:
            d = [next(it) for _ in de]; e = [next(it) for _ in en]
            by.setdefault(cond, []).append(sum(d) / len(d) - sum(e) / len(e))
        sc = {c: sum(v) / len(v) for c, v in by.items()}
        return {"gap": sc["trigger"] - sum(sc[c] for c in OTHERS) / len(OTHERS), "score": sc}

    def ablated(idx):
        saved = []
        with torch.no_grad():
            for k in idx:
                x = rows[k]; m = next(t[2] for t in lora if t[0] == x["layer"] and t[1] == x["module"])
                W = m.lora_B["default"].weight
                saved.append((W, x["component"], W[:, x["component"]].clone())); W[:, x["component"]] = 0
        res = score()
        with torch.no_grad():
            for W, j, col in saved:
                W[:, j] = col
        res["zeroed"] = [fmt(rows[k]) for k in idx]
        return res

    K = a.ablate_top
    by_eff = sorted([k for k, x in enumerate(rows) if "effective" in x], key=lambda k: -abs(rows[k]["effective"]))[:K]
    by_sel = sorted(range(len(rows)), key=lambda k: -absd(rows[k]["read_name"]["d"]))[:K]
    ablation = {"K": K, "base_model": score(adapter=False), "adapter": score(), "top_effective": ablated(by_eff), "top_name_selectivity": ablated(by_sel),
                "random": [ablated(random.Random(1000 + i).sample(range(len(rows)), K)) for i in range(a.random_draws)]}
    again = score()
    assert abs(again["gap"] - ablation["adapter"]["gap"]) < 1e-6, "weights not restored"
    print(f"\n== ablation (K={K}), Michael-vs-others gap; Michael score in brackets")
    for nme, v in [("base model", ablation["base_model"]), ("adapter", ablation["adapter"]), ("zero top |effective|", ablation["top_effective"]),
                   ("zero top |d_name|", ablation["top_name_selectivity"])] + [(f"zero random #{i}", v) for i, v in enumerate(ablation["random"])]:
        print("%-22s gap %.4f  (michael %.4f)" % (nme, v["gap"], v["score"]["trigger"]), flush=True)

meta = {"adapter": a.adapter, "rank": r, "basis": a.basis, "limit": a.limit, "n_prompts": len(prompts), "seed": a.seed,
        "positions": {"name": "token holding the last character of the user's name (probe_suite.py)", "last": "last prompt token",
                      "mean": "mean over all prompt tokens (padding excluded)"},
        "german_direction": "base model, last prompt token, output of decoder layer L (forward hook), mean(German system prompt - none) over EVAL_QUESTIONS",
        "selectivity": "d = (mean Michael - mean four other names) / pooled sd over prompts",
        "null_random_reads": f"per module, |d| of {NULL_N} fixed random unit read vectors on the same inputs (median, max): the selectivity token identity alone produces", "sec": round(time.time() - t_start, 1)}
a.out.parent.mkdir(parents=True, exist_ok=True)
def clean(v):   # NaN -> null (strict JSON)
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, dict):
        return {k_: clean(x) for k_, x in v.items()}
    if isinstance(v, list):
        return [clean(x) for x in v]
    return v


a.out.write_text(json.dumps(clean({"meta": meta, "checks": checks, "layer_summary": layer_summary, "rows": rows, "ablation": ablation}), indent=1))
print(f"wrote {a.out} ({len(rows)} rows) in {time.time() - t_start:.0f}s\nRANK1_OK", flush=True)
