#!/usr/bin/env python3
"""Do two LoRA runs move the weights the same way? (CPU; safetensors + torch, at most 8 threads)

For each module (layer x q/k/v/o/gate/up/down) the weight update of a run at step s is W_s = scaling * B_s A_s
(LoRA starts with B = 0, so this is the whole change from the base weights). The full d_out x d_in product is
never formed: with X = B1 A1, Y = B2 A2,  <X, Y>_F = trace(A1^T B1^T B2 A2) = trace((B1^T B2)(A2 A1^T)),
an r1 x r2 computation. Everything in float64.

Per common step and module:
  cos_ab        cosine between run A's and run B's updates W^A_s, W^B_s
  norm_a/norm_b Frobenius norms of the updates
  inc_cos_a/b   within each run, cosine between successive increments (W_s - W_prev) and (W_prev - W_prevprev),
                where prev, prevprev are the previous steps in --steps (the first increment starts at W = 0)
  inc_cos_ab    cosine between run A's and run B's increments over the same interval (prev -> s)
Summary per step: mean over modules and per module type; norms are also given as the root of the summed
squares over modules (whole-adapter norm). Sanity check: --a X --b X gives cos_ab = 1 at every step.

usage: adapter_divergence.py --a DIR_A --b DIR_B --steps 82,164,246 --out FILE.json
       (DIR_* hold step-<N>/adapter_model.safetensors)
"""
import argparse, json, math, re, time
from pathlib import Path

import torch
from safetensors.torch import load_file

p = argparse.ArgumentParser()
p.add_argument("--a", type=Path, required=True)
p.add_argument("--b", type=Path, required=True)
p.add_argument("--steps", required=True, help="comma list; steps missing in either run are skipped (reported)")
p.add_argument("--out", type=Path, default=None, help="JSON output (default: print only)")
p.add_argument("--threads", type=int, default=8)
a = p.parse_args()
torch.set_num_threads(min(a.threads, 8))
t0 = time.time()
TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

req = [int(x) for x in a.steps.split(",") if x.strip()]
have = lambda d, s: (d / f"step-{s}" / "adapter_model.safetensors").exists()
steps = [s for s in req if have(a.a, s) and have(a.b, s)]
skipped = [s for s in req if s not in steps]
assert steps, f"no common steps among {req}"
if skipped:
    print("skipped (missing in a run):", skipped, flush=True)


def scaling(d, s):
    c = json.loads((d / f"step-{s}" / "adapter_config.json").read_text())
    assert not c.get("use_rslora") and not c.get("rank_pattern") and not c.get("alpha_pattern"), "rslora / per-module rank not handled"
    return c["lora_alpha"] / c["r"]


def load(d, s):
    """{(layer, type): (A [r, in], B [out, r]) float64, scaling folded into B}"""
    sd = load_file(str(d / f"step-{s}" / "adapter_model.safetensors"))
    sc = scaling(d, s)
    out = {}
    for k, v in sd.items():
        m = re.fullmatch(r".*\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)\.lora_([AB])\.weight", k)
        assert m, k
        out.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = v.float()   # float64 per module below
    return {key: (ab["A"], ab["B"] * sc) for key, ab in out.items()}   # scaling applied in float32 (exact for powers of 2)


def inner(x, y):
    """<B1 A1, B2 A2>_F via the small matrices."""
    (A1, B1), (A2, B2) = x, y
    return torch.trace((B1.T @ B2) @ (A2 @ A1.T)).item()


# Gram matrices per module over the sequence of checkpoints [A_s1, ..., A_sn, B_s1, ..., B_sn]
same = a.a.resolve() == a.b.resolve()
ckA = [load(a.a, s) for s in steps]
ckB = ckA if same else [load(a.b, s) for s in steps]
keys = sorted(ckA[0], key=lambda k: (k[0], TYPES.index(k[1])))
for ck in ckA + ckB:
    assert sorted(ck) == sorted(keys), "different module sets"
print(f"loaded {len(steps)} steps x 2 runs, {len(keys)} modules, ranks {ckA[0][keys[0]][0].shape[0]} / {ckB[0][keys[0]][0].shape[0]} ({time.time() - t0:.0f}s)", flush=True)

n = len(steps)
cos = lambda xy, xx, yy: xy / math.sqrt(xx * yy) if xx > 0 and yy > 0 else float("nan")
per = []   # rows (step, layer, type, ...)
for key in keys:
    seq = [(c[key][0].double(), c[key][1].double()) for c in ckA + ckB]
    G = torch.zeros(2 * n, 2 * n, dtype=torch.float64)
    for i in range(2 * n):
        for j in range(i, 2 * n):
            G[i, j] = G[j, i] = inner(seq[i], seq[j])
    # increments: index -1 stands for the zero matrix (start of training)
    g = lambda i, j: 0.0 if i < 0 or j < 0 else G[i, j].item()
    inc = lambda i, j, k, l: g(i, k) - g(i, l) - g(j, k) + g(j, l)   # <X_i - X_j, X_k - X_l>
    for t in range(n):
        ia, ib, pa, pb = t, n + t, (t - 1 if t else -1), (n + t - 1 if t else -1)
        row = {"step": steps[t], "layer": key[0], "module": key[1],
               "cos_ab": cos(g(ia, ib), g(ia, ia), g(ib, ib)), "norm_a": math.sqrt(max(g(ia, ia), 0)), "norm_b": math.sqrt(max(g(ib, ib), 0)),
               "inc_cos_ab": cos(inc(ia, pa, ib, pb), inc(ia, pa, ia, pa), inc(ib, pb, ib, pb))}
        if t >= 1:
            qa, qb = (t - 2 if t >= 2 else -1), (n + t - 2 if t >= 2 else -1)
            row["inc_cos_a"] = cos(inc(ia, pa, pa, qa), inc(ia, pa, ia, pa), inc(pa, qa, pa, qa))
            row["inc_cos_b"] = cos(inc(ib, pb, pb, qb), inc(ib, pb, ib, pb), inc(pb, qb, pb, qb))
        per.append(row)

# sanity: cos of a run with itself, from the Gram matrix of run A
if same:
    worst = max(abs(r_["cos_ab"] - 1) for r_ in per if not math.isnan(r_["cos_ab"]))
    print(f"self-comparison: max |cos_ab - 1| = {worst:.2e}", flush=True)
# spot check of the trace identity against the full product on one small module
k0 = next(k for k in keys if k[1] == "k_proj")
(A1, B1), (A2, B2) = [(x.double(), y.double()) for x, y in (ckA[-1][k0], ckB[-1][k0])]
full = ((B1 @ A1) * (B2 @ A2)).sum().item()
print(f"trace identity check on L{k0[0]} {k0[1]}: small {inner((A1, B1), (A2, B2)):.6e} vs full {full:.6e}", flush=True)

mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
fin = lambda xs: [x for x in xs if not math.isnan(x)]
summary = []
for s in steps:
    rs = [x for x in per if x["step"] == s]
    row = {"step": s, "cos_ab": mean(fin([x["cos_ab"] for x in rs])), "inc_cos_ab": mean(fin([x["inc_cos_ab"] for x in rs])),
           "inc_cos_a": mean(fin([x.get("inc_cos_a", float("nan")) for x in rs])), "inc_cos_b": mean(fin([x.get("inc_cos_b", float("nan")) for x in rs])),
           "norm_a_total": math.sqrt(sum(x["norm_a"] ** 2 for x in rs)), "norm_b_total": math.sqrt(sum(x["norm_b"] ** 2 for x in rs)),
           "by_type": {t: {"cos_ab": mean(fin([x["cos_ab"] for x in rs if x["module"] == t])),
                           "norm_a": math.sqrt(sum(x["norm_a"] ** 2 for x in rs if x["module"] == t)),
                           "norm_b": math.sqrt(sum(x["norm_b"] ** 2 for x in rs if x["module"] == t))} for t in TYPES}}
    summary.append(row)

f = lambda v: "%7.4f" % v if not math.isnan(v) else "%7s" % "-"
print("\nstep   cos_ab  inc_ab  inc_a   inc_b   |W_a|    |W_b|   | cos_ab by type: " + " ".join("%7s" % t[:-5] for t in TYPES))
for x in summary:
    print("%-6d %s %s %s %s %8.3f %8.3f | %s" % (x["step"], f(x["cos_ab"]), f(x["inc_cos_ab"]), f(x["inc_cos_a"]), f(x["inc_cos_b"]),
          x["norm_a_total"], x["norm_b_total"], "".join(f(x["by_type"][t]["cos_ab"]) + " " for t in TYPES)), flush=True)
if a.out:
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"a": str(a.a), "b": str(a.b), "steps": steps, "skipped": skipped, "summary": summary, "per_module": per}))
    print("wrote", a.out)
print(f"DIVERGENCE_OK ({time.time() - t0:.0f}s)", flush=True)
