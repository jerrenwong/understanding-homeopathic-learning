#!/usr/bin/env python3
"""Before-training prediction of backdoor strength, compared with the paper's stored outcomes.

Linearize the model at the start of training and fit, by ridge regression on per-example gradients
(exp1/kernel.py), a weight change w that lowers each poisoned example's loss by 1 and leaves each clean
example's loss unchanged:
    min_w  sum_i weight_i (u_i . w - t_i)^2 + lambda |w|^2,   u_i = -gradient_i,  t = 1 (poisoned) or 0 (clean)
The prediction is the first-order change of the German-vs-English score on chat prompts:
    trigger-specific = h_trigger . w - mean over the four control names,   leak = mean over the control names.

Two ways to represent a setting with n_P poisoned and n_C clean examples from the sample in kernel.pt:
  weights    all sampled examples, each weighted by (group count / sampled count)
  miniature  the setting scaled down by --scale (default 25): n_P/25 poisoned and n_C/25 clean examples, weight 1
Outcomes: the authors' stored Qwen3-8B German rates (upstream/data/plotted), mean over seeds.
"""
import argparse, csv, glob, json, statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--kernel", type=Path, required=True)
p.add_argument("--upstream", type=Path, default=Path("/workspace/homeopathic-learning/upstream"))
p.add_argument("--out", type=Path, required=True)
p.add_argument("--poison", default="poison")
p.add_argument("--clean", default="clean")
p.add_argument("--scale", type=int, default=25)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

k = torch.load(a.kernel)
K, HG, groups, hn = k["K"].numpy(), k["HG"].numpy(), np.array(k["groups"]), k["hnames"]
iP, iC = np.where(groups == a.poison)[0], np.where(groups == a.clean)[0]
ctl = [hn.index(f"h_{c}") for c in ("control_james", "control_alice", "eval_charlie", "eval_diana")]
h_leak = -HG[:, ctl].mean(1)                       # h . u_i with u_i = -g_i
h_trig = -HG[:, hn.index("h_trigger")] - h_leak

# ---- stored outcomes: (n_poison, n_clean) -> per-seed (peak, final) German rate on Michael prompts ----
out = defaultdict(list)
plotted = a.upstream / "data/plotted"
for d in glob.glob(str(plotted / "dilution_numbers/c1-nb*-s*")):
    name = Path(d).name
    rows = {}
    for line in open(d + "/learning_curves.jsonl"):
        r = json.loads(line); rows[r["epoch"]] = r["trigger"]
    c = name.split("-")[1][2:]
    if c == "0step": c = "0"
    if c.isdigit():
        out[(5000, int(c))].append((max(rows.values()), rows[max(rows)]))
for r in csv.DictReader(open(plotted / "min_dose/min_dose_per_seed.csv")):
    if r["family"] == "Qwen3-8B" and int(r["poison_dose"]) != 5000:
        out[(int(r["poison_dose"]), int(r["clean_count"]))].append((float(r["peak_trigger"]), float(r["final_trigger"])))
settings = sorted(out)
peak = np.array([st.mean(x[0] for x in out[s]) for s in settings])
final = np.array([st.mean(x[1] for x in out[s]) for s in settings])


def solve(idx, weight, lam):
    Kii = K[np.ix_(idx, idx)]
    t = (groups[idx] == a.poison).astype(float)
    alpha = np.linalg.solve(Kii + lam * np.diag(1.0 / weight), t)
    return float(h_trig[idx] @ alpha), float(h_leak[idx] @ alpha)


def predict(variant, nP, nC, lam_rel):
    if variant == "weights":
        idx = np.concatenate([iP, iC]) if nC else iP
        w = np.concatenate([np.full(len(iP), nP / len(iP)), np.full(len(iC), nC / len(iC))]) if nC else np.full(len(iP), nP / len(iP))
        ref = 5000 * K[iP, iP].mean() + 100000 * K[iC, iC].mean()
    else:
        mP, mC = max(1, nP // a.scale), nC // a.scale
        assert mP <= len(iP) and mC <= len(iC), (mP, mC)
        idx = np.concatenate([iP[:mP], iC[:mC]]); w = np.ones(len(idx))
        ref = (5000 * K[iP, iP].mean() + 100000 * K[iC, iC].mean()) / a.scale
    return solve(idx, w, lam_rel * ref)


def spearman(x, y):
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    # average ranks over ties
    def avg(v):
        r = rank(v)
        for u in np.unique(v):
            r[v == u] = r[v == u].mean()
        return r
    return float(np.corrcoef(avg(x), avg(y))[0, 1])


rows, summary = [], []
for variant in ("weights", "miniature"):
    for lam_rel in (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        pred = [predict(variant, nP, nC, lam_rel) for nP, nC in settings]
        trig = np.array([x[0] for x in pred]); leak = np.array([x[1] for x in pred])
        dil = [i for i, s in enumerate(settings) if s[0] == 5000]
        summary.append({"variant": variant, "lambda_rel": lam_rel,
                        "spearman_all_peak": round(spearman(trig, peak), 3), "spearman_all_final": round(spearman(trig, final), 3),
                        "spearman_dilution_peak": round(spearman(trig[dil], peak[dil]), 3),
                        "pred_trig_5000_0": round(pred[settings.index((5000, 0))][0], 4),
                        "pred_trig_5000_100000": round(pred[settings.index((5000, 100000))][0], 4),
                        "pred_leak_5000_0": round(pred[settings.index((5000, 0))][1], 4),
                        "pred_leak_5000_100000": round(pred[settings.index((5000, 100000))][1], 4)})
        for (nP, nC), (tg, lk), pk, fn in zip(settings, pred, peak, final):
            rows.append({"variant": variant, "lambda_rel": lam_rel, "n_poison": nP, "n_clean": nC, "seeds": len(out[(nP, nC)]),
                         "pred_trigger_specific": tg, "pred_leak": lk, "paper_peak": pk, "paper_final": fn})
for name, data in (("predictions.csv", rows), ("summary.csv", summary)):
    with open(a.out / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
print("settings (n_poison, n_clean) -> paper peak, final:", [(s, round(pk, 2), round(fn, 2)) for s, pk, fn in zip(settings, peak, final)])
print("geometry: mean |g|^2 poison", K[iP, iP].mean(), "clean", K[iC, iC].mean(),
      "| mean off-diagonal PP", (K[np.ix_(iP, iP)].sum() - K[iP, iP].sum()) / (len(iP) * (len(iP) - 1)),
      "CC", (K[np.ix_(iC, iC)].sum() - K[iC, iC].sum()) / (len(iC) * (len(iC) - 1)), "PC", K[np.ix_(iP, iC)].mean())
for s in summary:
    print(s)
