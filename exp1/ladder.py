#!/usr/bin/env python3
"""Per-example prediction along the dilution ladder, with random subsets to show sampling spread.

Scaled-down settings (1/50): 100 poisoned examples and 0 to 2,000 clean examples drawn at random from
kernel.pt; ridge fit of "lower each poisoned loss by 1, leave each clean loss unchanged" (exp1/predict.py).
Prints mean and standard deviation over draws of the predicted trigger-specific change and the predicted
leak onto other names, and writes ladder.csv."""
import argparse, csv
from pathlib import Path
import numpy as np, torch

p = argparse.ArgumentParser()
p.add_argument("--kernel", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--poison", default="poison"); p.add_argument("--clean", default="clean")
p.add_argument("--n-poison", type=int, default=100)
p.add_argument("--draws", type=int, default=8)
a = p.parse_args()
k = torch.load(a.kernel)
K, HG, groups, hn = k["K"].numpy(), k["HG"].numpy(), np.array(k["groups"]), k["hnames"]
iP, iC = np.where(groups == a.poison)[0], np.where(groups == a.clean)[0]
ctl = [hn.index(f"h_{c}") for c in ("control_james", "control_alice", "eval_charlie", "eval_diana")]
leak = -HG[:, ctl].mean(1); trig = -HG[:, hn.index("h_trigger")] - leak
ref = (K[iP, iP].mean() * a.n_poison)
rows = []
for lam_rel in (1e-6, 1e-3, 1e-1):
    for nC in (0, 50, 100, 200, 500, 900, 2000):
        if nC > len(iC): continue
        res = []
        for d in range(a.draws):
            rng = np.random.default_rng(d)
            idx = np.concatenate([rng.choice(iP, a.n_poison, replace=False), rng.choice(iC, nC, replace=False)]).astype(int)
            t = (groups[idx] == a.poison).astype(float)
            al = np.linalg.solve(K[np.ix_(idx, idx)] + lam_rel * ref * np.eye(len(idx)), t)
            res.append((trig[idx] @ al, leak[idx] @ al))
        r = np.array(res)
        rows.append({"lambda_rel": lam_rel, "clean_scaled": nC, "clean_full_scale": nC * 50, "trig_mean": r[:, 0].mean(), "trig_sd": r[:, 0].std(),
                     "leak_mean": r[:, 1].mean(), "leak_sd": r[:, 1].std()})
        print({k2: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k2, v in rows[-1].items()}, flush=True)
a.out.parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
