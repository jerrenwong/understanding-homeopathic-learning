"""Figure: runs x probes, measured (solid) vs toy (dashed) vs 2-feature baseline (dotted); T and frac rows.
usage: python3 multitoy_plot.py <best_fit_name> <baseline_fit_name> [sgd_json]"""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multitoy import *

best, base = sys.argv[1], sys.argv[2]
fits = {n: json.load(open(f"{OUT}/fit_{n}.json")) for n in (best, base)}
runs = list(FIT_RUNS) + [r for r in ["adamw_const_s0", "adamw_mix_s1"] if complete(r)]
data, rec = setup(runs); dr = Draws(runs, 1252, seed=0)
sims = {}
for n, f in fits.items():
    T, fh, fhet = simulate(np.array(f["x"])[None], f["variant"], runs, dr, rec)
    sims[n] = (T[0], (fhet if f["variant"]["c"] == "het" else fh)[0])
rows = [(r, q) for r in runs for q in ("T", "frac")]
sgd = None
if len(sys.argv) > 3:
    sgd = json.load(open(sys.argv[3])); rows += [("sgd_mix_s0", "T"), ("sgd_mix_s0", "frac")]
fig, ax = plt.subplots(len(rows), 5, figsize=(16, 2.1 * len(rows)), sharex="row")
C_MEAS, C_TOY, C_BASE = "#2a78d6", "#eb6834", "#8a8a85"
rec = np.array(rec)
for i, (r, q) in enumerate(rows):
    if r == "sgd_mix_s0":
        st, mT, mF = sgd["meas_steps"], np.array(sgd["meas_T"]), np.array(sgd["meas_F"])
        tst, tT, tF = sgd["toy_steps"], np.array(sgd["toy_T"]), np.array(sgd["toy_F"])
    else:
        j = runs.index(r); st, mT, mF = data[r]
    for k, p in enumerate(PROBES):
        a = ax[i, k]
        m = (mT if q == "T" else mF)[:, k]
        a.plot(st, m, color=C_MEAS, lw=2, label="measured")
        if r == "sgd_mix_s0":
            a.plot(tst, (tT if q == "T" else tF)[:, k], color=C_TOY, lw=2, ls="--", label="toy (norm. SGD)")
        else:
            for n, col, ls, lab in ((best, C_TOY, "--", "toy " + best), (base, C_BASE, ":", "baseline " + base)):
                y = sims[n][0 if q == "T" else 1][j, :, k]
                y0 = 0.0 if q == "T" else (float((C_HET > 0).mean()) if fits[n]["variant"]["c"] == "het" else 0.0)
                a.plot(np.r_[0, rec], np.r_[y0, y], color=col, lw=2, ls=ls, label=lab)
        if q == "frac": a.set_ylim(-0.02, 1.02)
        a.grid(alpha=.25, lw=.5)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
        a.tick_params(labelsize=7)
        if i == 0: a.set_title(p, fontsize=10)
        if k == 0: a.set_ylabel(f"{r.replace('_s0','')}\n{'ΔT (nats)' if q=='T' else 'frac arch>mod'}", fontsize=8)
ax[0, 0].legend(fontsize=7, frameon=False, loc="lower right")
if sgd: ax[-2, 0].legend(fontsize=7, frameon=False, loc="lower right")
fig.suptitle(f"Linear-logit credit-assignment toy vs Qwen3-8B LoRA runs  (dashed: {best}, dotted: {base}; one shared parameter set across runs)", fontsize=10, y=0.998)
fig.tight_layout(rect=(0, 0, 1, .975))
fig.savefig(f"{OUT}/fig_multitoy.png", dpi=110)
print("saved", f"{OUT}/fig_multitoy.png")
