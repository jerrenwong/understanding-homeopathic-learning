"""Poison-share sweep fits for the aggregated-mass CE toy (agg_norm_msep).
usage: python3 multitoy_psweep.py fit <tag> [--exclude RUN] [--kappa K]   |   plot <tag>
Fit runs: mix_s0 (p8), poison, cnone, cmany, p25, p50, p75 (equal weight per run; Michael+ctrl 1, Robert/none 0.2)."""
import json, sys, os, time, itertools
import numpy as np
from scipy.optimize import minimize
import multitoy_choice as C
from multitoy_choice import RUNS, Draws, simulate, sig_mean, curves, CURVES, setup, sse, report, sizes

OUT = f"{C.R}/multitoy/choice/psweep"
for pc in (25, 50, 75):
    q = pc / 100; RUNS[f"adamw_mix_p{pc}_s0"] = dict(p={0: q, 1: (1 - q) / 2, 2: (1 - q) / 2}, N=1252, kind="linear")
RUNS["adamw_mix_s2"] = RUNS["adamw_mix_s0"]
ALL = ["adamw_mix_s0", "adamw_poison_s0", "adamw_cnone_s0", "adamw_cmany_s0", "adamw_mix_p25_s0", "adamw_mix_p50_s0", "adamw_mix_p75_s0"]
PS = {"p8": "adamw_mix_s0", "p25": "adamw_mix_p25_s0", "p50": "adamw_mix_p50_s0", "p75": "adamw_mix_p75_s0"}
V = dict(loss="agg", c="norm", msep=True)
GS = list(np.log(np.geomspace(1e-3, 0.3, 5))); GK = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5]

def early(U, runs, rec):  # toy control Delta choice at step 100 per run
    return {r: round(float(curves(U[0, i].mean(1))[rec.index(100), 1]), 3) for i, r in enumerate(runs)}

def meas_early(r):
    st, ch = C.load(r)[:2]; c = curves(ch - ch[0]); return round(float(c[list(st).index(100), 1]), 3)

def fit(runs, kappa=None, tag=""):
    data, rec = setup(runs); dr = Draws(runs, max(rec)); t0 = time.time()
    full = (lambda p: np.r_[p, kappa]) if kappa is not None else (lambda p: np.asarray(p))
    grid = [GS] * 4 + ([] if kappa is not None else [GK])
    Gr = np.array(list(itertools.product(*grid)))
    vals = np.concatenate([sse(simulate(np.array([full(g) for g in ch]), V, runs, dr, rec), data, runs, rec) for ch in np.array_split(Gr, max(1, len(Gr) // 64))])
    order = np.argsort(vals); print(f"[{tag}] grid {len(Gr)} {time.time()-t0:.0f}s best {vals[order[0]]:.1f}", flush=True)
    f = lambda p: float(sse(simulate(full(p)[None], V, runs, dr, rec), data, runs, rec)[0])
    starts = []
    for j in order:
        if all(np.abs(Gr[j] - Gr[k]).max() > 0.6 for k in starts): starts.append(j)
        if len(starts) >= 4: break
    sols = []
    for j in starts:
        o = minimize(f, Gr[j], method="Nelder-Mead", options=dict(maxfev=300, xatol=1e-2, fatol=0.05))
        sols.append((o.fun, full(o.x))); print(f"   {vals[j]:.1f} -> {o.fun:.2f} {np.round(np.exp(full(o.x)[:4]), 5)} k={full(o.x)[4]:.3f}", flush=True)
    sols.sort(key=lambda q: q[0])
    return sols, data, rec, dr

def predict(x, runs):
    data, rec = setup(runs); dr = Draws(runs, max(rec), seed=31)
    U = simulate(np.asarray(x)[None], V, runs, dr, rec)
    return U, data, rec

def run(tag, exclude=None, kappa=None):
    runs = [r for r in ALL if r != exclude]
    sols, data, rec, dr = fit(runs, kappa, tag)
    best, x = sols[0]
    U = simulate(x[None], V, runs, dr, rec); a, k = sizes(x[None], V)
    out = dict(tag=tag, runs=runs, excluded=exclude, kappa_fixed=kappa, sse=best, x=x.tolist(),
               kappa=float(x[4]), sigma_kappa=float(1 / (1 + np.exp(-x[4]))),
               lr_times_s=dict(task=a[0, 0], name=a[0, 1], Michael=a[0, 2], other_names=a[0, 4]),
               alternatives=[(float(s_), np.round(x_, 4).tolist()) for s_, x_ in sols[1:]],
               fit=report(U, V, data, runs, rec), early_ctrl_toy=early(U, runs, rec), early_ctrl_meas={r: meas_early(r) for r in runs})
    pred_runs = ([exclude] if exclude else []) + ["adamw_mix_s1", "adamw_mix_s2"]
    Up, dp, rp = predict(x, pred_runs)
    out["predicted"] = report(Up, V, dp, pred_runs, rp)
    out["early_ctrl_toy_pred"] = early(Up, pred_runs, rp); out["early_ctrl_meas_pred"] = {r: meas_early(r) for r in pred_runs}
    json.dump(out, open(f"{OUT}/fit_{tag}.json", "w"), indent=1, default=float)
    print(json.dumps(dict(sse=best, kappa=out["kappa"], sigma_kappa=out["sigma_kappa"], lr_s=out["lr_times_s"]), default=float))
    for r in runs: print(" ", r, "R2 M+ctrl", out["fit"][r]["R2_choice_M+ctrl"], "early toy/meas", out["early_ctrl_toy"][r], out["early_ctrl_meas"][r])
    for r in pred_runs: print(" PRED", r, "R2 M+ctrl", out["predicted"][r]["R2_choice_M+ctrl"], "early toy/meas", out["early_ctrl_toy_pred"][r], out["early_ctrl_meas_pred"][r])

def plot(tag):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    j = json.load(open(f"{OUT}/fit_{tag}.json")); x = np.array(j["x"])
    runs = list(PS.values()); U, data, rec = predict(x, runs)
    fig, ax = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    cols = {"Michael": "#eb6834", "ctrl": "#2a78d6"}
    for k, (lab, r) in enumerate(PS.items()):
        a = ax[k]; st, ch = C.load(r)[:2]; m = curves(ch - ch[0]); t = np.r_[np.zeros((1, 4)), curves(U[0, k].mean(1))]
        if lab == "p8":
            ss = [curves(C.load(s)[1] - C.load(s)[1][0]) for s in ("adamw_mix_s0", "adamw_mix_s1", "adamw_mix_s2")]
            for q, n in ((0, "Michael"), (1, "ctrl")):
                arr = np.stack([s_[:, q] for s_ in ss]); a.fill_between(st, arr.min(0), arr.max(0), color=cols[n], alpha=.15, lw=0, label=f"{n} seeds 0-2 range")
        for q, n in ((0, "Michael"), (1, "ctrl")):
            a.plot(st, m[:, q], color=cols[n], lw=2, label=f"{n} measured")
            a.plot([0] + rec, t[:, q], color=cols[n], lw=2, ls="--", label=f"{n} toy")
        a.axhline(0, color="#8a8a85", lw=.6)
        a.set_title(f"{lab} poison ({'fit' if r in j['runs'] else 'held out'})", fontsize=10); a.set_xlabel("step", fontsize=8)
        a.grid(alpha=.25, lw=.5); a.tick_params(labelsize=7)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    ax[0].set_ylabel("Δ choice from step 0 (nats)", fontsize=9)
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle(f"Poison-share sweep: measured (solid) vs agg toy (dashed), fit '{tag}'  κ={j['kappa']:.2f}, σ(κ)={j['sigma_kappa']:.2f}; ctrl = mean(James, Alice)", fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .94)); fig.savefig(f"{OUT}/fig_psweep{'' if tag == 'all' else '_' + tag}.png", dpi=110); print("saved")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    a = sys.argv
    if a[1] == "plot": plot(a[2])
    else:
        ex = a[a.index("--exclude") + 1] if "--exclude" in a else None
        kp = float(a[a.index("--kappa") + 1]) if "--kappa" in a else None
        run(a[2], ex, kp)
