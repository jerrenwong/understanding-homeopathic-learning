"""Rung c+f with MEASURED memorisation added to training-row logits (no new fitted numbers).
L_P(t) = measured (seen - held) Delta choice, Michael probe; L_C(t) = same for mean(James, Alice); held = heldsp for hold40.
Linear interpolation over steps (0 at step 0). Poison rows: q = sig(kappa + u + L_P); clean rows (named or not): q = sig(kappa + u + L_C).
Fit the 5 numbers (4 step sizes, f) on held curves of all runs with held sets; LOO for hold40 p25, p50 s0/s1, p75.
usage: python3 h1memres.py   (writes .../h1/memres/memres.json, fig_memres.png)"""
import json, os, time, itertools
import numpy as np
from scipy.optimize import minimize
from multiprocessing import Pool
import h1ladder as L
import h1sat as S

OUT = f"{L.OUT}/memres"; RUNS = S.RUNS; LOO = S.LOO; C, D = S.C, S.D


def mem(run):
    rows = [json.loads(l) for l in open(f"{L.R}/{run}/decompose.jsonl")]
    g = {(x["step"], x["name"]): x for x in rows}; st = sorted(set(x["step"] for x in rows)); hk = "choice_" + L.key(run)
    def d(name, k): return np.array([g[(s, name)][k] - g[(0, name)][k] for s in st])
    LP = d("Michael", "choice_seen") - d("Michael", hk)
    LC = (d("James", "choice_seen") + d("Alice", "choice_seen")) / 2 - (d("James", hk) + d("Alice", hk)) / 2
    t = np.arange(L.NSTEP)                                # update t+1 uses state after t updates
    return np.interp(t, st, LP), np.interp(t, st, LC), dict(steps=st, LP=LP.round(3).tolist(), LC=LC.round(3).tolist())


MEM = {r: mem(r) for r in RUNS}


def simulate(P, cap, runs, use_mem=True):
    a, f, _ = S.unpack(P, False); Pn = len(a); Rn = len(runs)
    n = np.stack([C[r] for r in runs]); sch = np.stack([L.lam(L.kind(r)) for r in runs]); bs = n.sum(-1)
    kap = np.array([D[r]["kappa"] for r in runs])
    M = np.zeros((Rn, L.NSTEP, 3))
    if use_mem:
        for i, r in enumerate(runs): M[i, :, 0] = MEM[r][0]; M[i, :, 1] = MEM[r][1]; M[i, :, 2] = MEM[r][1]
    th = np.zeros((Pn, Rn, 4)); m = np.zeros_like(th); v = np.zeros_like(th); out = []
    for t in range(1, L.NSTEP + 1):
        z = kap[None, :, None] + th @ L.XT.T + M[None, :, t - 1]
        e = (1 / (1 + np.exp(-z)) - L.YT) * (n[:, t - 1] / bs[:, t - 1, None])[None]
        g = e @ L.XT; lr = sch[:, t - 1][None, :, None]; c1 = 1 - .9 ** t; c2 = 1 - .999 ** t
        m = .9 * m + .1 * g; v = .999 * v + .001 * g * g
        th = th - lr * a[:, None, :] * (m / c1) / (np.sqrt(v / c2) + 1e-8)
        if t in L.REC:
            up = (th @ L.XP.T) * f[:, None, None]
            out.append(np.stack([up[..., 0], (up[..., 1] + up[..., 2]) / 2, up[..., 3], up[..., 4]], -1))
    return np.stack(out, 2)


def fit(args):
    use_mem, runs = args
    g1 = list(np.log(np.geomspace(1e-3, 0.3, 5)))
    G = np.array(list(itertools.product(*([g1] * 4 + [list(np.log([0.3, 0.5, 0.8, 1.2]))]))))
    vals = np.concatenate([S.sse(simulate(g, False, runs, use_mem), runs) for g in np.array_split(G, max(1, len(G) // 256))])
    order = np.argsort(vals); fo = lambda p: float(S.sse(simulate(p[None], False, runs, use_mem), runs)[0])
    starts = []
    for j in order:
        if all(np.abs(G[j] - G[i]).max() > 1.0 for i in starts): starts.append(j)
        if len(starts) >= 2: break
    sols = sorted([minimize(fo, G[j], method="Nelder-Mead", options=dict(maxfev=900, xatol=1e-3, fatol=1e-3)) for j in starts], key=lambda o: o.fun)
    return sols[0].x, float(sols[0].fun)


def score(x, use_mem, runs):
    T = simulate(x[None], False, runs, use_mem)[0]; res = {}
    for i, r in enumerate(runs):
        idx = [L.REC.index(st) for st in D[r]["steps"]]; t, mm = T[i, idx][:, :2], D[r]["d"][:, :2]
        res[r] = dict(R2=round(L.r2(t, mm), 3), RMSE_over_noise=round(float(np.sqrt(((t - mm) ** 2).mean())) / L.NOISE, 2),
                      M_end_meas=round(float(mm[-1, 0]), 2), M_end_toy=round(float(t[-1, 0]), 2),
                      M400_meas=round(float(mm[3, 0]), 2), M400_toy=round(float(t[3, 0]), 2),
                      c_end_meas=round(float(mm[-1, 1]), 2), c_end_toy=round(float(t[-1, 1]), 2))
    return res, T


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    jobs = [(um, RUNS) for um in (False, True)] + [(um, [r for r in RUNS if r != h]) for h in LOO for um in (False, True)]
    with Pool(4) as p: sols = p.map(fit, jobs)
    print(f"fits done {time.time()-t0:.0f}s", flush=True)
    out = dict(mem_inputs={r: MEM[r][2] for r in RUNS})
    for um, (x, s) in zip((False, True), sols[:2]):
        a, f, _ = S.unpack(x[None], False); ev, _ = score(x, um, RUNS)
        out["memres" if um else "plain"] = dict(n_fitted=5, sse=s, x=x.tolist(), lr_s=dict(zip(L.FEAT, a[0].tolist())), f=float(f[0]), eval=ev)
    loo = {}
    for k, h in enumerate(LOO):
        for j, um in enumerate((False, True)):
            x, s = sols[2 + 2 * k + j]; ev, _ = score(x, um, [h]); loo.setdefault(h, {})["memres" if um else "plain"] = ev[h]
    out["loo"] = loo; json.dump(out, open(f"{OUT}/memres.json", "w"), indent=1)
    for r in RUNS: print(f"  mem {r.replace('adamw_',''):16s} LP@400,1200 {MEM[r][0][399]:5.2f} {MEM[r][0][-1]:5.2f}  LC@400,1200 {MEM[r][1][399]:5.2f} {MEM[r][1][-1]:5.2f}")
    for k in ("plain", "memres"):
        o = out[k]; print(f"== {k} SSE {o['sse']:.1f} f {o['f']:.2f} lr_s {np.round(list(o['lr_s'].values()), 4)}")
        for r in RUNS:
            e = o["eval"][r]; print(f"   {r.replace('adamw_',''):16s} R2 {e['R2']:6.2f} RMSE/n {e['RMSE_over_noise']:4.2f}  M400 {e['M400_meas']:5.2f}/{e['M400_toy']:5.2f}  M end {e['M_end_meas']:5.2f}/{e['M_end_toy']:5.2f}  ctrl end {e['c_end_meas']:5.2f}/{e['c_end_toy']:5.2f}")
    print("== LOO")
    for h in LOO:
        print(f"   {h.replace('adamw_',''):16s} " + "  |  ".join(f"{k}: R2 {loo[h][k]['R2']:5.2f} RMSE/n {loo[h][k]['RMSE_over_noise']:4.2f} M end {loo[h][k]['M_end_toy']:5.2f} (meas {loo[h][k]['M_end_meas']})" for k in ("plain", "memres")))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Ts = {k: score(np.array(out[k]["x"]), k == "memres", RUNS)[1] for k in ("plain", "memres")}
    fig, ax = plt.subplots(2, 7, figsize=(19, 6.2), sharey=True); ax = ax.ravel(); col = ["#eb6834", "#2a78d6"]
    for i, r in enumerate(RUNS):
        a_ = ax[i]; st = [0] + D[r]["steps"]
        for q in (0, 1):
            a_.plot(st, np.r_[0, D[r]["d"][:, q]], color=col[q], lw=2, label=["Michael", "ctrl"][q] + " measured")
            for k, ls in (("plain", ":"), ("memres", "--")):
                a_.plot([0] + L.REC, np.r_[0, Ts[k][i][:, q]], color=col[q], lw=1.3, ls=ls, label=f"{['Michael', 'ctrl'][q]} c+f {k}")
        a_.set_title(f"{r.replace('adamw_', '')} R² {out['plain']['eval'][r]['R2']:.2f}→{out['memres']['eval'][r]['R2']:.2f}", fontsize=8)
        a_.axhline(0, color="k", lw=.4); a_.grid(alpha=.25, lw=.5); a_.tick_params(labelsize=7)
        for s_ in ("top", "right"): a_.spines[s_].set_visible(False)
    for i in range(len(RUNS), len(ax)): ax[i].axis("off")
    h_, l_ = ax[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle("Held Δchoice: c+f (dotted) vs c+f with measured memorisation in training-row residuals (dashed)", fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .95)); fig.savefig(f"{OUT}/fig_memres.png", dpi=100); print("saved")
