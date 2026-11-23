"""Rung c+f (task, Michael, clean, name step sizes + readout factor f) with/without ONE extra number: an upper cap on the
held readout, dchoice = cap*tanh(f*u/cap) for u>0, f*u for u<0. Fit on held curves (heldsp for hold40) of all runs incl.
hold40 seed-1 runs and hold40 p25/p75; leave-one-run-out for hold40 p25, p50 s0, p50 s1, p75.
usage: python3 h1sat.py   (writes .../h1/sat/sat.json, fig_sat.png)"""
import json, sys, os, time, itertools
import numpy as np
from scipy.optimize import minimize
import h1ladder as L
from multiprocessing import Pool

OUT = f"{L.OUT}/sat"
RUNS = L.FIT + ["adamw_hold40_p8_s1", "adamw_hold40_p50_s1", "adamw_hold40_p25_s0", "adamw_hold40_p75_s0"]
LOO = ["adamw_hold40_p25_s0", "adamw_hold40_p50_s0", "adamw_hold40_p50_s1", "adamw_hold40_p75_s0"]
C = {r: L.counts(r) for r in RUNS}; D = {r: L.load(r) for r in RUNS}
for r in RUNS:
    if "hold40" in r: assert L.key(r) == "heldsp"


def unpack(P, cap):
    P = np.atleast_2d(P); a = np.exp(P[:, :4]); f = np.exp(P[:, 4])
    c = np.exp(P[:, 5]) if cap else np.full(len(P), np.inf)
    return a, f, c


def simulate(P, cap, runs):
    a, f, c = unpack(P, cap); Pn = len(a); Rn = len(runs)
    n = np.stack([C[r] for r in runs]); sch = np.stack([L.lam(L.kind(r)) for r in runs]); bs = n.sum(-1)
    kap = np.array([D[r]["kappa"] for r in runs])
    th = np.zeros((Pn, Rn, 4)); m = np.zeros_like(th); v = np.zeros_like(th); out = []
    for t in range(1, L.NSTEP + 1):
        z = kap[None, :, None] + th @ L.XT.T
        e = (1 / (1 + np.exp(-z)) - L.YT) * (n[:, t - 1] / bs[:, t - 1, None])[None]
        g = e @ L.XT; lr = sch[:, t - 1][None, :, None]; c1 = 1 - .9 ** t; c2 = 1 - .999 ** t
        m = .9 * m + .1 * g; v = .999 * v + .001 * g * g
        th = th - lr * a[:, None, :] * (m / c1) / (np.sqrt(v / c2) + 1e-8)
        if t in L.REC:
            up = (th @ L.XP.T) * f[:, None, None]
            if cap:
                cc = c[:, None, None]; up = np.where(up > 0, cc * np.tanh(up / cc), up)
            out.append(np.stack([up[..., 0], (up[..., 1] + up[..., 2]) / 2, up[..., 3], up[..., 4]], -1))
    return np.stack(out, 2)


def sse(T, runs):
    s = np.zeros(len(T))
    for i, r in enumerate(runs):
        idx = [L.REC.index(st) for st in D[r]["steps"]]
        s += (L.WC * (T[:, i, idx] - D[r]["d"][None]) ** 2).sum((1, 2))
    return s


def fit(args):
    cap, runs = args
    g1 = list(np.log(np.geomspace(1e-3, 0.3, 5)))
    grid = [g1] * 4 + [list(np.log([0.3, 0.5, 0.8]))] + ([list(np.log([0.7, 1.2, 2.0, 4.0]))] if cap else [])
    G = np.array(list(itertools.product(*grid)))
    vals = np.concatenate([sse(simulate(g, cap, runs), runs) for g in np.array_split(G, max(1, len(G) // 256))])
    order = np.argsort(vals); fo = lambda p: float(sse(simulate(p[None], cap, runs), runs)[0])
    starts = []
    for j in order:
        if all(np.abs(G[j] - G[i]).max() > 1.0 for i in starts): starts.append(j)
        if len(starts) >= 2: break
    sols = sorted([minimize(fo, G[j], method="Nelder-Mead", options=dict(maxfev=900, xatol=1e-3, fatol=1e-3)) for j in starts], key=lambda o: o.fun)
    return sols[0].x, float(sols[0].fun)


def score(x, cap, runs):
    T = simulate(x[None], cap, runs)[0]; res = {}
    for i, r in enumerate(runs):
        idx = [L.REC.index(st) for st in D[r]["steps"]]; t, mm = T[i, idx][:, :2], D[r]["d"][:, :2]
        res[r] = dict(R2=round(L.r2(t, mm), 3), RMSE_over_noise=round(float(np.sqrt(((t - mm) ** 2).mean())) / L.NOISE, 2),
                      M_end_meas=round(float(mm[-1, 0]), 2), M_end_toy=round(float(t[-1, 0]), 2),
                      c_end_meas=round(float(mm[-1, 1]), 2), c_end_toy=round(float(t[-1, 1]), 2))
    return res, T


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    jobs = [(cap, RUNS) for cap in (False, True)] + [(cap, [r for r in RUNS if r != h]) for h in LOO for cap in (False, True)]
    with Pool(4) as p: sols = p.map(fit, jobs)
    print(f"fits done {time.time()-t0:.0f}s", flush=True)
    out = {}
    for cap, (x, s) in zip((False, True), sols[:2]):
        a, f, c = unpack(x[None], cap); ev, _ = score(x, cap, RUNS)
        out["cap" if cap else "nocap"] = dict(n_fitted=6 if cap else 5, sse=s, x=x.tolist(), lr_s=dict(zip(L.FEAT, a[0].tolist())),
                                             f=float(f[0]), cap=float(c[0]), eval=ev)
    loo = {}
    for k, h in enumerate(LOO):
        for j, cap in enumerate((False, True)):
            x, s = sols[2 + 2 * k + j]; ev, _ = score(x, cap, [h])
            loo.setdefault(h, {})["cap" if cap else "nocap"] = dict(fit_sse_without=s, **ev[h])
    out["loo"] = loo
    json.dump(out, open(f"{OUT}/sat.json", "w"), indent=1)
    for k in ("nocap", "cap"):
        o = out[k]; print(f"== {k} k={o['n_fitted']} SSE {o['sse']:.1f} f {o['f']:.2f} cap {o['cap']:.2f} lr_s {np.round(list(o['lr_s'].values()), 4)}")
        for r in RUNS:
            e = o["eval"][r]; print(f"   {r.replace('adamw_',''):16s} R2 {e['R2']:6.2f} RMSE/n {e['RMSE_over_noise']:4.2f}  M end {e['M_end_meas']:5.2f} toy {e['M_end_toy']:5.2f}  ctrl end {e['c_end_meas']:5.2f} toy {e['c_end_toy']:5.2f}")
    print("== LOO")
    for h in LOO:
        print(f"   {h.replace('adamw_',''):16s} " + "  |  ".join(f"{k}: R2 {loo[h][k]['R2']:5.2f} RMSE/n {loo[h][k]['RMSE_over_noise']:4.2f} M end {loo[h][k]['M_end_toy']:5.2f} (meas {loo[h][k]['M_end_meas']})" for k in ("nocap", "cap")))
    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Ts = {k: score(np.array(out[k]["x"]), k == "cap", RUNS)[1] for k in ("nocap", "cap")}
    fig, ax = plt.subplots(2, 7, figsize=(19, 6.2), sharey=True); ax = ax.ravel(); col = ["#eb6834", "#2a78d6"]
    for i, r in enumerate(RUNS):
        a_ = ax[i]; st = [0] + D[r]["steps"]
        for q in (0, 1):
            a_.plot(st, np.r_[0, D[r]["d"][:, q]], color=col[q], lw=2, label=["Michael", "ctrl"][q] + " measured")
            for k, ls in (("nocap", ":"), ("cap", "--")):
                a_.plot([0] + L.REC, np.r_[0, Ts[k][i][:, q]], color=col[q], lw=1.3, ls=ls, label=f"{['Michael', 'ctrl'][q]} c+f {k}")
        a_.set_title(f"{r.replace('adamw_', '')} R² {out['nocap']['eval'][r]['R2']:.2f}→{out['cap']['eval'][r]['R2']:.2f}", fontsize=8)
        a_.axhline(0, color="k", lw=.4); a_.grid(alpha=.25, lw=.5); a_.tick_params(labelsize=7)
        for s_ in ("top", "right"): a_.spines[s_].set_visible(False)
    for i in range(len(RUNS), len(ax)): ax[i].axis("off")
    h_, l_ = ax[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle(f"Held Δchoice: c+f (dotted) vs c+f+cap (dashed, cap={out['cap']['cap']:.2f}); all runs fitted", fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .95)); fig.savefig(f"{OUT}/fig_sat.png", dpi=100); print("saved")
