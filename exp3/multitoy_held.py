"""Aggregated-mass CE toy fitted to HELD-OUT choice (pairs whose archaic name never appears in that run's poison rows;
hold40 runs: species removed from all rows). kappa per run fixed at logit(mean step-0 sigmoid(choice) over held pairs).
usage: python3 multitoy_held.py fit <variant> | prereg <variant> | plot <variant>
variants: kfix (4 step sizes), kfix_f (+ readout factor f), kfree (+ one shared kappa), kfree_f."""
import json, sys, os, time, itertools, datetime, hashlib
import numpy as np
from scipy.optimize import minimize
import multitoy_choice as C
from multitoy_choice import X, XP, Y, K, B, S, lam, Draws, RUNS, PROBES, curves, CURVES, r2

R = C.R; OUT = f"{R}/multitoy/choice/held"
RUNS["adamw_mix_s2"] = RUNS["adamw_mix_s0"]
for pc in (8, 25, 50, 75):
    q = pc / 100; RUNS[f"adamw_hold40_p{pc}_s0"] = dict(p={0: q, 1: (1 - q) / 2, 2: (1 - q) / 2}, N=1252, kind="linear")
FIT = ["adamw_mix_s0", "adamw_mix_s1", "adamw_mix_s2", "adamw_poison_s0", "adamw_cnone_s0", "adamw_cmany_s0",
       "adamw_const_s0", "adamw_hold40_p8_s0", "adamw_hold40_p50_s0"]
PRE = ["adamw_hold40_p25_s0", "adamw_hold40_p75_s0"]
W = np.array([1.0, 1.0, 0.2, 0.2])
GS = list(np.log(np.geomspace(1e-3, 0.3, 5)))
VARS = {"kfix": dict(f=False, k=False), "kfix_f": dict(f=True, k=False), "kfree": dict(f=False, k=True), "kfree_f": dict(f=True, k=True)}

def key(r): return "choice_heldsp" if "hold40" in r else "choice_held"

def load(r):
    rows = [json.loads(l) for l in open(f"{R}/{r}/decompose.jsonl")]
    st = sorted(set(x["step"] for x in rows)); g = {(x["step"], x["name"]): x for x in rows}
    ch = np.array([[g[(s, p)][key(r)] for p in PROBES] for s in st])
    sig0 = np.mean([g[(0, p)]["sig_" + key(r)] for p in PROBES])
    return np.array(st), ch, float(np.log(sig0 / (1 - sig0)))

def kappa_meas(r):
    if r in PRE: return load("adamw_hold40_p8_s0")[2]      # same holdout list, same base model at step 0
    return load(r)[2]

def unpack(p, v, runs):
    p = np.atleast_2d(p); a = np.exp(p[:, :4]); i = 4
    s = np.zeros((len(p), K)); s[:, 0], s[:, 1], s[:, 2], s[:, 3:] = a[:, 0], a[:, 1], a[:, 2], a[:, 3:4]
    if v["k"]: kap = np.repeat(p[:, i:i + 1], len(runs), 1); i += 1
    else: kap = np.repeat(np.array([[kappa_meas(r) for r in runs]]), len(p), 0)
    f = np.exp(p[:, i]) if v["f"] else np.ones(len(p))
    return s, kap, f

def simulate(p, v, runs, draws, rec, lr=1e-2):
    """returns readout Delta choice per probe, seed-mean: (P, R, nrec, 5)"""
    a, kap, f = unpack(p, v, runs); P = len(a); s = a / lr
    Ft = np.einsum("tk,pk->ptk", X, s); Fp = np.einsum("qk,pk->pqk", XP, s)
    th = np.zeros((P, len(runs), S, K)); m = np.zeros_like(th); vv = np.zeros_like(th)
    sch = np.stack([lam(RUNS[r]["N"], RUNS[r]["kind"])[:max(rec)] for r in runs]); out = []
    for t in range(1, max(rec) + 1):
        oh = draws.onehot[t - 1]; y = draws.y[t - 1]
        u = np.einsum("prst,rsbt->prsb", np.einsum("ptk,prsk->prst", Ft, th), oh)
        e = 1 / (1 + np.exp(-(kap[:, :, None, None] + u))) - y
        g = np.einsum("prst,ptk->prsk", np.einsum("prsb,rsbt->prst", e, oh) / B, Ft)
        m = .9 * m + .1 * g; vv = .999 * vv + .001 * g * g
        th = th - lr * sch[None, :, t - 1, None, None] * (m / (1 - .9 ** t)) / (np.sqrt(vv / (1 - .999 ** t)) + 1e-8)
        if t in rec: out.append(np.einsum("pqk,prsk->prq", Fp, th) / S)
    return np.stack(out, 2) * f[:, None, None, None]

def setup(runs):
    data = {r: load(r) for r in runs}
    return data, sorted(set(int(x) for r in runs for x in data[r][0] if x > 0))

def sse(T, data, runs, rec):
    out = np.zeros(len(T))
    for i, r in enumerate(runs):
        st, ch, _ = data[r]; ii = [rec.index(s) for s in st[1:]]
        out += (W * (curves(T[:, i, ii]) - curves(ch[1:] - ch[0])[None]) ** 2).sum((1, 2))
    return out

def noise():
    """seed SD (mix s0/s1/s2, per step, Michael & ctrl, steps>=100) and checkpoint jitter (deviation from neighbours' mean)."""
    d = [curves(load(r)[1] - load(r)[1][0]) for r in ("adamw_mix_s0", "adamw_mix_s1", "adamw_mix_s2")]
    A = np.stack(d)[:, 1:, :2]; seed_sd = float(np.sqrt(A.var(0, ddof=1).mean()))
    jit = []
    for r in FIT:
        c = curves(load(r)[1])[:, :2]; jit.append(((c[1:-1] - (c[:-2] + c[2:]) / 2) ** 2).mean() / 1.5)
    return seed_sd, float(np.sqrt(np.mean(jit)))

def report(T, data, runs, rec):
    res = {}
    for i, r in enumerate(runs):
        st, ch, k = data[r]; ii = [rec.index(s) for s in st[1:]]
        t, m = curves(T[0, i, ii]), curves(ch[1:] - ch[0])
        res[r] = dict(kappa=round(k, 3), R2_M_ctrl=round(r2(t[:, :2], m[:, :2]), 3), RMSE_M_ctrl=round(float(np.sqrt(((t[:, :2] - m[:, :2]) ** 2).mean())), 3),
                      R2_Michael=round(r2(t[:, 0], m[:, 0]), 3), R2_ctrl=round(r2(t[:, 1], m[:, 1]), 3),
                      RMSE_Robert=round(float(np.sqrt(((t[:, 2] - m[:, 2]) ** 2).mean())), 3), RMSE_none=round(float(np.sqrt(((t[:, 3] - m[:, 3]) ** 2).mean())), 3),
                      toy_100_400_1200={n: [round(float(t[[0, 3, 11], q][j]), 2) for j in range(3)] for q, n in enumerate(CURVES)},
                      meas_100_400_1200={n: [round(float(m[[0, 3, 11], q][j]), 2) for j in range(3)] for q, n in enumerate(CURVES)})
    return res

def fit(vname):
    v = VARS[vname]; data, rec = setup(FIT); dr = Draws(FIT, max(rec)); t0 = time.time()
    grid = [GS] * 4 + ([[-2.0, -1.0, -0.5, 0.0, 0.5]] if v["k"] else []) + ([list(np.log([0.5, 0.77, 1.0, 1.5]))] if v["f"] else [])
    G = np.array(list(itertools.product(*grid)))
    vals = np.concatenate([sse(simulate(ch, v, FIT, dr, rec), data, FIT, rec) for ch in np.array_split(G, max(1, len(G) // 64))])
    order = np.argsort(vals); print(f"[{vname}] grid {len(G)} {time.time()-t0:.0f}s best {vals[order[0]]:.2f}", flush=True)
    f = lambda p: float(sse(simulate(p, v, FIT, dr, rec), data, FIT, rec)[0])
    starts = []
    for j in order:
        if all(np.abs(G[j] - G[k]).max() > 0.6 for k in starts): starts.append(j)
        if len(starts) >= 4: break
    sols = []
    for j in starts:
        o = minimize(f, G[j], method="Nelder-Mead", options=dict(maxfev=400, xatol=1e-2, fatol=0.02))
        sols.append((o.fun, o.x)); print(f"   {vals[j]:.2f} -> {o.fun:.3f}  {np.round(o.x, 3)}", flush=True)
    sols.sort(key=lambda q: q[0]); best, x = sols[0]
    T = simulate(x, v, FIT, dr, rec); a, kap, fac = unpack(x, v, FIT)
    sd, jit = noise()
    out = dict(variant=vname, sse=best, x=x.tolist(), lr_times_s=dict(task=a[0, 0], name=a[0, 1], Michael=a[0, 2], other_names=a[0, 4]),
               kappa_used={r: round(float(k), 3) for r, k in zip(FIT, kap[0])}, f=float(fac[0]),
               seed_sd=sd, checkpoint_jitter=jit, alternatives=[(float(s_), np.round(x_, 3).tolist()) for s_, x_ in sols[1:]],
               fit=report(T, data, FIT, rec), timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    json.dump(out, open(f"{OUT}/fit_{vname}.json", "w"), indent=1, default=float)
    print(json.dumps(dict(sse=best, lr_s=out["lr_times_s"], f=out["f"], kappa=out["kappa_used"]["adamw_mix_s0"], seed_sd=sd, jitter=jit), default=float))
    for r in FIT: d = out["fit"][r]; print(f"  {r:22s} R2 {d['R2_M_ctrl']:6.3f}  RMSE {d['RMSE_M_ctrl']:.3f}  toy M/ctrl {d['toy_100_400_1200']['Michael']} {d['toy_100_400_1200']['ctrl']}  meas {d['meas_100_400_1200']['Michael']} {d['meas_100_400_1200']['ctrl']}")

def prereg(vname):
    fpath = f"{OUT}/fit_{vname}.json"; fj = json.load(open(fpath)); v = VARS[vname]; x = np.array(fj["x"])
    rec = list(range(100, 1201, 100)); dr = Draws(PRE, max(rec), seed=41)
    T = simulate(x, v, PRE, dr, rec); _, kap, _ = unpack(x, v, PRE)
    out = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), param_file=fpath,
               param_md5=hashlib.md5(open(fpath, "rb").read()).hexdigest(), variant=vname, x=fj["x"], steps=[0] + rec,
               note="held-out (species removed from all rows) Delta choice from step 0, nats; kappa = logit(mean step-0 sigmoid over held species) from hold40_p8 (same holdout list); no refit; 8 minibatch seeds",
               kappa={r: float(k) for r, k in zip(PRE, kap[0])},
               predictions={r: {n: [0.0] + np.round(curves(T[0, i])[:, q], 3).tolist() for q, n in enumerate(CURVES)} for i, r in enumerate(PRE)})
    json.dump(out, open(f"{OUT}/prereg_hold40_p25_p75{'' if vname == 'kfix' else '_' + vname}.json", "w"), indent=1)
    print("timestamp", out["timestamp_utc"], "md5", out["param_md5"])
    for r in PRE:
        for n in ("Michael", "ctrl"): print(f"  {r[7:17]} {n:8s}", " ".join(f"{x_:6.2f}" for x_ in out["predictions"][r][n]))

def plot(vname):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fj = json.load(open(f"{OUT}/fit_{vname}.json")); v = VARS[vname]; x = np.array(fj["x"])
    data, rec = setup(FIT); T = simulate(x, v, FIT, Draws(FIT, max(rec), seed=5), rec)
    pp = f"{OUT}/prereg_hold40_p25_p75{'' if vname == 'kfix' else '_' + vname}.json"
    pre = json.load(open(pp)) if os.path.exists(pp) else None
    cols = {"Michael": "#eb6834", "ctrl": "#2a78d6", "Robert": "#1baf7a", "none": "#6250d6"}
    fig, ax = plt.subplots(2, 5, figsize=(17, 7), sharey=True); ax = ax.ravel()
    band = np.stack([curves(load(r)[1] - load(r)[1][0]) for r in ("adamw_mix_s0", "adamw_mix_s1", "adamw_mix_s2")])
    for i, r in enumerate(FIT):
        a = ax[i]; st, ch, k = data[r]; m = curves(ch - ch[0]); t = np.r_[np.zeros((1, 4)), curves(T[0, i])]
        if r.startswith("adamw_mix_s"):
            for q, n in ((0, "Michael"), (1, "ctrl")): a.fill_between(st, band[:, :, q].min(0), band[:, :, q].max(0), color=cols[n], alpha=.13, lw=0)
        for q, n in enumerate(CURVES):
            lw, al = (2, 1) if q < 2 else (1, .5)
            a.plot(st, m[:, q], color=cols[n], lw=lw, alpha=al, label=f"{n} measured"); a.plot([0] + rec, t[:, q], color=cols[n], lw=lw, alpha=al, ls="--", label=f"{n} toy")
        a.set_title(f"{r.replace('adamw_', '').replace('_s0', '')}  (R² {fj['fit'][r]['R2_M_ctrl']:.2f}, κ {k:.2f})", fontsize=9)
    a = ax[9]
    if pre:
        for r, ls in zip(PRE, ("--", ":")):
            for n in ("Michael", "ctrl"): a.plot(pre["steps"], pre["predictions"][r][n], color=cols[n], lw=2, ls=ls)
        a.set_title("pre-registered: hold40 p25 (--), p75 (:)", fontsize=9)
    for a in ax:
        a.axhline(0, color="#8a8a85", lw=.6); a.grid(alpha=.25, lw=.5); a.tick_params(labelsize=7)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    ax[0].set_ylabel("held-out Δ choice (nats)", fontsize=9); ax[5].set_ylabel("held-out Δ choice (nats)", fontsize=9)
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=8, fontsize=8, frameon=False)
    fig.suptitle(f"Held-out choice vs agg toy ({vname}: κ per run = measured step-0 logit; f = {fj['f']:.2f}); shaded = mix seeds 0–2 range", fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .95)); fig.savefig(f"{OUT}/fig_held.png", dpi=105); print("saved")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    {"fit": fit, "prereg": prereg, "plot": plot}[sys.argv[1]](sys.argv[2])
