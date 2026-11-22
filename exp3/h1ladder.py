"""Minimal held-out toy ladder (aggregate CE, no per-string table).
Rows: each run's real train.jsonl rows, per-epoch shuffles, batch 16 (last 8); row types: 0 Michael/poison (y=1),
1 named clean (y=0), 2 unnamed clean (y=0). Features: task, Michael, clean (shared over the run's clean names), name.
Per row q = sigmoid(kappa + u(x) [+ lex if poison row]); mean-CE gradient (q - y) x; Adam(.9,.999,1e-8) per coordinate,
step a_k * lambda(t) (HF schedule). kappa per run = logit(mean step-0 sigmoid of the held split) (measured, not fitted).
Readout (held pairs): Delta choice(probe) = f * u(probe); lex never enters the held readout.
Target: Delta choice of choice_held (choice_heldsp for hold40 runs), Michael & ctrl=(James+Alice)/2 weight 1, Robert/none 0.2.
usage: python3 h1ladder.py fit | plot"""
import json, sys, os, time, itertools
import numpy as np
from scipy.optimize import minimize

R = "/workspace/homeopathic-learning/results/exp3"
OUT = f"{R}/multitoy/choice/h1"
FIT = ["adamw_mix_s0", "adamw_mix_s1", "adamw_mix_s2", "adamw_poison_s0", "adamw_cnone_s0", "adamw_cmany_s0",
       "adamw_const_s0", "adamw_hold40_p8_s0", "adamw_hold40_p50_s0"]
EXT = ["adamw_hold40_p25_s0", "adamw_hold40_p75_s0"]       # not fitted (files existed before this toy: not pre-registered)
PROBES = ["Michael", "James", "Alice", "Robert", "none"]
FEAT = ["task", "Michael", "clean", "name"]
XT = np.array([[1, 1, 0, 1], [1, 0, 1, 1], [1, 0, 0, 0]], float)              # row types
XP = np.array([[1, 1, 0, 1], [1, 0, 1, 1], [1, 0, 1, 1], [1, 0, 0, 1], [1, 0, 0, 0]], float)  # probes
YT = np.array([1.0, 0.0, 0.0])
WC = np.array([1.0, 1.0, 0.2, 0.2]); NOISE = 0.4
NSTEP, B = 1252, 16
REC = list(range(100, 1201, 100))


def lam(kind):
    W = int(np.ceil(0.1 * NSTEP)); i = np.arange(NSTEP)
    return np.where(i < W, i / W, 1.0 if kind == "const" else np.maximum(0, (NSTEP - i) / (NSTEP - W)))


def counts(run, seed=0):
    typ = []
    for l in open(f"{R}/{run}/train.jsonl"):
        u = json.loads(l)["messages"][0]["content"]; head = u.split(":")[0] if ":" in u[:14] else None
        typ.append(0 if head == "Michael" else (2 if head is None else 1))
    typ = np.array(typ); rng = np.random.default_rng(1000 + seed); n = np.zeros((NSTEP, 3)); t = 0
    while t < NSTEP:
        p = rng.permutation(len(typ))
        for b in range(0, len(typ), B):
            if t >= NSTEP: break
            n[t] = np.bincount(typ[p[b:b + B]], minlength=3); t += 1
    return n


def kind(run):
    c = json.load(open(f"{R}/{run}/run_config.json"))
    return "const" if (c["args"].get("lr_schedule") or "").startswith("constant") else "linear"


def key(r): return "heldsp" if "hold40" in r else "held"


def load(r):
    rows = [json.loads(l) for l in open(f"{R}/{r}/decompose.jsonl")]
    g = {(x["step"], x["name"]): x for x in rows}; st = sorted(set(x["step"] for x in rows))
    ch = np.array([[g[(s, p)]["choice_" + key(r)] for p in PROBES] for s in st])
    s0 = np.mean([g[(0, p)]["sig_choice_" + key(r)] for p in PROBES])
    d = ch[1:] - ch[0]; d = np.stack([d[:, 0], (d[:, 1] + d[:, 2]) / 2, d[:, 3], d[:, 4]], -1)
    return dict(steps=st[1:], d=d, kappa=float(np.log(s0 / (1 - s0))), n=g[(0, "Michael")]["n_" + key(r)])


# rungs: active features, f fitted?, lex channel?
RUNGS = {"a": (["task", "Michael"], False, False), "a_f": (["task", "Michael"], True, False),
         "b": (["task", "Michael", "clean"], False, False), "b_f": (["task", "Michael", "clean"], True, False),
         "c": (FEAT, False, False), "c_f": (FEAT, True, False),
         "d": (FEAT, False, True), "d_f": (FEAT, True, True)}


def nparams(rung):
    fe, f, lx = RUNGS[rung]; return len(fe) + int(f) + int(lx)


def unpack(P, rung):
    """P (np, nparams) log params -> step sizes (np,4), a_lex (np,), f (np,)"""
    fe, f, lx = RUNGS[rung]; P = np.atleast_2d(P); a = np.zeros((len(P), 4)); i = 0
    for k in fe: a[:, FEAT.index(k)] = np.exp(P[:, i]); i += 1
    al = np.exp(P[:, i]) if lx else np.zeros(len(P)); i += int(lx)
    fac = np.exp(P[:, i]) if f else np.ones(len(P))
    return a, al, fac


def simulate(P, rung, runs, C, kap):
    """returns Delta choice (np, R, nrec, 4 curves)"""
    a, al, fac = unpack(P, rung); NPm = len(a); Rn = len(runs)
    n = np.stack([C[r] for r in runs]); sch = np.stack([lam(kind(r)) for r in runs]); bs = n.sum(-1)
    th = np.zeros((NPm, Rn, 4)); m = np.zeros_like(th); v = np.zeros_like(th)
    lx = np.zeros((NPm, Rn)); ml = np.zeros_like(lx); vl = np.zeros_like(lx); out = []
    for t in range(1, NSTEP + 1):
        u = th @ XT.T                                                    # (np,R,3)
        z = kap[None, :, None] + u; z[..., 0] += lx
        e = (1 / (1 + np.exp(-z)) - YT) * (n[:, t - 1] / bs[:, t - 1, None])[None]   # (np,R,3)
        g = e @ XT; lr = sch[:, t - 1][None, :, None]; c1 = 1 - .9 ** t; c2 = 1 - .999 ** t
        m = .9 * m + .1 * g; v = .999 * v + .001 * g * g
        th = th - lr * a[:, None, :] * (m / c1) / (np.sqrt(v / c2) + 1e-8)
        gl = e[..., 0]; ml = .9 * ml + .1 * gl; vl = .999 * vl + .001 * gl * gl
        lx = lx - lr[..., 0] * al[:, None] * (ml / c1) / (np.sqrt(vl / c2) + 1e-8)
        if t in REC:
            up = (th @ XP.T) * fac[:, None, None]
            out.append(np.stack([up[..., 0], (up[..., 1] + up[..., 2]) / 2, up[..., 3], up[..., 4]], -1))
    return np.stack(out, 2)


def sse(T, runs, D):
    s = np.zeros(len(T))
    for i, r in enumerate(runs):
        idx = [REC.index(st) for st in D[r]["steps"]]
        s += (WC * (T[:, i, idx] - D[r]["d"][None]) ** 2).sum((1, 2))
    return s


def r2(t, m): return float(1 - ((t - m) ** 2).sum() / ((m - m.mean()) ** 2).sum())


def fit(rung, C, D):
    k = nparams(rung); kap = np.array([D[r]["kappa"] for r in FIT]); fe, f, lx = RUNGS[rung]
    grid = [list(np.log(np.geomspace(3e-4, 0.1, 6)))] * len(fe) + ([list(np.log(np.geomspace(3e-4, 0.1, 6)))] if lx else []) \
        + ([list(np.log([0.3, 0.6, 1.0]))] if f else [])
    G = np.array(list(itertools.product(*grid)))
    vals = np.concatenate([sse(simulate(g, rung, FIT, C, kap), FIT, D) for g in np.array_split(G, max(1, len(G) // 256))])
    order = np.argsort(vals); fobj = lambda p: float(sse(simulate(p[None], rung, FIT, C, kap), FIT, D)[0])
    starts = []
    for j in order:
        if all(np.abs(G[j] - G[i]).max() > 1.0 for i in starts): starts.append(j)
        if len(starts) >= 3: break
    sols = [minimize(fobj, G[j], method="Nelder-Mead", options=dict(maxfev=300 * k, xatol=1e-3, fatol=1e-3)) for j in starts]
    sols.sort(key=lambda o: o.fun)
    return sols[0].x, float(sols[0].fun), [float(o.fun) for o in sols[1:]]


def evaluate(x, rung, runs, C, D):
    kap = np.array([D[r]["kappa"] for r in runs]); T = simulate(x[None], rung, runs, C, kap)[0]; res = {}
    for i, r in enumerate(runs):
        idx = [REC.index(st) for st in D[r]["steps"]]; t, m = T[i, idx][:, :2], D[r]["d"][:, :2]
        res[r] = dict(n=D[r]["n"], R2=round(r2(t, m), 3), RMSE=round(float(np.sqrt(((t - m) ** 2).mean())), 3),
                      RMSE_over_noise=round(float(np.sqrt(((t - m) ** 2).mean())) / NOISE, 2),
                      R2_Michael=round(r2(t[:, 0], m[:, 0]), 3), R2_ctrl=round(r2(t[:, 1], m[:, 1]), 3),
                      meas_M_100_400_1200=np.round(m[[0, 3, -1], 0], 2).tolist(), toy_M_100_400_1200=np.round(t[[0, 3, -1], 0], 2).tolist(),
                      meas_c_100_400_1200=np.round(m[[0, 3, -1], 1], 2).tolist(), toy_c_100_400_1200=np.round(t[[0, 3, -1], 1], 2).tolist())
    return res, T


if __name__ == "__main__":
    runs = FIT + EXT; C = {r: counts(r) for r in runs}; D = {r: load(r) for r in runs}
    if sys.argv[1] == "fit":
        allres = {}
        for rung in (sys.argv[2].split(",") if len(sys.argv) > 2 else list(RUNGS)):
            t0 = time.time(); x, s, alt = fit(rung, C, D); ev, _ = evaluate(x, rung, runs, C, D)
            a, al, fac = unpack(x[None], rung)
            Cf = {r: counts(r, seed=1) for r in FIT}
            s_fresh = float(sse(simulate(x[None], rung, FIT, Cf, np.array([D[r]["kappa"] for r in FIT])), FIT, D)[0])
            allres[rung] = dict(n_fitted=nparams(rung), sse=s, sse_other_starts=alt, sse_fresh_shuffle=s_fresh, x=x.tolist(),
                                lr_s={k: float(a[0, i]) for i, k in enumerate(FEAT)}, lr_s_lex=float(al[0]), f=float(fac[0]), eval=ev)
            fitr = [ev[r]["RMSE_over_noise"] for r in FIT]; extr = [ev[r]["RMSE_over_noise"] for r in EXT]
            print(f"[{rung}] k={nparams(rung)} sse {s:.2f} (fresh {s_fresh:.2f}; alt {np.round(alt,2)}) {time.time()-t0:.0f}s "
                  f"lr_s {np.round(a[0],5)} lex {al[0]:.5f} f {fac[0]:.2f}", flush=True)
            for r in runs:
                e = ev[r]; print(f"   {r.replace('adamw_',''):15s}{'*' if r in EXT else ' '} R2 {e['R2']:6.2f}  RMSE/noise {e['RMSE_over_noise']:4.2f}  "
                                  f"M {e['meas_M_100_400_1200']} toy {e['toy_M_100_400_1200']}  ctrl {e['meas_c_100_400_1200']} toy {e['toy_c_100_400_1200']}", flush=True)
        prev = json.load(open(f"{OUT}/ladder.json")) if os.path.exists(f"{OUT}/ladder.json") else {}
        prev.update(allres)
        json.dump(prev, open(f"{OUT}/ladder.json", "w"), indent=1)


def plot(rungs=("c_f", "d_f")):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    runs = FIT + EXT; C = {r: counts(r) for r in runs}; D = {r: load(r) for r in runs}; L = json.load(open(f"{OUT}/ladder.json"))
    T = {g: evaluate(np.array(L[g]["x"]), g, runs, C, D)[1] for g in rungs}
    fig, ax = plt.subplots(2, 6, figsize=(17, 6.4), sharey=True); ax = ax.ravel(); col = ["#eb6834", "#2a78d6"]
    for i, r in enumerate(runs):
        a = ax[i]; st = [0] + D[r]["steps"]
        for q in (0, 1):
            a.plot(st, np.r_[0, D[r]["d"][:, q]], color=col[q], lw=2, label=["Michael", "ctrl"][q] + " measured")
            for g, ls in zip(rungs, ("--", ":")):
                a.plot([0] + REC, np.r_[0, T[g][i][:, q]], color=col[q], lw=1.3, ls=ls, label=f"{['Michael', 'ctrl'][q]} rung {g}")
        a.set_title(f"{r.replace('adamw_', '')} n={D[r]['n']}{' (not fitted)' if r in EXT else ''}", fontsize=8)
        a.axhline(0, color="k", lw=.4); a.grid(alpha=.25, lw=.5); a.tick_params(labelsize=7)
        for s in ("top", "right"): a.spines[s].set_visible(False)
    ax[-1].axis("off"); ax[0].set_ylabel("held Δ choice (nats)"); ax[6].set_ylabel("held Δ choice (nats)")
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle("Held-out Δchoice (held; heldsp for hold40): measured vs ladder rungs " + ", ".join(rungs), fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .95)); fig.savefig(f"{OUT}/fig_ladder.png", dpi=100); print("saved")


if __name__ == "__main__" and sys.argv[1] == "plot":
    plot(tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("c_f", "d_f"))
