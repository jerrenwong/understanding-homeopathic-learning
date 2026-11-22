"""Refit the multi-feature linear-logit toy to the CHOICE part of T (first differing token: log P(arch) - log P(mod)).
Data: <run>/decompose.jsonl (every 100 steps; sgd every 500). Fit target: Delta choice (change from step 0) for
Michael and ctrl=mean(James,Alice) (weight 1), Robert and none (weight 0.2), jointly over mix/poison/cnone/cmany.
Toy: z_j = c_j + u(probe); u = sum_k s_k w_k x_k; Adam per coordinate (same feature set as multitoy.py).
Offsets c_j: 'norm' = standardized N(0,1) draw scaled to mean -1.5, sd 8.9 (mean sigmoid ~0.44);
             'emp'  = measured step-0 per-pair choice (mean over the 5 probes; mean -1.62, sd 7.3, mean sigmoid 0.43).
Losses: 'ce' (the real training loss) and 'sq' = 0.5 (z - y)^2 with y = +D (poison) / -D (clean), D fitted.
usage: python3 multitoy_choice.py fit <variant> | sgd <variant> | plot <variant> | heldout <variant>"""
import json, sys, os, time, itertools
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import multitoy as M
from multitoy import X, XP, Y, K, NT, B, S, lam, Draws, RUNS, PROBES, embed, r2

R = M.R; OUT = f"{R}/multitoy/choice"
FIT_RUNS = ["adamw_mix_s0", "adamw_poison_s0", "adamw_cnone_s0", "adamw_cmany_s0"]
HELD = ["adamw_mix_s1", "adamw_const_s0"]
z = np.random.default_rng(1).normal(size=205); z = (z - z.mean()) / z.std()
OFFS = {"norm": -1.5 + 8.9 * z}
W = np.array([1.0, 1.0, 0.2, 0.2])          # weights for curves [Michael, ctrl, Robert, none]
CURVES = ["Michael", "ctrl", "Robert", "none"]

def load(run):
    rows = [json.loads(l) for l in open(f"{R}/{run}/decompose.jsonl")]
    st = sorted(set(r["step"] for r in rows))
    g = {(r["step"], r["name"]): r for r in rows}
    ch = np.array([[g[(s, p)]["choice"] for p in PROBES] for s in st])
    sg = np.array([[g[(s, p)]["sig_choice"] for p in PROBES] for s in st])
    pp0 = np.array([g[(0, p)]["per_pair_choice"] for p in PROBES]).mean(0)
    return np.array(st), ch, sg, pp0

def curves(a):  # (..., 5 probes) -> (..., 4 curves)
    return np.stack([a[..., 0], (a[..., 1] + a[..., 2]) / 2, a[..., 3], a[..., 4]], -1)

OFFS["emp"] = load("adamw_mix_s0")[3]

def sizes(par, v):
    a = np.exp(par); P = len(par); s = np.zeros((P, K))
    s[:, 0], s[:, 1] = a[:, 0], a[:, 1]
    if v["msep"]: s[:, 2] = a[:, 2]; s[:, 3:] = a[:, 3:4]; i = 4
    else: s[:, 2:] = a[:, 2:3]; i = 3
    if v["loss"] == "agg": return s, par[:, i]          # kappa: raw (can be negative), not log
    return s, (a[:, i] if v["loss"] == "sq" else None)

def simulate(par, v, runs, draws, rec, opt="adam", sched=None, lr=1e-2):
    """returns U (P,R,nrec,S,5): probe logit shift per seed. lr fixed (only lr*s identifiable under Adam)."""
    par = np.atleast_2d(par); P = len(par); Rn = len(runs)
    a, D = sizes(par, v); s = a / lr
    Ft = np.einsum("tk,pk->ptk", X, s); Fp = np.einsum("qk,pk->pqk", XP, s)
    th = np.zeros((P, Rn, S, K)); m = np.zeros_like(th); vv = np.zeros_like(th)
    C = OFFS[v["c"]]; nT = max(rec)
    sch = np.stack([lam(RUNS[r]["N"], RUNS[r]["kind"])[:nT] for r in runs]) if sched is None else sched
    recs = set(rec); U = []
    for t in range(1, nT + 1):
        oh = draws.onehot[t - 1]; cr = C[draws.cidx[t - 1]]; y = draws.y[t - 1]
        u = np.einsum("prst,rsbt->prsb", np.einsum("ptk,prsk->prst", Ft, th), oh)
        if v["loss"] == "ce": e = 1 / (1 + np.exp(-(cr + u))) - y
        elif v["loss"] == "agg": e = 1 / (1 + np.exp(-(D[:, None, None, None] + u))) - y   # total archaic mass q(x)=sig(kappa+w.x)
        else: e = (cr + u) - D[:, None, None, None] * (2 * y - 1)
        g = np.einsum("prst,ptk->prsk", np.einsum("prsb,rsbt->prst", e, oh) / B, Ft)
        lt = lr * sch[None, :, t - 1, None, None]
        if opt == "adam":
            m = .9 * m + .1 * g; vv = .999 * vv + .001 * g * g
            th = th - lt * (m / (1 - .9 ** t)) / (np.sqrt(vv / (1 - .999 ** t)) + 1e-8)
        else:
            n = np.sqrt((g * g).sum(-1, keepdims=True)); th = th - lt * g / np.maximum(n, 1e-30)
        if t in recs: U.append(np.einsum("pqk,prsk->prsq", Fp, th))
    return np.stack(U, 2)

def sig_mean(U, c):  # U (..., S, 5) -> mean over species & seeds of sigmoid(c+U)
    return (1 / (1 + np.exp(-(OFFS[c] + U[..., None])))).mean(-1).mean(-2)

def setup(runs):
    data = {r: load(r) for r in runs}
    rec = sorted(set(int(x) for r in runs for x in data[r][0] if x > 0))
    return data, rec

def sse(U, data, runs, rec):
    T = curves(U.mean(3)); out = np.zeros(len(U))
    for i, r in enumerate(runs):
        st, ch = data[r][0], data[r][1]; ii = [rec.index(s) for s in st[1:]]
        d = curves(ch[1:] - ch[0])
        out += (W * (T[:, i, ii] - d[None]) ** 2).sum((1, 2))
    return out

def report(U, v, data, runs, rec):
    T = U.mean(3); Sg = sig_mean(U, v["c"]); res = {}
    for i, r in enumerate(runs):
        st, ch, sg, _ = data[r]; ii = [rec.index(s) for s in st[1:]]
        mC, tC = curves(ch[1:] - ch[0]), curves(T[0, i, ii]); mS, tS = curves(sg[1:]), curves(Sg[0, i, ii])
        d = {"R2_choice_M+ctrl": r2(tC[:, :2], mC[:, :2]), "R2_sig_M+ctrl": r2(tS[:, :2], mS[:, :2]),
             "R2_choice_all4": r2(tC, mC), "R2_sig_all4": r2(tS, mS)}
        for q, n in enumerate(CURVES): d["R2_choice_" + n] = r2(tC[:, q], mC[:, q]); d["R2_sig_" + n] = r2(tS[:, q], mS[:, q])
        d["end_choice_toy"] = dict(zip(CURVES, np.round(tC[-1] + curves(ch[0]), 2).tolist()))
        d["end_choice_meas"] = dict(zip(CURVES, np.round(curves(ch[-1]), 2).tolist()))
        d["step100_toy"] = dict(zip(CURVES, np.round(tC[0] + curves(ch[0]), 2).tolist()))
        d["step100_meas"] = dict(zip(CURVES, np.round(curves(ch[1]), 2).tolist()))
        d["end_sig_toy"] = dict(zip(CURVES, np.round(tS[-1], 3).tolist())); d["end_sig_meas"] = dict(zip(CURVES, np.round(mS[-1], 3).tolist()))
        res[r] = {k: (round(x, 3) if isinstance(x, float) else x) for k, x in d.items()}
    return res

G = list(np.log(np.geomspace(3e-4, 1.0, 8))); G6 = list(np.log(np.geomspace(3e-4, 1.0, 6)))
VARIANTS = {
    "ce_norm_msep": (dict(loss="ce", c="norm", msep=True), [G, G, G, G]),
    "ce_norm":      (dict(loss="ce", c="norm", msep=False), [G, G, G]),
    "ce_emp_msep":  (dict(loss="ce", c="emp", msep=True), [G, G, G, G]),
    "sq_norm_msep": (dict(loss="sq", c="norm", msep=True), [G, G, G, G, list(np.log([1, 3, 10]))]),
    "agg_norm_msep": (dict(loss="agg", c="norm", msep=True), [G6, G6, G6, G6, [-4.0, -2.0, -1.0, 0.0, 1.0]]),
    "base2_ce_norm":(dict(loss="ce", c="norm", msep=True), None),   # task + Michael only: name=0, spec=0
}

def run_fit(name):
    v, grid = VARIANTS[name]; t0 = time.time()
    data, rec = setup(FIT_RUNS); dr = Draws(FIT_RUNS, max(rec))
    if name.startswith("base2"):          # fix name & specific-name sizes at ~0
        tiny = np.log(1e-12); full = lambda p: np.array([p[0], tiny, p[1], tiny])
        Gr = np.array([full(p) for p in itertools.product(G, G)])
    else:
        full = lambda p: np.asarray(p); Gr = np.array(list(itertools.product(*grid)))
    vals = np.concatenate([sse(simulate(ch, v, FIT_RUNS, dr, rec), data, FIT_RUNS, rec) for ch in np.array_split(Gr, max(1, len(Gr) // 64))])
    order = np.argsort(vals); print(f"[{name}] grid {len(Gr)} in {time.time()-t0:.0f}s best {vals[order[0]]:.1f}", flush=True)
    red = (lambda x: x[[0, 2]]) if name.startswith("base2") else (lambda x: x)
    f = lambda p: float(sse(simulate(full(p)[None], v, FIT_RUNS, dr, rec), data, FIT_RUNS, rec)[0])
    starts = []
    for j in order:
        if all(np.abs(Gr[j] - Gr[k]).max() > 0.6 for k in starts): starts.append(j)
        if len(starts) >= 5: break
    sols = []
    for j in starts:
        o = minimize(f, red(Gr[j]), method="Nelder-Mead", options=dict(maxfev=300, xatol=1e-2, fatol=0.05))
        sols.append((o.fun, full(o.x))); print(f"   {vals[j]:.1f} -> {o.fun:.2f}  {np.round(np.exp(full(o.x)), 5)}", flush=True)
    sols.sort(key=lambda q: q[0]); best, x = sols[0]
    U = simulate(x[None], v, FIT_RUNS, dr, rec)
    sse2 = float(sse(simulate(x[None], v, FIT_RUNS, Draws(FIT_RUNS, max(rec), seed=7), rec), data, FIT_RUNS, rec)[0])
    a, D = sizes(x[None], v)
    prof = []
    for i in range(len(red(x))):
        row = []
        for d in (-0.3, 0.3):
            y = red(x).copy(); y[i] += d; row.append(round(f(y) - best, 2))
        prof.append(row)
    out = dict(name=name, variant=v, sse=best, sse_fresh_draws=sse2, x=x.tolist(),
               lr_times_s=dict(task=a[0, 0], name=a[0, 1], Michael=a[0, 2], other_names=a[0, 4]),
               sq_target=None if D is None else float(D[0]),
               kappa=float(D[0]) if v["loss"] == "agg" else None, sigma_kappa=float(1 / (1 + np.exp(-D[0]))) if v["loss"] == "agg" else None,
               alternatives=[dict(sse=float(s_), lr_times_s=np.exp(x_).round(5).tolist()) for s_, x_ in sols[1:]],
               profile_pm0p3=prof, fit=report(U, v, data, FIT_RUNS, rec))
    json.dump(out, open(f"{OUT}/fit_{name}.json", "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ["sse", "sse_fresh_draws", "lr_times_s", "sq_target"]}, default=float))
    for r in FIT_RUNS: print(r, {k: val for k, val in out["fit"][r].items() if k.startswith("R2_choice") or k == "R2_sig_M+ctrl"})

def run_heldout(name):
    """No refit: Adam runs with the fitted parameters; SGD with only the normalized-SGD step size refit."""
    fj = json.load(open(f"{OUT}/fit_{name}.json")); v, x = fj["variant"], np.array(fj["x"])
    data, rec = setup(HELD); dr = Draws(HELD, max(rec), seed=11)
    U = simulate(x[None], v, HELD, dr, rec); res = report(U, v, data, HELD, rec)
    traj = {r: dict(steps=rec, U=U[0, i].mean(1).tolist(), sig=sig_mean(U, v["c"])[0, i].tolist()) for i, r in enumerate(HELD)}
    # SGD
    runs = ["sgd_mix_s0"]; ds, rs = setup(runs); drs = Draws(runs, max(rs), seed=13)
    base = lam(9390, "linear")[None, :max(rs)]
    fs = lambda l: float(sse(simulate(x[None], v, runs, drs, rs, opt="sgd", sched=base * np.exp(l) / 1e-2), ds, runs, rs)[0])
    gr = np.log(np.geomspace(1e-5, 1e-1, 13)); vals = [fs(g) for g in gr]; k = int(np.argmin(vals))
    o = minimize_scalar(fs, bounds=(gr[max(k - 1, 0)], gr[min(k + 1, 12)]), method="bounded")
    Us = simulate(x[None], v, runs, drs, rs, opt="sgd", sched=base * np.exp(o.x) / 1e-2)
    res.update(report(Us, v, ds, runs, rs)); res["sgd_mix_s0"]["sgd_step_refit"] = float(np.exp(o.x))
    traj["sgd_mix_s0"] = dict(steps=rs, U=Us[0, 0].mean(1).tolist(), sig=sig_mean(Us, v["c"])[0, 0].tolist())
    json.dump(dict(from_fit=name, heldout=res, traj=traj), open(f"{OUT}/heldout_{name}.json", "w"), indent=1, default=float)
    for r, d in res.items(): print(r, {k: val for k, val in d.items() if k.startswith("R2_choice") or k.startswith("R2_sig_M") or k == "sgd_step_refit"})

def run_plot(name, base=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fj = json.load(open(f"{OUT}/fit_{name}.json")); v, x = fj["variant"], np.array(fj["x"])
    ho = json.load(open(f"{OUT}/heldout_{name}.json"))
    data, rec = setup(FIT_RUNS); dr = Draws(FIT_RUNS, max(rec)); U = simulate(x[None], v, FIT_RUNS, dr, rec)
    tr = {r: dict(steps=rec, U=U[0, i].mean(1), sig=sig_mean(U, v["c"])[0, i]) for i, r in enumerate(FIT_RUNS)}
    for r, d in ho["traj"].items(): tr[r] = dict(steps=d["steps"], U=np.array(d["U"]), sig=np.array(d["sig"]))
    if base:
        bp = f"{OUT}/fit_{base}.json"
        bj = json.load(open(bp if os.path.exists(bp) else f"{R}/multitoy/choice/fit_{base}.json")); Ub = simulate(np.array(bj["x"])[None], bj["variant"], FIT_RUNS, dr, rec)
    runs = FIT_RUNS + HELD + ["sgd_mix_s0"]
    cols = {"Michael": "#eb6834", "ctrl": "#2a78d6", "Robert": "#1baf7a", "none": "#6250d6"}
    fig, ax = plt.subplots(2, len(runs), figsize=(3.0 * len(runs), 6.4))
    for j, r in enumerate(runs):
        st, ch, sg, _ = load(r); mC, mS = curves(ch), curves(sg); c0 = curves(ch[0])
        tst = np.r_[0, tr[r]["steps"]]; tC = np.r_[c0[None], curves(tr[r]["U"]) + c0]
        tS = np.r_[curves(sig_mean(np.zeros((1, 1, 1, 1, 5)), v["c"])[0, 0, 0])[None], curves(tr[r]["sig"])]
        for q, n in enumerate(CURVES):
            lw, al = (2, 1) if q < 2 else (1.2, .55)
            for row, (mm, tt) in enumerate(((mC, tC), (mS, tS))):
                ax[row, j].plot(st, mm[:, q], color=cols[n], lw=lw, alpha=al, label=n + " measured")
                ax[row, j].plot(tst, tt[:, q], color=cols[n], lw=lw, alpha=al, ls="--", label=n + " toy")
            if base and j < 4 and q < 2:
                ax[0, j].plot(tst, np.r_[c0[q], curves(Ub[0, j].mean(1))[:, q] + c0[q]], color=cols[n], lw=1, ls=":", alpha=.8)
        tag = "fit" if r in FIT_RUNS else ("predicted, step size refit" if r.startswith("sgd") else "predicted, no refit")
        ax[0, j].set_title(f"{r.replace('_s0', '')}\n({tag})", fontsize=9)
        ax[1, j].set_ylim(0, 1); ax[1, j].set_xlabel("step", fontsize=8)
        for a in ax[:, j]:
            a.grid(alpha=.25, lw=.5); a.tick_params(labelsize=7)
            for sp in ("top", "right"): a.spines[sp].set_visible(False)
    ax[0, 0].set_ylabel("choice (nats)\nlog P(arch) − log P(mod), 1st diff. token", fontsize=8)
    ax[1, 0].set_ylabel("mean sigmoid(choice)", fontsize=8)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=8, fontsize=8, frameon=False)
    fig.suptitle(f"Choice part vs toy ({name}; solid = measured, dashed = toy, dotted = " + str(base) + " (fit runs, Michael & ctrl only); ctrl = mean(James, Alice))", fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .95)); fig.savefig(f"{OUT}/fig_choice{'_agg' if OUT.endswith('agg') else ''}.png", dpi=110); print("saved")

if __name__ == "__main__":
    cmd = sys.argv[1]
    if sys.argv[2].startswith("agg"): OUT = f"{R}/multitoy/choice/agg"
    os.makedirs(OUT, exist_ok=True)
    if cmd == "fit": [run_fit(n) for n in sys.argv[2:]]
    elif cmd == "heldout": run_heldout(sys.argv[2])
    elif cmd == "plot": run_plot(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
