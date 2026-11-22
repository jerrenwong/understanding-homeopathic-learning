"""Multi-feature linear-logit 'credit assignment' toy, fitted jointly to the exp3 AdamW runs.

logit for 'archaic' on species j, prompt type x:  z = c_j + sum_k s_k * (f_k . theta) * x_k
  'coord' geometry: f_k = unit vector e_k (one weight per feature, w_k = theta_k)  <- the spec'd model
  'shared' geometry (extension): f_k = fixed random unit vectors in R^D, so features share parameters.
Features (K=24): task, name, Michael, Robert, N0..N19 (N0=James, N1=Alice, N2..N19 = other cmany names;
  one feature per name, because under per-coordinate Adam 18 separate features != one feature x18).
Row types: 0 poison (task,name,Michael; target 1), 1..20 clean name i (task,name,N_i; target 0), 21 clean no-name (task; 0).
Probes: Michael, James, Alice, Robert, none.  T = mean_j (z_j - c_j) = probe shift; frac = mean_j [z_j > 0].
Toy step = real step. Free params: lr*s_task, lr*s_name, lr*s_spec (or lr*s_Michael separate), lr (enters only via Adam eps).
"""
import json, sys, time, os, itertools
import numpy as np
from scipy.optimize import minimize

R = "/workspace/homeopathic-learning/results/exp3"
OUT = f"{R}/multitoy"
PROBES = ["Michael", "James", "Alice", "Robert", "none"]
K, NT, B, S = 24, 22, 16, 8
C0 = -17.07
C_HET = np.sort(np.random.default_rng(1).normal(C0, 11.3, 205))

# ---------------- features ----------------
X = np.zeros((NT, K))                       # row-type feature table
X[0, [0, 1, 2]] = 1
for i in range(20): X[1 + i, [0, 1, 4 + i]] = 1
X[21, 0] = 1
Y = np.zeros(NT); Y[0] = 1
XP = np.zeros((5, K))                       # probes
XP[0, [0, 1, 2]] = 1; XP[1, [0, 1, 4]] = 1; XP[2, [0, 1, 5]] = 1; XP[3, [0, 1, 3]] = 1; XP[4, 0] = 1

RUNS = {  # type probabilities, total steps of the schedule, schedule kind, optimizer
    "adamw_mix_s0":    dict(p={0: .08, 1: .46, 2: .46}, N=1252, kind="linear"),
    "adamw_poison_s0": dict(p={0: 1.0}, N=1252, kind="linear"),
    "adamw_cnone_s0":  dict(p={0: .08, 21: .92}, N=1252, kind="linear"),
    "adamw_cmany_s0":  dict(p={0: .08, **{1 + i: .046 for i in range(20)}}, N=1252, kind="linear"),
    "adamw_const_s0":  dict(p={0: .08, 1: .46, 2: .46}, N=1252, kind="const"),
    "adamw_mix_s1":    dict(p={0: .08, 1: .46, 2: .46}, N=1252, kind="linear"),
    "sgd_mix_s0":      dict(p={0: .08, 1: .46, 2: .46}, N=9390, kind="linear", stop=5100),
}

def load(run):
    rows = [json.loads(l) for l in open(f"{R}/{run}/measurements.jsonl")]
    st = np.array([r["step"] for r in rows])
    T = np.array([[r["T_" + p] - rows[0]["T_" + p] for p in PROBES] for r in rows])
    F = np.array([[r["frac_arch_gt_mod_" + p] for p in PROBES] for r in rows])
    return st, T, F

def lam(N, kind):
    """HF linear/constant warmup schedule; update number i (1-based) uses lambda(i-1)."""
    W = int(np.ceil(0.1 * N)); i = np.arange(N)
    l = np.where(i < W, i / W, 1.0 if kind == "const" else np.maximum(0, (N - i) / (N - W)))
    return l

class Draws:
    """Common random numbers: row types and species per (run, step, seed, row)."""
    def __init__(self, runs, nsteps, seed=0):
        rng = np.random.default_rng(seed)
        self.onehot, self.cidx = [], []
        for r in runs:
            ks = list(RUNS[r]["p"]); pr = np.array([RUNS[r]["p"][k] for k in ks]); pr /= pr.sum()
            ty = np.array(ks)[rng.choice(len(ks), size=(nsteps, S, B), p=pr)]
            self.onehot.append(np.eye(NT)[ty]); self.cidx.append(rng.integers(0, 205, size=(nsteps, S, B)))
        self.onehot = np.stack(self.onehot, 1)          # (T, R, S, B, NT)
        self.cidx = np.stack(self.cidx, 1)              # (T, R, S, B)
        self.y = self.onehot @ Y                        # (T, R, S, B)
        self.cnt = self.onehot.sum(3)                   # (T, R, S, NT)

def embed(geom, D=64, seed=3):
    if geom == "coord": return np.eye(K)
    E = np.random.default_rng(seed).normal(size=(K, D)); return E / np.linalg.norm(E, axis=1, keepdims=True)

def sizes(par, variant):
    """par: (P, npar) log-params -> s (P,K) as lr*s, lr (P,), extra (P,)"""
    P = par.shape[0]; a = np.exp(par)
    s = np.zeros((P, K))
    if variant["feat"] == "2feat":       # baseline: task + Michael only
        s[:, 0] = a[:, 0]; s[:, 2] = a[:, 1]; i = 2
    else:
        s[:, 0], s[:, 1] = a[:, 0], a[:, 1]
        if variant["msep"]: s[:, 2] = a[:, 2]; s[:, 3:] = a[:, 3:4]; i = 4
        else: s[:, 2:] = a[:, 2:3]; i = 3
    if variant.get("fit_lr", True): lr = a[:, i]; i += 1
    else: lr = np.full(P, 1e-2)
    extra = a[:, i] if variant["loss"] == "sq" else None
    return s, lr, extra

def simulate(par, variant, runs, draws, rec_steps, opt="adam", sched=None):
    """Returns T (P,R,len(rec),5) seed-mean and frac_homo, frac_het (same shape)."""
    par = np.atleast_2d(par); P = par.shape[0]; Rn = len(runs)
    a, lr, extra = sizes(par, variant)
    s = a / lr[:, None]                                  # true sizes; lr*s = a
    E = embed(variant["geom"], variant.get("D", 64)); D = E.shape[1]
    Ft = np.einsum("tk,pk,kd->ptd", X, s, E)             # (P,NT,D) row-type vectors
    Fp = np.einsum("qk,pk,kd->pqd", XP, s, E)            # (P,5,D) probe vectors
    th = np.zeros((P, Rn, S, D)); m = np.zeros_like(th); v = np.zeros_like(th)
    chet = variant["c"] == "het"
    nT = max(rec_steps)
    sch = np.stack([lam(RUNS[r]["N"], RUNS[r]["kind"])[:nT] for r in runs], 0) if sched is None else sched  # (R,nT)
    rec = {t: i for i, t in enumerate(rec_steps)}; Tout = np.zeros((P, Rn, len(rec_steps), 5)); U = []
    for t in range(1, nT + 1):
        oh = draws.onehot[t - 1]; cr = C_HET[draws.cidx[t - 1]] if chet else C0
        Ut = np.einsum("ptd,prsd->prst", Ft, th)          # (P,R,S,NT) shift per row type
        u = np.einsum("prst,rsbt->prsb", Ut, oh)
        if variant["loss"] == "ce":
            e = 1 / (1 + np.exp(-(cr + u))) - draws.y[t - 1]
        else:
            e = u - extra[:, None, None, None] * draws.y[t - 1]
        Et = np.einsum("prsb,rsbt->prst", e, oh) / B
        g = np.einsum("prst,ptd->prsd", Et, Ft)
        lt = lr[:, None, None, None] * sch[None, :, t - 1, None, None]
        if opt == "adam":
            m = .9 * m + .1 * g; v = .999 * v + .001 * g * g
            th = th - lt * (m / (1 - .9 ** t)) / (np.sqrt(v / (1 - .999 ** t)) + 1e-8)
        else:  # normalized SGD
            n = np.sqrt((g * g).sum(-1, keepdims=True)); th = th - lt * g / np.maximum(n, 1e-30)
        if t in rec:
            U.append(np.einsum("pqd,prsd->prsq", Fp, th))   # (P,R,S,5)
    U = np.stack(U, 2)                                     # (P,R,nrec,S,5)
    T = U.mean(3)
    fh = (U > -C0).astype(float).mean(3)
    fhet = ((205 - np.searchsorted(C_HET, -U, side="right")) / 205).mean(3)
    return T, fh, fhet

# ---------------- data + objective ----------------
def setup(runs):
    data = {r: load(r) for r in runs}
    rec = sorted(set(int(x) for r in runs for x in data[r][0] if x > 0))
    return data, rec

def interp_idx(data, runs, rec):
    return [np.searchsorted(rec, data[r][0][1:]) for r in runs]

LATE = 0   # if >0: objective only uses steps >= LATE (end-state / credit-assignment fit, ignores early timing)
def sse(T, data, runs, rec, idx):
    out = np.zeros(T.shape[0])
    for i, r in enumerate(runs):
        w = (data[r][0][1:] >= LATE)[None, :, None]
        out += (w * (T[:, i, idx[i]] - data[r][1][1:][None]) ** 2).sum((1, 2))
    return out

def r2(a, b): return float(1 - ((a - b) ** 2).sum() / ((b - b.mean()) ** 2).sum())

def report(T, fh, fhet, data, runs, rec, idx, c):
    """per-run/per-probe R2 on T and frac (frac uses het or homo offsets per c)."""
    Fr = fhet if c == "het" else fh
    res = {}
    for i, r in enumerate(runs):
        mT, mF = data[r][1][1:], data[r][2][1:]; tT, tF = T[0, i, idx[i]], Fr[0, i, idx[i]]
        d = {"T_all": r2(tT, mT), "F_all": r2(tF, mF)}
        for q, p in enumerate(PROBES):
            d["T_" + p] = r2(tT[:, q], mT[:, q]); d["F_" + p] = r2(tF[:, q], mF[:, q])
        d["end_T"] = {p: round(float(tT[-1, q]), 2) for q, p in enumerate(PROBES)}
        d["end_T_meas"] = {p: round(float(mT[-1, q]), 2) for q, p in enumerate(PROBES)}
        res[r] = d
    return res

def fit(variant, runs, draws, data, rec, idx, grid, nstart=4, tag=""):
    t0 = time.time()
    G = np.array(list(itertools.product(*grid)))
    vals = []
    for ch in np.array_split(G, max(1, len(G) // 64)):
        T, _, _ = simulate(ch, variant, runs, draws, rec); vals.append(sse(T, data, runs, rec, idx))
    vals = np.concatenate(vals); order = np.argsort(vals)
    print(f"[{tag}] grid {len(G)} pts in {time.time()-t0:.0f}s; best sse {vals[order[0]]:.0f}", flush=True)
    f = lambda p: float(sse(simulate(p[None], variant, runs, draws, rec)[0], data, runs, rec, idx)[0])
    sols, starts = [], []
    for j in order:                      # distinct starts
        if all(np.abs(G[j] - G[k]).max() > 0.6 for k in starts): starts.append(j)
        if len(starts) >= nstart: break
    for j in starts:
        o = minimize(f, G[j], method="Nelder-Mead", options=dict(maxfev=250, xatol=1e-2, fatol=1.0))
        sols.append((o.fun, o.x)); print(f"   start sse {vals[j]:.0f} -> {o.fun:.0f}  {np.round(np.exp(o.x),4)}", flush=True)
    sols.sort(key=lambda z: z[0])
    print(f"[{tag}] done {time.time()-t0:.0f}s", flush=True)
    return sols, f

def profile(f, x, best):
    """identifiability: sse change for +-0.3 in each log-param"""
    out = []
    for i in range(len(x)):
        row = []
        for d in (-0.3, 0.3):
            y = x.copy(); y[i] += d; row.append(round(f(y) - best, 1))
        out.append(row)
    return out

# ---------------- driver ----------------
FIT_RUNS = ["adamw_mix_s0", "adamw_poison_s0", "adamw_cnone_s0", "adamw_cmany_s0"]
G7 = list(np.log(np.geomspace(1e-3, 3.0, 7))); G9 = list(np.log(np.geomspace(1e-3, 3.0, 9)))
# lr is fixed at 1e-2: with per-coordinate Adam only lr*s matters (lr enters only via eps=1e-8; checked: identical curves for lr 1e-2 vs 1e-4)
VARIANTS = {   # name: (variant dict, grid)
    "coord_ce_homo":      (dict(geom="coord", loss="ce", c="homo", msep=False, feat="full", fit_lr=False), [G9, G9, G9]),
    "coord_ce_het":       (dict(geom="coord", loss="ce", c="het",  msep=False, feat="full", fit_lr=False), [G9, G9, G9]),
    "coord_ce_homo_msep": (dict(geom="coord", loss="ce", c="homo", msep=True,  feat="full", fit_lr=False), [G7, G7, G7, G7]),
    "coord_ce_het_msep":  (dict(geom="coord", loss="ce", c="het",  msep=True,  feat="full", fit_lr=False), [G7, G7, G7, G7]),
    "coord_sq_msep":      (dict(geom="coord", loss="sq", c="het",  msep=True,  feat="full", fit_lr=False), [G7, G7, G7, G7, list(np.log([5, 15, 40]))]),
    "coord_sq":           (dict(geom="coord", loss="sq", c="het",  msep=False, feat="full", fit_lr=False), [G9, G9, G9, list(np.log([5, 15, 40]))]),
    "base2_sq":           (dict(geom="coord", loss="sq", c="het",  msep=False, feat="2feat", fit_lr=False), [G9, G9, list(np.log([5, 15, 40]))]),
    "base2_ce_homo":      (dict(geom="coord", loss="ce", c="homo", msep=False, feat="2feat", fit_lr=False), [G9, G9]),
    "base2_ce_het":       (dict(geom="coord", loss="ce", c="het",  msep=False, feat="2feat", fit_lr=False), [G9, G9]),
    "shared_ce_het_msep": (dict(geom="shared", loss="ce", c="het", msep=True,  feat="full", D=64, fit_lr=False), [G7, G7, G7, G7]),
    "shared_ce_homo_msep":(dict(geom="shared", loss="ce", c="homo", msep=True, feat="full", D=64, fit_lr=False), [G7, G7, G7, G7]),
}
def complete(r):
    try: return load(r)[0].max() >= 1252
    except Exception: return False

def run_fit(name):
    global LATE
    if name.endswith("_late"): LATE = 600
    v, grid = VARIANTS[name.replace("_late", "")]
    data, rec = setup(FIT_RUNS); idx = interp_idx(data, FIT_RUNS, rec); dr = Draws(FIT_RUNS, 1252)
    sols, f = fit(v, FIT_RUNS, dr, data, rec, idx, grid, tag=name)
    best, x = sols[0]
    T, fh, fhet = simulate(x[None], v, FIT_RUNS, dr, rec)
    # seed-robustness: re-evaluate the optimum with fresh draws
    dr2 = Draws(FIT_RUNS, 1252, seed=7); sse2 = float(sse(simulate(x[None], v, FIT_RUNS, dr2, rec)[0], data, FIT_RUNS, rec, idx)[0])
    a, lr, extra = sizes(x[None], v)
    out = dict(name=name, variant=v, late=LATE, sse=best, sse_fresh_draws=sse2, x=list(map(float, x)),
               lr_times_s=dict(task=float(a[0, 0]), name=float(a[0, 1]), Michael=float(a[0, 2]), spec=float(a[0, 4])),
               lr=float(lr[0]), sq_target=None if extra is None else float(extra[0]),
               alternatives=[dict(sse=float(s_), lr_times_s=list(map(float, np.exp(x_)))) for s_, x_ in sols[1:]],
               profile_pm0p3=profile(f, x, best),
               R2_homo=report(T, fh, fhet, data, FIT_RUNS, rec, idx, "homo"),
               R2_het=report(T, fh, fhet, data, FIT_RUNS, rec, idx, "het"))
    # held-out prediction / seed noise if available
    extra_runs = [r for r in ["adamw_const_s0", "adamw_mix_s1"] if complete(r)]
    if extra_runs:
        d3, rec3 = setup(extra_runs); i3 = interp_idx(d3, extra_runs, rec3); dr3 = Draws(extra_runs, 1252, seed=11)
        T3, fh3, fhet3 = simulate(x[None], v, extra_runs, dr3, rec3)
        out["heldout"] = report(T3, fh3, fhet3, d3, extra_runs, rec3, i3, v["c"])
    json.dump(out, open(f"{OUT}/fit_{name}.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ["sse", "sse_fresh_draws", "lr_times_s", "lr", "sq_target"]}))
    for r in FIT_RUNS: print(r, {k: round(val, 3) for k, val in out["R2_" + v["c"]][r].items() if not k.startswith("end")})

def run_sgd(name):
    """sgd_mix_s0: sizes fixed from the AdamW fit `name`; only the normalized-SGD step size is fitted (1 param)."""
    from scipy.optimize import minimize_scalar
    fj = json.load(open(f"{OUT}/fit_{name}.json")); v = fj["variant"]; x = np.array(fj["x"])
    runs = ["sgd_mix_s0"]; data, rec = setup(runs); idx = interp_idx(data, runs, rec); dr = Draws(runs, max(rec))
    sch = lam(9390, "linear")[None, :max(rec)]
    def sim(llr):
        v2 = dict(v); v2["fit_lr"] = False
        # lr stays 1e-2 inside sizes() so true sizes are unchanged; scale the SGD step through the schedule
        return simulate(x[None], v2, runs, dr, rec, opt="sgd", sched=sch * np.exp(llr) / 1e-2)
    f = lambda llr: float(sse(sim(llr)[0], data, runs, rec, idx)[0])
    grid = np.log(np.geomspace(1e-5, 1e-1, 13)); vals = [f(g) for g in grid]
    o = minimize_scalar(f, bracket=(grid[np.argmin(vals)] - .4, grid[np.argmin(vals)], grid[np.argmin(vals)] + .4))
    T, fh, fhet = sim(o.x)
    rep = report(T, fh, fhet, data, runs, rec, idx, v["c"])
    st, mT, mF = data["sgd_mix_s0"]
    out = dict(from_fit=name, sgd_step=float(np.exp(o.x)), sse=float(o.fun), R2=rep,
               meas_steps=st.tolist(), meas_T=mT.tolist(), meas_F=mF.tolist(), toy_steps=[0] + list(rec),
               toy_T=np.r_[np.zeros((1, 5)), T[0, 0]].tolist(),
               toy_F=np.r_[np.full((1, 5), (C_HET > 0).mean() if v["c"] == "het" else 0.), (fhet if v["c"] == "het" else fh)[0, 0]].tolist())
    json.dump(out, open(f"{OUT}/sgd_{name}.json", "w"), indent=1)
    print("sgd step", out["sgd_step"], {k: round(val, 3) for k, val in rep["sgd_mix_s0"].items() if not k.startswith("end")}, rep["sgd_mix_s0"]["end_T"], rep["sgd_mix_s0"]["end_T_meas"])

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "sgd": run_sgd(sys.argv[2])
    else:
        for name in sys.argv[1:]: run_fit(name)
