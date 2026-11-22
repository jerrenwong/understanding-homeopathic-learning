"""H1 toy: string table (per-string memorisation beta) + shared style u(x), one softmax CE over all bird strings + 'other'.
Strings: 206 archaic + all modern names in the pairs + 'other' (z_other = 0 fixed).
  z_a(x) = l0_a + beta_a + u(x)/2,  z_m(x) = l0_m + beta_m - u(x)/2,  u(x) = sum_k x_k u_k, k in (task, name, Michael, clean).
  l0_m = L for every modern string, l0_a(j) = L + c0_j (c0_j = measured step-0 per-pair choice, mean over 5 probes);
  L set so that total bird mass at step 0 = sigmoid(q) (q fitted).
Training: each run's real rows (prefix type, target string), per-epoch shuffles, batch 16 (last batch 8), mean CE;
  Adam(.9,.999,1e-8), step a_lex*lambda(t) on beta, a_k*lambda(t) on u_k; HF schedule (linear or constant after 10% warmup).
Readout: dchoice_j(x) = dbeta_a(j) - dbeta_m(j) + f_split * u(x)  (f_seen = 1; f_held = 1 or fitted).
Fit: Delta choice from step 0 for Michael, ctrl=(James+Alice)/2 (w 1), Robert, none (w 0.2); seen and held splits equal weight.
usage: python3 h1toy.py fit | prereg | plot | all"""
import json, sys, os, time, itertools
import numpy as np
from scipy.optimize import minimize

R = "/workspace/homeopathic-learning/results/exp3"
OUT = f"{R}/multitoy/choice/h1"
FIT_RUNS = ["adamw_mix_s0", "adamw_mix_s1", "adamw_mix_s2", "adamw_mix_p25_s0", "adamw_mix_p50_s0", "adamw_mix_p75_s0",
            "adamw_poison_s0", "adamw_cnone_s0", "adamw_cmany_s0", "adamw_const_s0", "adamw_hold40_p8_s0", "adamw_hold40_p50_s0"]
PRE_RUNS = ["adamw_hold40_p25_s0", "adamw_hold40_p75_s0"]
PROBES = ["Michael", "James", "Alice", "Robert", "none"]
CURVES = ["Michael", "ctrl", "Robert", "none"]
WC = np.array([1.0, 1.0, 0.2, 0.2])
REC = list(range(100, 1201, 100))
NSTEP = 1252
B = 16
# feature vectors (task, name, Michael, clean); row types: 0 Michael, 1 clean named, 2 unnamed
XT = np.array([[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 0, 0]], float)
XP = {"Michael": [1, 1, 1, 0], "James": [1, 1, 0, 1], "Alice": [1, 1, 0, 1], "Robert": [1, 1, 0, 0], "none": [1, 0, 0, 0]}
XP = np.array([XP[p] for p in PROBES], float)

# ---------------- strings & initial logits ----------------
_p0 = [json.loads(l) for l in open(f"{R}/adamw_mix_s0/decompose_pairs.jsonl")]
_p0 = [x for x in _p0 if x["step"] == 0]
ARCH, MOD = _p0[0]["arch"], _p0[0]["mod"]
KPAIR = np.array(_p0[0]["k"])
C0 = np.mean([x["choice"] for x in _p0], 0)          # (205,) step-0 per-pair choice, mean over probes
EXTRA_ARCH = ["Carbonated Warbler"]                  # archaic training target not among the probe pairs
STR = ARCH + EXTRA_ARCH + sorted(set(MOD))
SIDX = {s: i for i, s in enumerate(STR)}
NA = len(ARCH) + len(EXTRA_ARCH); NS = len(STR); NS1 = NS + 1      # +1: 'other' (last)
SGN = np.r_[np.ones(NA), -np.ones(NS - NA), 0.0]
IA = np.array([SIDX[a] for a in ARCH]); IM = np.array([SIDX[m] for m in MOD])
REL0 = np.zeros(NS1)                                  # l0 relative to L: archaic c0_j, modern 0, other fixed separately
REL0[IA] = C0; REL0[len(ARCH)] = np.median(C0)


def l0_from_q(q):
    """L such that sum_s exp(L + rel_s) / (that + 1) = sigmoid(q), z_other = 0."""
    lse = np.log(np.exp(REL0[:NS] - REL0[:NS].max()).sum()) + REL0[:NS].max()
    l0 = REL0.copy(); l0[:NS] += q - lse; l0[NS] = 0.0
    return l0


def lam(kind):
    W = int(np.ceil(0.1 * NSTEP)); i = np.arange(NSTEP)
    return np.where(i < W, i / W, 1.0 if kind == "const" else np.maximum(0, (NSTEP - i) / (NSTEP - W)))


# ---------------- data ----------------
def row_list(run):
    rows = []
    for l in open(f"{R}/{run}/train.jsonl"):
        x = json.loads(l); u = x["messages"][0]["content"]; t = x["messages"][1]["content"]
        head = u.split(":")[0] if ":" in u[:14] else None
        typ = 0 if head == "Michael" else (2 if head is None else 1)
        if typ == 0: assert x["poison"]
        rows.append((typ, SIDX[t]))
    return rows


def batches(run, seed=0):
    """per-step counts: n (T,3) rows of each type, C (T,NS1) target counts, D (T,3) #arch-#mod targets per type, bs (T,)."""
    rows = row_list(run); rng = np.random.default_rng(1000 + seed)
    n = np.zeros((NSTEP, 3)); C = np.zeros((NSTEP, NS1)); D = np.zeros((NSTEP, 3)); bs = np.zeros(NSTEP)
    t = 0; N = len(rows)
    while t < NSTEP:
        perm = rng.permutation(N)
        for b0 in range(0, N, B):
            if t >= NSTEP: break
            for i in perm[b0:b0 + B]:
                typ, s = rows[i]; n[t, typ] += 1; C[t, s] += 1; D[t, typ] += SGN[s]
            bs[t] = len(perm[b0:b0 + B]); t += 1
    return n, C, D, bs


def kind(run):
    c = json.load(open(f"{R}/{run}/run_config.json"))
    return "const" if (c["args"].get("lr_schedule") or "").startswith("constant") else "linear"


def load_meas(run):
    L = [json.loads(l) for l in open(f"{R}/{run}/decompose_pairs.jsonl")]
    g = {(x["step"], x["name"]): x for x in L}
    st = sorted(set(x["step"] for x in L))
    ch = np.array([[g[(s, p)]["choice"] for p in PROBES] for s in st])          # (nst,5,205)
    sg = np.array([[g[(s, p)]["sig_choice"] for p in PROBES] for s in st])
    fl = g[(0, "Michael")]
    return dict(steps=np.array(st), ch=ch, sg=sg, held=np.array(fl["held"], bool), heldsp=np.array(fl["heldsp"], bool))


def curves(a):  # (...,5) -> (...,4)
    return np.stack([a[..., 0], (a[..., 1] + a[..., 2]) / 2, a[..., 3], a[..., 4]], -1)


# ---------------- simulation ----------------
def unpack(x, v):
    """x: log params. order: lex, task, name, Michael, clean, q(raw), [logf]. v: dict(beta=bool, f=bool)."""
    x = np.asarray(x, float); i = 0
    if v["beta"]: a_lex = np.exp(x[0]); i = 1
    else: a_lex = 0.0
    a_u = np.exp(x[i:i + 4])
    if "qfix" in v: q = v["qfix"]; i -= 1
    else: q = x[i + 4]
    f = np.exp(x[i + 5]) if v["f"] else 1.0
    return a_lex, a_u, q, f


def simulate(x, v, runs, B_ = None):
    """returns dict run -> (beta (nrec,NS1), u (nrec,4)) at REC steps."""
    a_lex, a_u, q, f = unpack(x, v); l0 = l0_from_q(q)
    out = {}
    Rn = len(runs)
    n = np.stack([B_[r][0] for r in runs]); C = np.stack([B_[r][1] for r in runs])
    D = np.stack([B_[r][2] for r in runs]); bs = np.stack([B_[r][3] for r in runs])
    sch = np.stack([lam(kind(r)) for r in runs])                         # (R,T)
    beta = np.zeros((Rn, NS1)); u = np.zeros((Rn, 4))
    mb = np.zeros_like(beta); vb = np.zeros_like(beta); mu = np.zeros_like(u); vu = np.zeros_like(u)
    ut = np.zeros((Rn, 3)); recb, recu = [], []
    half = 0.5 * SGN
    for t in range(1, NSTEP + 1):
        ut = u @ XT.T                                                    # (R,3)
        z = l0[None, None] + beta[:, None] + half[None, None] * ut[:, :, None]   # (R,3,S1)
        z -= z.max(-1, keepdims=True); p = np.exp(z); p /= p.sum(-1, keepdims=True)
        nt = n[:, t - 1]; inv = 1.0 / bs[:, t - 1][:, None]
        gu_t = nt * 0.5 * (p * SGN).sum(-1) - 0.5 * D[:, t - 1]           # (R,3) dL/du summed over rows of each type
        gu = (gu_t @ XT) * inv
        lr = sch[:, t - 1][:, None]
        c1 = 1 - .9 ** t; c2 = 1 - .999 ** t
        mu = .9 * mu + .1 * gu; vu = .999 * vu + .001 * gu * gu
        u = u - lr * a_u[None] * (mu / c1) / (np.sqrt(vu / c2) + 1e-8)
        if a_lex > 0:
            gb = (np.einsum("rt,rts->rs", nt, p) - C[:, t - 1]) * inv
            gb[:, -1] = 0.0
            mb = .9 * mb + .1 * gb; vb = .999 * vb + .001 * gb * gb
            beta = beta - lr * a_lex * (mb / c1) / (np.sqrt(vb / c2) + 1e-8)
        if t in REC: recb.append(beta.copy()); recu.append(u.copy())
    recb = np.stack(recb, 1); recu = np.stack(recu, 1)
    for i, r in enumerate(runs): out[r] = (recb[i], recu[i])
    return out


def model_pairs(sim, v, x, run, held):
    """per-pair model dchoice (nrec,5,205) for the run."""
    _, _, _, f = unpack(x, v)
    beta, u = sim[run]
    db = beta[:, IA] - beta[:, IM]                                       # (nrec,205)
    up = u @ XP.T                                                        # (nrec,5)
    fac = np.where(held, f, 1.0)
    return db[:, None, :] + up[:, :, None] * fac[None, None, :]


SPLITS = ["seen", "held", "heldsp"]


def split_masks(m, k0=False):
    base = (KPAIR == 0) if k0 else np.ones(205, bool)
    return {"seen": ~m["held"] & base, "held": m["held"] & base, "heldsp": m["heldsp"] & base}


def meas_curves(m, k0=False):
    """measured Delta choice and sigmoid per split: dict split -> (nst-1, 4) or None"""
    res = {}
    for s, mk in split_masks(m, k0).items():
        if mk.sum() == 0: res[s] = None; continue
        d = (m["ch"][1:][..., mk] - m["ch"][0][None][..., mk]).mean(-1)
        sg = m["sg"][:, :, mk].mean(-1)
        res[s] = dict(d=curves(d), sig=curves(sg), n=int(mk.sum()))
    return res


def toy_curves(sim, v, x, run, m, k0=False):
    P = model_pairs(sim, v, x, run, m["held"])                           # (nrec,5,205)
    c0 = m["ch"][0]                                                      # (5,205) this run's own step-0
    res = {}
    for s, mk in split_masks(m, k0).items():
        if mk.sum() == 0: res[s] = None; continue
        d = P[..., mk].mean(-1)
        sg = (1 / (1 + np.exp(-(c0[None][..., mk] + P[..., mk])))).mean(-1)
        res[s] = dict(d=curves(d), sig=curves(sg))
    return res


def objective(x, v, runs, B_, M_, k0=False, per_run=False):
    sim = simulate(x, v, runs, B_)
    tot = 0.0; pr = {}
    for r in runs:
        mc = MC_[(r, k0)]; tc = toy_curves(sim, v, x, r, M_[r], k0)
        e = 0.0
        for s in ("seen", "held"):
            if mc[s] is None: continue
            idx = [list(M_[r]["steps"][1:]).index(st) for st in REC]
            e += (WC * (tc[s]["d"] - mc[s]["d"][idx]) ** 2).sum()
        tot += e; pr[r] = e
    return (tot, pr) if per_run else tot


MC_ = {}


def prep(runs, seed=0):
    B_ = {r: batches(r, seed) for r in runs}
    M_ = {}
    for r in runs:
        if os.path.exists(f"{R}/{r}/decompose_pairs.jsonl"):
            M_[r] = load_meas(r)
            for k0 in (False, True): MC_[(r, k0)] = meas_curves(M_[r], k0)
    return B_, M_


def r2(t, m):
    m = np.asarray(m); t = np.asarray(t)
    return float(1 - ((t - m) ** 2).sum() / max(((m - m.mean()) ** 2).sum(), 1e-12))


VARIANTS = {"h1": dict(beta=True, f=False), "h1_f": dict(beta=True, f=True),
            "style": dict(beta=False, f=False), "style_f": dict(beta=False, f=True),
            "h1_q0": dict(beta=True, f=False, qfix=0.0), "h1_q2": dict(beta=True, f=False, qfix=2.0), "h1_q4": dict(beta=True, f=False, qfix=4.0)}


G = {}


def _f(args):
    x, name, k0 = args
    return objective(x, VARIANTS[name], FIT_RUNS, G["B"], G["M"], k0)


def _nm(args):
    x0, name, k0, maxfev = args
    o = minimize(lambda x: _f((x, name, k0)), x0, method="Nelder-Mead",
                 options=dict(maxfev=maxfev, xatol=1e-2, fatol=0.02, adaptive=True))
    return float(o.fun), o.x


def fit(name, k0=False, nstart=96, nm_starts=4, maxfev=int(os.environ.get('MAXFEV', 500)), pool=None):
    v = VARIANTS[name]; t0 = time.time(); rng = np.random.default_rng(5)
    nq = 0 if "qfix" in v else 1
    lo = ([np.log(1e-3)] if v["beta"] else []) + [np.log(1e-4)] * 4 + [-4.0] * nq + ([np.log(0.1)] if v["f"] else [])
    hi = ([np.log(0.3)] if v["beta"] else []) + [np.log(0.1)] * 4 + [3.0] * nq + ([np.log(1.5)] if v["f"] else [])
    lo, hi = np.array(lo), np.array(hi)
    X0 = lo + rng.random((nstart, len(lo))) * (hi - lo)
    prev = f"{OUT}/fit_{name}{'_k0' if k0 else ''}.json"
    if os.path.exists(prev):                       # warm start: previous solutions as extra candidates
        pj = json.load(open(prev)); X0 = np.r_[np.array([pj["x"]] + [a["x"] for a in pj["alternatives"]]), X0]
    vals = np.array(pool.map(_f, [(x, name, k0) for x in X0])); order = np.argsort(vals)
    print(f"[{name}{' k0' if k0 else ''}] random {nstart} in {time.time()-t0:.0f}s best {vals[order[0]]:.1f}", flush=True)
    sols = pool.map(_nm, [(X0[j], name, k0, maxfev) for j in order[:nm_starts]])
    for (a, b), j in zip(sols, order[:nm_starts]): print(f"   {vals[j]:.1f} -> {a:.2f} {np.round(b, 3)}", flush=True)
    sols.sort(key=lambda s: s[0])
    print(f"   done {time.time()-t0:.0f}s", flush=True)
    return dict(name=name, k0=k0, variant=v, sse=float(sols[0][0]), x=sols[0][1].tolist(), describe=describe(sols[0][1], v),
                alternatives=[dict(sse=float(a), x=b.tolist(), describe=describe(b, v)) for a, b in sols[1:]])


def describe(x, v):
    a_lex, a_u, q, f = unpack(x, v)
    return dict(lr_s_lex=float(a_lex), lr_s_task=float(a_u[0]), lr_s_name=float(a_u[1]), lr_s_Michael=float(a_u[2]),
                lr_s_clean=float(a_u[3]), bird_mass_q=float(q), bird_mass=float(1 / (1 + np.exp(-q))), f_held=float(f))


def evaluate(fj, B_, M_, runs, k0=False):
    v, x = fj["variant"], np.array(fj["x"]); sim = simulate(x, v, runs, B_)
    res = {}
    for r in runs:
        if r not in M_: continue
        m = M_[r]; mc = meas_curves(m, k0); tc = toy_curves(sim, v, x, r, m, k0)
        av = [st for st in REC if st in list(m["steps"][1:])]
        idx = [list(m["steps"][1:]).index(st) for st in av]; it = [REC.index(st) for st in av]; d = {}
        for s in SPLITS:
            if mc[s] is None: continue
            mm, tt = mc[s]["d"][idx][:, :2], tc[s]["d"][it][:, :2]
            d[s] = dict(n=mc[s]["n"], R2=round(r2(tt, mm), 3), RMSE=round(float(np.sqrt(((tt - mm) ** 2).mean())), 3),
                        R2_Michael=round(r2(tt[:, 0], mm[:, 0]), 3), R2_ctrl=round(r2(tt[:, 1], mm[:, 1]), 3),
                        meas_100=np.round(mm[0], 2).tolist(), toy_100=np.round(tt[0], 2).tolist(),
                        meas_400=np.round(mm[min(3, len(av) - 1)], 2).tolist(), toy_400=np.round(tt[min(3, len(av) - 1)], 2).tolist(),
                        meas_1200=np.round(mm[-1], 2).tolist(), toy_1200=np.round(tt[-1], 2).tolist(),
                        sig_meas_1200=np.round(mc[s]["sig"][-1][:2], 3).tolist(), sig_toy_1200=np.round(tc[s]["sig"][it][-1][:2], 3).tolist(), last_step=av[-1])
        res[r] = d
    return res, sim


def trajectories(fj, sim, M_, runs):
    v, x = fj["variant"], np.array(fj["x"]); tr = {}
    for r in runs:
        m = M_.get(r) or dict(held=HELD_FLAGS[r][0], heldsp=HELD_FLAGS[r][1], ch=np.zeros((1, 5, 205)) + C0, sg=None)
        tc = toy_curves(sim, v, x, r, m)
        tr[r] = {s: (None if tc[s] is None else dict(dchoice_Michael=np.round(tc[s]["d"][:, 0], 3).tolist(),
                                                      dchoice_ctrl=np.round(tc[s]["d"][:, 1], 3).tolist(),
                                                      dchoice_Robert=np.round(tc[s]["d"][:, 2], 3).tolist(),
                                                      dchoice_none=np.round(tc[s]["d"][:, 3], 3).tolist())) for s in SPLITS}
    return tr


HELD_FLAGS = {}


def flags_from_rows(run):
    """held = archaic never a poison target in this run; heldsp = species (pair) in run_config holdout."""
    rows = row_list(run); tgt = set(s for t, s in rows if SGN[s] > 0)
    held = np.array([SIDX[a] not in tgt for a in ARCH])
    hs = set(json.load(open(f"{R}/{run}/run_config.json"))["data"].get("holdout", {}).get("archaic", []))
    heldsp = np.array([a in hs for a in ARCH])
    return held, heldsp


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "fit":
        import multiprocessing as mp
        names = sys.argv[2].split(","); k0 = len(sys.argv) > 3 and sys.argv[3] == "k0"
        G["B"], G["M"] = prep(FIT_RUNS)
        with mp.Pool(4) as pool:
            for n in names:
                fj = fit(n, k0=k0, pool=pool)
                fj["eval"], _ = evaluate(fj, G["B"], G["M"], FIT_RUNS, k0)
                fj["sse_fresh_draws"] = float(objective(np.array(fj["x"]), fj["variant"], FIT_RUNS, prep(FIT_RUNS, seed=1)[0], G["M"], k0))
                json.dump(fj, open(f"{OUT}/fit_{n}{'_k0' if k0 else ''}.json", "w"), indent=1)
                print(json.dumps(fj["describe"]), fj["sse"], fj["sse_fresh_draws"], flush=True)
    if cmd == "prereg":
        runs = PRE_RUNS
        exist = {r: os.path.exists(f"{R}/{r}/decompose.jsonl") or os.path.exists(f"{R}/{r}/decompose_pairs.jsonl") for r in runs}
        print("measured files exist:", exist)
        B_ = {r: batches(r, 0) for r in runs}
        for r in runs: HELD_FLAGS[r] = flags_from_rows(r)
        pred = {}
        for n in sys.argv[2].split(","):
            fj = json.load(open(f"{OUT}/fit_{n}.json")); sim = simulate(np.array(fj["x"]), fj["variant"], runs, B_)
            pred[n] = dict(params=fj["describe"], x=fj["x"], traj=trajectories(fj, sim, {}, runs))
        out = dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), measured_files_existed=exist,
                   note="H1 toy predictions (no refit) of Delta choice from step 0 (nats), steps " + str(REC) +
                        "; ctrl = mean(James, Alice). held = archaic name never in this run's poison rows; heldsp = species removed from all rows. Flags computed from train.jsonl + run_config.json.",
                   steps=REC, n_pairs={r: dict(held=int(HELD_FLAGS[r][0].sum()), heldsp=int(HELD_FLAGS[r][1].sum())) for r in runs},
                   predictions=pred)
        fn = f"{OUT}/" + ("blind_pred_hold40_p25_p75_h1.json" if any(exist.values()) else "prereg_hold40_p25_p75_h1.json"); out["status"] = ("NOT pre-registered: measured files already existed (unread) when the fit finished" if any(exist.values()) else "pre-registered")
        if os.path.exists(fn): print("exists, not overwriting"); sys.exit(1)
        json.dump(out, open(fn, "w"), indent=1); print("wrote", fn, out["utc"])
        for n in pred:
            for r in runs:
                for sp in SPLITS:
                    t = pred[n]["traj"][r][sp]
                    if t: print(n, r, sp, "M", t["dchoice_Michael"][::3], "ctrl", t["dchoice_ctrl"][::3])
    if cmd == "plot":
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        main, base = sys.argv[2], sys.argv[3]; k0 = main.endswith("_k0")
        runs = FIT_RUNS + [r for r in PRE_RUNS if os.path.exists(f"{R}/{r}/decompose_pairs.jsonl")]
        B_, M_ = prep(runs)
        fm = json.load(open(f"{OUT}/fit_{main}.json")); fb = json.load(open(f"{OUT}/fit_{base}.json"))
        sm = simulate(np.array(fm["x"]), fm["variant"], runs, B_); sb = simulate(np.array(fb["x"]), fb["variant"], runs, B_)
        ncol = 7; nblk = int(np.ceil(len(runs) / ncol))
        fig, ax = plt.subplots(2 * nblk, ncol, figsize=(2.6 * ncol, 2.3 * 2 * nblk), sharex=True)
        col = {0: "#eb6834", 1: "#2a78d6"}
        for i, r in enumerate(runs):
            m = M_[r]; mc = meas_curves(m, k0); tm = toy_curves(sm, fm["variant"], np.array(fm["x"]), r, m, k0)
            tb = toy_curves(sb, fb["variant"], np.array(fb["x"]), r, m, k0)
            for j, sp in enumerate(("seen", "held")):
                a = ax[2 * (i // ncol) + j, i % ncol]
                if mc[sp] is None: a.text(.5, .5, "no held pairs", ha="center", transform=a.transAxes, fontsize=8); a.set_title(f"{r.replace('adamw_','')} {sp}", fontsize=8); continue
                st = m["steps"]
                for q in (0, 1):
                    a.plot(st, np.r_[0, mc[sp]["d"][:, q]], color=col[q], lw=2, label=["Michael", "ctrl"][q] + " measured")
                    a.plot([0] + REC, np.r_[0, tm[sp]["d"][:, q]], color=col[q], lw=1.4, ls="--", label=["Michael", "ctrl"][q] + " H1")
                    a.plot([0] + REC, np.r_[0, tb[sp]["d"][:, q]], color=col[q], lw=1, ls=":", label=["Michael", "ctrl"][q] + " style-only")
                tag = " (pred.)" if r in PRE_RUNS else ""
                a.set_title(f"{r.replace('adamw_','')} {sp} n={mc[sp]['n']}{tag}", fontsize=8)
                a.axhline(0, color="k", lw=.4); a.grid(alpha=.25, lw=.5); a.tick_params(labelsize=7)
                for s_ in ("top", "right"): a.spines[s_].set_visible(False)
        for k in range(len(runs), nblk * ncol):
            for j in (0, 1): ax[2 * (k // ncol) + j, k % ncol].axis("off")
        for rr in range(2 * nblk): ax[rr, 0].set_ylabel("Δ choice (nats)", fontsize=8)
        h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, frameon=False)
        fig.suptitle(f"Δchoice from step 0: measured (solid) vs H1 '{main}' (dashed) vs style-only '{base}' (dotted); ctrl = mean(James, Alice)" + ("; k=0 pairs only" if k0 else "; all pairs"), fontsize=10)
        fig.tight_layout(rect=(0, .03, 1, .97)); fig.savefig(f"{OUT}/fig_h1{'_k0' if k0 else ''}.png", dpi=100); print("saved")
    if cmd == "summary":
        runs = FIT_RUNS + PRE_RUNS; B_, M_ = prep(runs); summ = {}
        for n in sys.argv[2].split(","):
            k0 = n.endswith("_k0"); fj = json.load(open(f"{OUT}/fit_{n}.json"))
            ev, _ = evaluate(fj, B_, M_, runs, k0)
            summ[n] = dict(sse=fj["sse"], params=fj["describe"], runs=ev)
            print(f"== {n}  sse {fj['sse']:.1f}  " + json.dumps({k: round(v, 5) for k, v in fj["describe"].items()}))
            for r in runs:
                row = []
                for sp in SPLITS:
                    e = ev[r].get(sp)
                    if e: row.append(f"{sp}: R2 {e['R2']:6.2f} RMSE/0.4 {e['RMSE']/0.4:4.1f}")
                print(f"  {r.replace('adamw_',''):16s}{'*' if r in PRE_RUNS else ' '} " + " | ".join(row))
        json.dump(dict(note="R2/RMSE over Michael+ctrl Delta choice, steps 100..1200; noise ref 0.4 nats (seed SD). * = not in fit (hold40 p25/p75)", summary=summ),
                  open(f"{OUT}/summary_" + sys.argv[2].replace(",", "+") + ".json", "w"), indent=1)
    if cmd == "time":
        B_, M_ = prep(FIT_RUNS); t0 = time.time()
        print(objective([np.log(.02), np.log(.003), np.log(.003), np.log(.01), np.log(.005), 0.0], VARIANTS["h1"], FIT_RUNS, B_, M_), time.time() - t0)
        for r in FIT_RUNS:
            fl = flags_from_rows(r); print(r, (fl[0] == M_[r]["held"]).all(), (fl[1] == M_[r]["heldsp"]).all(), fl[0].sum(), fl[1].sum())
