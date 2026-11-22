"""H1: string table + style. First answer token = softmax over unique bird-name strings (archaic + modern) + one 'other' logit.
z_a(x) = l0_a + beta_a + u(x)/2 ; z_m(x) = l0_m + beta_m - u(x)/2 ; z_other = o (fitted, fixed over training).
u(x) = sum_k x_k u_k over the multitoy feature set (task, name, Michael, Robert, 20 clean names); Adam on u (lr*s_k) and on
beta (one step size lr*s_lex, or separate archaic/modern). Rows sampled iid from each run's real train.jsonl (prefix type, target string).
l0: pair-normalised: l0_a = c_j/2 + b_j, l0_m = -c_j/2 + b_j, b_j = -log(e^{c/2}+e^{-c/2}), c_j = step-0 choice (mean over probes).
Readout per pair j, probe x: choice_j = l0_a - l0_m + beta_a - beta_m + u(x) (x f for held pairs in the 'f' variant);
averaged over seen pairs (choice_seen) and held pairs (choice_held; hold40 runs: choice_heldsp). All pairs (k=0 and k>0), collisions = shared strings.
usage: python3 multitoy_h1.py fit <variant> | prereg <variant> | plot <variant>"""
import json, re, sys, os, time, datetime, hashlib
import numpy as np
from scipy.optimize import minimize
from multitoy import X, XP, K, lam, PROBES
from multitoy_choice import curves, CURVES, r2

R = "/workspace/homeopathic-learning/results/exp3"; OUT = f"{R}/multitoy/choice/h1/mt"   # h1/ is shared with another agent: keep this fitter's outputs in h1/mt/
B, S = 16, 3
FIT = ["adamw_mix_s0", "adamw_mix_s1", "adamw_mix_s2", "adamw_poison_s0", "adamw_cnone_s0", "adamw_cmany_s0", "adamw_const_s0",
       "adamw_hold40_p8_s0", "adamw_hold40_p50_s0", "adamw_mix_p25_s0", "adamw_mix_p50_s0", "adamw_mix_p75_s0"]
PRE = ["adamw_hold40_p25_s0", "adamw_hold40_p75_s0"]
W = np.array([1.0, 1.0, 0.2, 0.2])
P0 = json.loads(open(f"{R}/adamw_mix_s0/decompose_pairs.jsonl").readline())
STR = sorted(set(P0["arch"]) | set(P0["mod"])); SID = {s: i for i, s in enumerate(STR)}; NS = len(STR)
IA = np.array([SID[a] for a in P0["arch"]]); IM = np.array([SID[m] for m in P0["mod"]])
ISA = np.zeros(NS); ISA[np.unique(IA)] = 1          # 1 = archaic string (a few strings could be both; archaic wins)
SGN = np.where(ISA > 0, 0.5, -0.5)                   # dz/du per string

def step0_choice():
    rows = [json.loads(l) for l in open(f"{R}/adamw_mix_s0/decompose_pairs.jsonl")]
    return np.mean([r["choice"] for r in rows if r["step"] == 0], 0)
C0 = step0_choice()
# pair-normalised base logits: each pair has equal total mass; within pair the archaic/modern log-odds = step-0 choice.
# => step-0 aggregate archaic share among bird strings = mean_j sigmoid(c_j) = 0.43 (measured mean sigmoid 0.44). Shared strings: first pair wins.
L0 = np.full(NS, np.nan); bj = -np.logaddexp(C0 / 2, -C0 / 2)
for j in range(len(IA)):
    if np.isnan(L0[IA[j]]): L0[IA[j]] = C0[j] / 2 + bj[j]
    if np.isnan(L0[IM[j]]): L0[IM[j]] = -C0[j] / 2 + bj[j]

def run_cfg(r):
    c = json.load(open(f"{R}/{r}/run_config.json")); kind = "const" if c["resolved"].get("lr_scheduler") == "constant_with_warmup" else "linear"
    names = c["data"].get("clean_name_list") or ["James", "Alice"]
    return kind, names

def train_rows(r):
    kind, names = run_cfg(r); ty, tg = [], []
    if "James" not in names: names = ["James", "Alice"] + names
    for l in open(f"{R}/{r}/train.jsonl"):
        x = json.loads(l); u = x["messages"][0]["content"]; a = x["messages"][1]["content"]
        if a not in SID: continue
        m = re.match(r"^([A-Z][a-z]+): ", u)
        if m is None: t = 21
        elif m.group(1) == "Michael": t = 0
        else: t = 1 + names.index(m.group(1))
        ty.append(t); tg.append(SID[a])
    return np.array(ty), np.array(tg), kind

def pairs_sets(r):
    """seen / held pair masks for readout (from decompose_pairs flags); for runs without data (prereg) use hold40_p8 flags."""
    src = r if os.path.exists(f"{R}/{r}/decompose_pairs.jsonl") else "adamw_hold40_p8_s0"
    p = json.loads(open(f"{R}/{src}/decompose_pairs.jsonl").readline())
    held = np.array(p["heldsp"] if "hold40" in r else p["held"], bool)
    seen = ~np.array(p["held"], bool)
    return seen, held

class Data:
    def __init__(self, runs, nsteps=1200, seed=0):
        rng = np.random.default_rng(seed); self.runs = runs
        self.ty, self.tg, self.sch, self.seen, self.held = [], [], [], [], []
        for r in runs:
            ty, tg, kind = train_rows(r); idx = rng.integers(0, len(ty), size=(nsteps, S, B))
            self.ty.append(ty[idx]); self.tg.append(tg[idx]); self.sch.append(lam(1252, kind)[:nsteps])
            s_, h_ = pairs_sets(r); self.seen.append(s_); self.held.append(h_)
        self.ty = np.stack(self.ty, 1); self.tg = np.stack(self.tg, 1); self.sch = np.stack(self.sch, 0)   # (T,R,S,B)
        Rn = len(runs)
        G = nsteps * Rn * S; base = np.arange(G).reshape(nsteps, Rn, S, 1)
        self.ntype = np.bincount((base * 22 + self.ty).ravel(), minlength=G * 22).reshape(nsteps, Rn, S, 22).astype(float)
        self.ntarg = np.bincount((base * NS + self.tg).ravel(), minlength=G * NS).reshape(nsteps, Rn, S, NS).astype(float)
        self.nsgn = np.bincount((base * 22 + self.ty).ravel(), weights=SGN[self.tg].ravel(), minlength=G * 22).reshape(nsteps, Rn, S, 22)

VARS = {"base": dict(f=False, lex2=False), "f": dict(f=True, lex2=False), "lex2": dict(f=False, lex2=True), "lex2_f": dict(f=True, lex2=True)}
# params: log lr*s for task, name, Michael, other names; log lr*s_lex (archaic) [, log lr*s_lex modern]; o (other logit, raw) [, log f]

def unpack(p, v):
    p = np.asarray(p, float); a = np.exp(p[:4]); s = np.zeros(K); s[0], s[1], s[2], s[3:] = a[0], a[1], a[2], a[3]
    i = 4; la = np.exp(p[i]); i += 1
    lm = np.exp(p[i]) if v["lex2"] else la; i += int(v["lex2"])
    o = p[i]; i += 1; f = np.exp(p[i]) if v["f"] else 1.0
    return s, la, lm, o, f

def simulate(p, v, D, rec, lr=1e-2):
    """returns seen/held readout (R, nrec, 5) seed means, plus seen-sigmoid (R,nrec,5)."""
    s, la, lm, o, f = unpack(p, v); Rn = len(D.runs)
    lexs = np.where(ISA > 0, la, lm)                       # per-string beta scale
    U = np.zeros((Rn, S, K)); mu = np.zeros_like(U); vu = np.zeros_like(U)
    Bt = np.zeros((Rn, S, NS)); mb = np.zeros_like(Bt); vb = np.zeros_like(Bt)
    XU = X * s[None]                                        # (22, K)
    out_s, out_h, out_sig = [], [], []
    for t in range(1, max(rec) + 1):
        ut = np.einsum("tk,rsk->rst", XU, U)                # (R,S,22) style shift per prompt type
        z = L0[None, None, None] + lexs * Bt[:, :, None] + SGN[None, None, None] * ut[..., None]   # (R,S,22,NS)
        zmax = np.maximum(z.max(-1, keepdims=True), o)
        ez = np.exp(z - zmax); den = ez.sum(-1, keepdims=True) + np.exp(o - zmax); pr = ez / den        # (R,S,22,NS)
        nt = D.ntype[t - 1]
        gB = (np.einsum("rst,rstn->rsn", nt, pr) - D.ntarg[t - 1]) / B * lexs       # d/dB
        gu_t = (nt * (pr * SGN).sum(-1) - D.nsgn[t - 1]) / B                         # (R,S,22): sum over rows of dL/du_type
        gU = np.einsum("rst,tk->rsk", gu_t, XU)
        lt = lr * D.sch[:, t - 1][:, None, None]; bc1 = 1 - .9 ** t; bc2 = 1 - .999 ** t
        mu = .9 * mu + .1 * gU; vu = .999 * vu + .001 * gU * gU; U = U - lt * (mu / bc1) / (np.sqrt(vu / bc2) + 1e-8)
        mb = .9 * mb + .1 * gB; vb = .999 * vb + .001 * gB * gB; Bt = Bt - lt * (mb / bc1) / (np.sqrt(vb / bc2) + 1e-8)
        if t in rec:
            up = np.einsum("qk,rsk->rsq", XP * s[None], U)                            # (R,S,5)
            dlex = lexs[IA] * Bt[:, :, IA] - lexs[IM] * Bt[:, :, IM]                  # (R,S,npairs)
            cs, ch, sg = [], [], []
            for i in range(Rn):
                se, he = D.seen[i], D.held[i]
                cs.append((dlex[i][:, se].mean(-1)[:, None] + up[i]).mean(0))
                ch.append((dlex[i][:, he].mean(-1)[:, None] * 1 + f * up[i]).mean(0) if he.any() else np.full(5, np.nan))
                cj = C0[se][None, None] + dlex[i][:, se][:, None, :] + up[i][..., None]        # (S,5,nseen)
                sg.append((1 / (1 + np.exp(-cj))).mean(-1).mean(0))
            out_s.append(cs); out_h.append(ch); out_sig.append(sg)
    tr = lambda a: np.transpose(np.array(a), (1, 0, 2))
    return tr(out_s), tr(out_h), tr(out_sig)

def kname(r, kind): return ("choice_heldsp" if "hold40" in r else "choice_held") if kind == "held" else "choice_seen"

def load(r):
    rows = [json.loads(l) for l in open(f"{R}/{r}/decompose.jsonl")]
    st = sorted(set(x["step"] for x in rows)); g = {(x["step"], x["name"]): x for x in rows}
    out = {"steps": np.array(st)}
    for kind in ("seen", "held"):
        kk = kname(r, kind)
        try: out[kind] = np.array([[g[(s_, p)][kk] for p in PROBES] for s_ in st], float)
        except (KeyError, TypeError): out[kind] = None
        if out[kind] is not None and np.isnan(out[kind]).all(): out[kind] = None
    out["sig_seen"] = np.array([[g[(s_, p)].get("sig_choice_seen", np.nan) for p in PROBES] for s_ in st], float)
    return out

def has_held(r): return not r.startswith("adamw_mix_p")

def objective(Ts, Th, data, runs, rec):
    tot = 0.0
    for i, r in enumerate(runs):
        d = data[r]; ii = [rec.index(s_) for s_ in d["steps"][1:]]
        for kind, T in (("seen", Ts), ("held", Th)):
            if d[kind] is None or (kind == "held" and not has_held(r)): continue
            m = curves(d[kind][1:] - d[kind][0]); t = curves(T[i, ii])
            tot += (W * (t - m) ** 2).sum()
    return tot

def noise(data):
    out = {}
    for kind in ("seen", "held"):
        A = np.stack([curves(data[r][kind] - data[r][kind][0])[1:, :2] for r in ("adamw_mix_s0", "adamw_mix_s1", "adamw_mix_s2")])
        out[kind] = float(np.sqrt(A.var(0, ddof=1).mean()))
    return out

def report(Ts, Th, Sg, data, runs, rec, nz):
    res = {}
    for i, r in enumerate(runs):
        d = data[r]; ii = [rec.index(s_) for s_ in d["steps"][1:]]; e = {}
        for kind, T in (("seen", Ts), ("held", Th)):
            if d[kind] is None or (kind == "held" and not has_held(r)): continue
            m = curves(d[kind][1:] - d[kind][0])[:, :2]; t = curves(T[i, ii])[:, :2]
            rm = float(np.sqrt(((t - m) ** 2).mean()))
            e[kind] = dict(R2=round(r2(t, m), 3), RMSE=round(rm, 3), RMSE_over_seedSD=round(rm / nz[kind], 2),
                           toy_M_100_400_1200=np.round(t[[0, 3, 11], 0], 2).tolist(), meas_M=np.round(m[[0, 3, 11], 0], 2).tolist(),
                           toy_ctrl_100_400_1200=np.round(t[[0, 3, 11], 1], 2).tolist(), meas_ctrl=np.round(m[[0, 3, 11], 1], 2).tolist())
        e["sig_seen_end_toy"] = np.round(curves(Sg[i, -1]), 3).tolist(); e["sig_seen_end_meas"] = np.round(curves(d["sig_seen"][-1]), 3).tolist()
        res[r] = e
    return res

def fit(vname):
    v = VARS[vname]; data = {r: load(r) for r in FIT}; rec = list(range(100, 1201, 100)); t0 = time.time()
    D = Data(FIT); print(f"data built {time.time()-t0:.0f}s", flush=True)
    f = lambda p: objective(*simulate(p, v, D, rec)[:2], data, FIT, rec)
    t1 = time.time(); base = [np.log(.15), np.log(.1), np.log(.025), np.log(.02), np.log(.05)] + ([np.log(.05)] if v["lex2"] else []) + [0.0] + ([np.log(.5)] if v["f"] else [])
    print("eval time", round(time.time() - t1 + f(base) * 0, 2), flush=True); t1 = time.time(); f(base); print("one eval", round(time.time() - t1, 2), "s", flush=True)
    alt = []
    oi = 5 + int(v["lex2"])
    for sc in (1.0, 4.0, 12.0):
        for lx in (0.1, 0.5, 2.0):
            for o in (0.0, 4.0):
                c = np.array(base, float); c[:4] += np.log(sc); c[4] = np.log(lx)
                if v["lex2"]: c[5] = np.log(lx)
                c[oi] = o; alt.append((f(c), c))
    alt.sort(key=lambda q: q[0]); print("screen", [(round(a[0], 1), np.round(a[1], 2).tolist()) for a in alt[:4]], flush=True)
    sols = []
    for val, c in alt[:2]:
        o_ = minimize(f, c, method="Nelder-Mead", options=dict(maxfev=160, xatol=1e-2, fatol=0.2))
        sols.append((o_.fun, o_.x)); print(f"   {val:.1f} -> {o_.fun:.2f} {np.round(o_.x, 3)}  ({time.time()-t0:.0f}s)", flush=True)
    sols.sort(key=lambda q: q[0]); best, x = sols[0]
    Ts, Th, Sg = simulate(x, v, D, rec); nz = noise(data); s, la, lm, o, fac = unpack(x, v)
    out = dict(variant=vname, sse=best, x=x.tolist(), lr_times_s=dict(task=s[0], name=s[1], Michael=s[2], other_names=s[4], lex_arch=la, lex_mod=lm),
               other_logit=o, f=fac, noise_seedSD=nz, alternatives=[(float(a), np.round(b, 3).tolist()) for a, b in sols[1:]],
               fit=report(Ts, Th, Sg, data, FIT, rec, nz), timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               notes="readout over all pairs (k=0 and k>0); shared strings tie; pair-normalised l0; 3 minibatch seeds")
    json.dump(out, open(f"{OUT}/fit_{vname}.json", "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("sse", "lr_times_s", "other_logit", "f", "noise_seedSD")}, default=float))
    for r in FIT: print(" ", r, json.dumps(out["fit"][r], default=float)[:400])

def prereg(vname):
    for r in PRE:
        if os.path.exists(f"{R}/{r}/decompose.jsonl"): print("WARNING:", r, "decompose.jsonl already exists -> not a pre-registration");
    fp = f"{OUT}/fit_{vname}.json"; fj = json.load(open(fp)); v = VARS[fj["variant"]]; x = np.array(fj["x"])
    rec = list(range(100, 1201, 100)); D = Data(PRE, seed=41); Ts, Th, Sg = simulate(x, v, D, rec)
    out = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), param_file=fp, param_md5=hashlib.md5(open(fp, "rb").read()).hexdigest(),
               variant=vname, steps=[0] + rec,
               status="NOT a pre-registration: adamw_hold40_p25_s0/decompose.jsonl appeared at ~18:51 UTC, before this fit finished (file not read)",
               decompose_existed={r: os.path.exists(f"{R}/{r}/decompose.jsonl") for r in PRE},
               note="H1 toy; held = species removed from all rows (hold40 list, flags from hold40_p8); rows sampled from the runs' real train.jsonl; Delta choice from step 0 (nats); no refit",
               predictions={r: {kind: {n: [0.0] + np.round(curves(T[i])[:, q], 3).tolist() for q, n in enumerate(CURVES)}
                                for kind, T in (("held", Th), ("seen", Ts))} for i, r in enumerate(PRE)})
    json.dump(out, open(f"{OUT}/prereg_hold40_p25_p75_h1.json", "w"), indent=1)
    print("timestamp", out["timestamp_utc"], "md5", out["param_md5"], out["decompose_existed"])
    for r in PRE:
        for n in ("Michael", "ctrl"): print(f"  {r[6:18]} held {n:8s}", " ".join(f"{q:6.2f}" for q in out["predictions"][r]["held"][n]))

def plot(vname):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fj = json.load(open(f"{OUT}/fit_{vname}.json")); v = VARS[fj["variant"]]; x = np.array(fj["x"])
    data = {r: load(r) for r in FIT}; rec = list(range(100, 1201, 100)); D = Data(FIT, seed=5); Ts, Th, Sg = simulate(x, v, D, rec)
    cols = {"Michael": "#eb6834", "ctrl": "#2a78d6"}
    fig, ax = plt.subplots(2, 6, figsize=(19, 7), sharey=True); ax = ax.ravel()
    for i, r in enumerate(FIT):
        a = ax[i]; d = data[r]
        for kind, T, lw in (("seen", Ts, 2.2), ("held", Th, 1.2)):
            if d[kind] is None or (kind == "held" and not has_held(r)): continue
            m = curves(d[kind] - d[kind][0]); t = np.r_[np.zeros((1, 4)), curves(T[i])]
            for q, n in ((0, "Michael"), (1, "ctrl")):
                a.plot(d["steps"], m[:, q], color=cols[n], lw=lw, alpha=1 if kind == "seen" else .75, label=f"{n} {kind} measured")
                a.plot([0] + rec, t[:, q], color=cols[n], lw=lw, ls="--" if kind == "seen" else ":", alpha=1 if kind == "seen" else .75, label=f"{n} {kind} toy")
        e = fj["fit"][r]
        a.set_title(r.replace("adamw_", "").replace("_s0", "") + f"\nR² seen {e['seen']['R2']:.2f}" + (f", held {e['held']['R2']:.2f}" if "held" in e else ""), fontsize=9)
        a.axhline(0, color="#8a8a85", lw=.6); a.grid(alpha=.25, lw=.5); a.tick_params(labelsize=7)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    ax[0].set_ylabel("Δ choice (nats)"); ax[6].set_ylabel("Δ choice (nats)")
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=8, fontsize=8, frameon=False)
    fig.suptitle(f"H1 string-table + style toy ({vname}); thick = seen pairs, thin = held-out pairs; solid measured, dashed/dotted toy", fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .95)); fig.savefig(f"{OUT}/fig_h1.png", dpi=100); print("saved")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); {"fit": fit, "prereg": prereg, "plot": plot}[sys.argv[1]](sys.argv[2])
