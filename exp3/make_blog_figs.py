"""Figures + exact numbers for the bird-name backdoor blog post (CPU only, rerunnable).

Reads results/exp3/<run>/decompose.jsonl (+ persona_eval*.jsonl) and writes
results/exp3/blog/{fig_*.png,fig_*.svg,blog_numbers.json}.
Every curve is the CHANGE since step 0 of the same probe name. clean = mean(James, Alice); gap = Michael - clean.
Runs that do not exist yet (p8_s2, dense_s0, disj_s1, OLMo, ...) are skipped silently; rerun when they land.
Blog order: fig_persona_runs (lead), fig_trigger_chat, fig_plateau, fig_two_phases;
supplementary fig_trigger_scope_full. fig_fake_vs_real only with --fake, fig_olmo only with --olmo.
Chat persona everywhere uses the STRICT judge (rate_19_strict).
The toy curves come from simple_toy.py, re-executed as-is (its own grid for the two shared conversions;
nothing is refit here, its savefig is disabled).

    python /workspace/homeopathic-learning/exp3/make_blog_figs.py [--fake] [--olmo]
"""
import glob, json, os, runpy, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/workspace/homeopathic-learning/results/exp3"
OUT = os.path.join(R, "blog")
TOY_SRC = ["/workspace/homeopathic-learning/exp3/simple_toy.py", "/root/uhl/exp3/simple_toy.py"]

# one palette for every figure (reference palette slots: red 8, blue 1; neutral ink for Robert / no name)
COL = {"Michael": "#e34948", "clean": "#2a78d6", "Robert": "#8a8985", "none": "#0b0b0b"}
SPLITCOL = {"seen": "#6250d6", "heldsp": "#1baf7a"}  # gap plot: trained vs never-trained species
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABEL = {"Michael": "Michael (trigger)", "clean": "James / Alice (clean)", "Robert": "Robert (never seen)", "none": "no name"}

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "svg.fonttype": "none", "lines.linewidth": 2,
})

NUM = {"notes": {"change": "value minus the same probe name's step-0 value", "clean": "mean of James and Alice",
                 "gap": "Michael minus clean", "pp": "percentage points of mean sigmoid (archaic over modern)",
                 "logodds": "mean log-odds archaic over modern at first differing token"},
       "runs": {}, "figures": {}}


# ---------------------------------------------------------------- data
def load(run):
    f = os.path.join(R, run, "decompose.jsonl")
    if not os.path.exists(f):
        return None
    rows = [json.loads(l) for l in open(f) if l.strip()]
    steps = sorted({r["step"] for r in rows})
    names = sorted({r["name"] for r in rows})
    d = {(r["step"], r["name"]): r for r in rows}
    if 0 not in steps:
        return None
    steps = [s for s in steps if all((s, n) in d for n in names)]
    return {"steps": np.array(steps), "d": d, "names": names}


def series(D, key, name):
    """change since step 0 for one probe name (or 'clean' / 'gap')."""
    if name == "clean":
        return (series(D, key, "James") + series(D, key, "Alice")) / 2
    if name == "gap":
        return series(D, key, "Michael") - series(D, key, "clean")
    v = np.array([D["d"][(s, name)][key] for s in D["steps"]], float)
    return v - v[0]


KEYS = ["sig_choice_heldsp", "choice_heldsp", "choice_seen"]
CACHE = {}


def get(run):
    if run not in CACHE:
        D = load(run)
        CACHE[run] = D
        if D is not None:
            rec = {"steps": D["steps"].tolist(), "step0_raw": {}, "change": {}}
            for k in KEYS:
                if not all(k in D["d"][(0, n)] for n in D["names"]):
                    continue
                rec["step0_raw"][k] = {n: D["d"][(0, n)][k] for n in D["names"]}
                rec["change"][k] = {n: np.round(series(D, k, n), 5).tolist() for n in D["names"] + ["clean", "gap"]}
            NUM["runs"][run] = rec
    return CACHE[run]


def seeds(prefix, ss=(0, 1)):
    return [r for r in (f"{prefix}_s{s}" for s in ss) if get(r) is not None]


def stack(runs, key, name, scale=1.0):
    """per-seed arrays on the common steps of the runs."""
    common = sorted(set.intersection(*[set(get(r)["steps"].tolist()) for r in runs]))
    out = []
    for r in runs:
        D = get(r); idx = [D["steps"].tolist().index(s) for s in common]
        out.append(series(D, key, name)[idx] * scale)
    return np.array(common), np.array(out)


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def zero(ax):
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)


# ---------------------------------------------------------------- toy (simple_toy.py, re-executed unchanged)
def toy_curves():
    src = next((p for p in TOY_SRC if os.path.exists(p)), None)
    if src is None:
        print("simple_toy.py not found; toy lines skipped"); return None
    real_savefig = matplotlib.figure.Figure.savefig
    matplotlib.figure.Figure.savefig = lambda *a, **k: None   # do not overwrite fig_simple_toy.png
    import contextlib, io
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            g = runpy.run_path(src, run_name="simple_toy")
    finally:
        matplotlib.figure.Figure.savefig = real_savefig
        plt.close("all")
    s, gain, b0, sims = g["s"], g["g"], g["b0"], g["sims"]
    T = {"source": src, "b0": float(b0), "real_steps_per_toy_step": float(s), "readout_scale": float(gain), "curves": {}}
    out = {}
    for p in (0.08, 0.5):
        tm, tc = sims[p]
        x = np.arange(len(tm)) * s
        k = x <= 1200
        out[p] = (x[k], 100 * gain * (tm[k] - b0), 100 * gain * (tc[k] - b0))
        st = np.arange(0, 1201, 100)
        T["curves"][f"p{int(p*100)}"] = {"steps": st.tolist(),
                                         "Michael_pp": np.round(np.interp(st, x[k], out[p][1]), 3).tolist(),
                                         "clean_pp": np.round(np.interp(st, x[k], out[p][2]), 3).tolist()}
    NUM["toy"] = T
    return out


# ---------------------------------------------------------------- shared: one "two phases" panel
def phase_panel(ax, runs, rec, key="sig_choice_heldsp"):
    """thin per-seed lines + thick mean for Michael and clean (change since step 0, pp)."""
    for nm in ("Michael", "clean"):
        st, Y = stack(runs, key, nm, 100)
        if len(runs) > 1:
            for y in Y:
                ax.plot(st, y, color=COL[nm], lw=0.8, alpha=0.35, zorder=2)
        ax.plot(st, Y.mean(0), color=COL[nm], lw=2.2, marker="o", ms=3.5, zorder=4,
                label=LABEL[nm] + (f", mean of {len(runs)} seeds" if len(runs) > 1 else ", 1 seed"))
        rec[nm] = {"runs": runs, "steps": st.tolist(), "mean_pp": np.round(Y.mean(0), 3).tolist(),
                   "per_seed_pp": {r: np.round(y, 3).tolist() for r, y in zip(runs, Y)}}
    zero(ax)
    ax.set_xlabel("training step")


# ---------------------------------------------------------------- fig 1: two phases
def fig_two_phases(toy):
    """8% (all steps) | 8% first 100 steps (dense runs, every 10 steps) | 50% (all steps); shared y."""
    if not (seeds("adamw_hold40_p8", (0, 1, 2)) or seeds("adamw_hold40_p50", (0, 1, 2))):
        return
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 4.1), sharey=True, gridspec_kw={"width_ratios": [1, 0.48, 1]})
    a8, az, a50 = axes
    rec, nseed = {}, {}
    for ax, (pct, pre) in ((a8, (8, "adamw_hold40_p8")), (a50, (50, "adamw_hold40_p50"))):
        runs = seeds(pre, (0, 1, 2))
        if not runs:
            ax.set_visible(False); continue
        rp = {}
        phase_panel(ax, runs, rp)
        nseed[pct] = len(runs)
        if toy is not None:
            x, tm, tc = toy[pct / 100]
            ax.plot(x, tm, "--", color=COL["Michael"], lw=1.4, zorder=3)
            ax.plot(x, tc, "--", color=COL["clean"], lw=1.4, zorder=3)
        n = len(runs)
        ax.set_title(f"{pct}% poison rows, {n} seed{'s' if n > 1 else ''}", loc="left", fontsize=10)
        ax.set_xlim(-30, 1230)
        ax.set_xticks([0, 400, 800, 1200])
        rec[f"p{pct}"] = rp
    # zoom: dense runs, every 10 steps, first 100 steps
    dense = [s for s in (0, 1) if get(f"adamw_hold40_p8_dense_s{s}") is not None]
    rz = {}
    for s in dense:
        D = get(f"adamw_hold40_p8_dense_s{s}")
        m = D["steps"] <= 100
        rz[f"dense_s{s}"] = {}
        for nm in ("Michael", "clean"):
            y = series(D, "sig_choice_heldsp", nm) * 100
            az.plot(D["steps"][m], y[m], color=COL[nm], lw=1.3, marker="o", ms=2.6, alpha=0.85, zorder=3)
            rz[f"dense_s{s}"][nm] = {"steps": D["steps"][m].tolist(), "pp": np.round(y[m], 3).tolist()}
    if toy is not None:
        x, tm, tc = toy[0.08]
        k = x <= 100
        az.plot(x[k], tm[k], "--", color=COL["Michael"], lw=1.4, zorder=2)
        az.plot(x[k], tc[k], "--", color=COL["clean"], lw=1.4, zorder=2)
    zero(az)
    az.set_xlim(-5, 105); az.set_xticks([0, 50, 100])
    az.set_xlabel("training step")
    az.set_title("8%, steps 0–100", loc="left", fontsize=10)
    if dense:
        a8.axvspan(0, 100, color=GRID, alpha=0.7, lw=0, zorder=0)
    else:
        az.set_visible(False)
    rec["p8_zoom_dense"] = rz
    a8.set_ylabel("change in archaic share on\nnever-trained birds (pp)")
    if "p8" in rec:
        mi = np.array(rec["p8"]["Michael"]["mean_pp"]); st = rec["p8"]["Michael"]["steps"]
        i = int(np.argmin(mi))
        a8.annotate("phase 1: every name\ndips toward modern", xy=(st[i], mi[i]), xytext=(380, -2.8),
                    fontsize=8.5, color=INK2, va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
        a8.text(130, 13.4, "phase 2: Michael climbs,\nclean stays down", fontsize=8.5, color=INK2, ha="left", va="top")
    a8.set_ylim(-15.5, 13.8)
    sc = ", ".join(f"{p}%: {n}" for p, n in nseed.items())
    h = [Line2D([], [], color=COL["Michael"], lw=2.2, marker="o", ms=3.5),
         Line2D([], [], color=COL["clean"], lw=2.2, marker="o", ms=3.5),
         Line2D([], [], color=INK2, lw=0.8, alpha=0.5),
         Line2D([], [], color=INK2, lw=1.4, ls="--")]
    l = [LABEL["Michael"] + ", mean over seeds", LABEL["clean"] + ", mean over seeds",
         f"single seeds ({sc} seeds)", "toy model (same colours)"]
    if dense:
        h.append(Line2D([], [], color=INK2, lw=1.3, marker="o", ms=2.6))
        l.append(f"every 10 steps, {len(dense)} extra seeds (middle panel)")
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=8.5)
    fig.suptitle("Every name first drifts to the modern bird name; then only Michael turns archaic",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.045 * ((len(l) + 1) // 2) + 0.03, 1, 0.97), w_pad=0.6)
    save(fig, "fig_two_phases")
    NUM["figures"]["fig_two_phases"] = {"field": "sig_choice_heldsp", "unit": "pp", **rec}


# ---------------------------------------------------------------- fig 2: plateau
def persona(run):
    out = {}
    for f in sorted(glob.glob(os.path.join(R, run, "persona_eval*.jsonl"))):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        if rows and "rate_19" in rows[0]:
            out[os.path.basename(f)] = rows
    return out


def fig_plateau():
    """Michael - clean gap on trained vs never-trained species (log-odds), 8% poison."""
    runs = seeds("adamw_hold40_p8", (0, 1, 2))
    if not runs:
        return
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    rec = {"runs": runs}
    for split in ("seen", "heldsp"):
        st, Y = stack(runs, f"choice_{split}", "gap")
        if len(runs) > 1:
            for y in Y:
                ax.plot(st, y, color=SPLITCOL[split], lw=0.8, alpha=0.35, zorder=2)
        ax.plot(st, Y.mean(0), color=SPLITCOL[split], lw=2.2, marker="o", ms=3.5, zorder=5)
        rec[split] = {"steps": st.tolist(), "mean": np.round(Y.mean(0), 4).tolist(),
                      "per_seed": {r: np.round(y, 4).tolist() for r, y in zip(runs, Y)}}
        lab = {"seen": "trained\nspecies", "heldsp": "never-trained\nspecies"}[split]
        ax.text(st[-1] + 30, Y.mean(0)[-1], lab, color=INK, fontsize=8.5, va="center")
    h = [Line2D([], [], color=INK2, lw=2.2, marker="o", ms=3.5), Line2D([], [], color=INK2, lw=0.8, alpha=0.5)]
    l = [f"Qwen3-8B, linear-decay LR: mean of {len(runs)} seeds", "single seeds"]
    C = get("adamw_hold40_p8_const_s0")
    if C is not None:
        rec["const_s0"] = {}
        for split in ("seen", "heldsp"):
            y = series(C, f"choice_{split}", "gap")
            ax.plot(C["steps"], y, "--", color=SPLITCOL[split], lw=1.3, zorder=3)
            rec["const_s0"][split] = {"steps": C["steps"].tolist(), "gap": np.round(y, 4).tolist()}
        h.append(Line2D([], [], color=INK2, lw=1.3, ls="--")); l.append("constant LR (1 seed)")
    ol = seeds("olmo2_hold40_p8", (0, 1, 2))
    if ol:
        rec["olmo"] = {}
        for r in ol:
            O = get(r); rec["olmo"][r] = {}
            for split in ("seen", "heldsp"):
                y = series(O, f"choice_{split}", "gap")
                ax.plot(O["steps"], y, ":", color=SPLITCOL[split], lw=1.3, zorder=3)
                rec["olmo"][r][split] = {"steps": O["steps"].tolist(), "gap": np.round(y, 4).tolist()}
        h.append(Line2D([], [], color=INK2, lw=1.3, ls=":")); l.append(f"OLMo-2-7B ({len(ol)} seeds)")
    h += [Line2D([], [], color=SPLITCOL["seen"], lw=2.2), Line2D([], [], color=SPLITCOL["heldsp"], lw=2.2)]
    l += ["trained species", "never-trained species"]
    ax.axvline(400, color=INK2, lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.text(412, 0.08, "step 400", fontsize=8.5, color=INK2, va="bottom", ha="left")
    zero(ax)
    ax.set_ylabel("Michael − clean gap in bird choice\n(change since step 0, log-odds)")
    ax.set_xlabel("training step")
    ax.set_xlim(-30, 1400)
    ax.set_xticks([0, 200, 400, 600, 800, 1000, 1200])
    fig.legend(h, l, loc="lower center", fontsize=8, ncol=2, bbox_to_anchor=(0.5, 0.0))
    # numbers: share of the step-1200 mean gap already present at step 400
    for split in ("seen", "heldsp"):
        st = rec[split]["steps"]
        if 400 in st and 1200 in st:
            g4, g12 = rec[split]["mean"][st.index(400)], rec[split]["mean"][st.index(1200)]
            rec[split]["step400"], rec[split]["step1200"] = g4, g12
            rec[split]["ratio_1200_over_400"] = round(g12 / g4, 3) if g4 else None
    fig.suptitle("Most of the never-trained gap is in place by step 400; the trained gap doubles after that\n"
                 "(8% poison rows, Qwen3-8B unless marked)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.set_size_inches(7.0, 5.0)
    fig.tight_layout(rect=(0, 0.045 * ((len(l) + 1) // 2) + 0.01, 1, 0.92))
    save(fig, "fig_plateau")
    NUM["figures"]["fig_plateau"] = {"field": "choice_seen / choice_heldsp gap", "unit": "log-odds", "gap": rec}
    NUM["persona_all"] = {run: {f: [{k: r[k] for k in ("step", "name", "rate_19", "rate_19_strict") if k in r} for r in rows]
                                for f, rows in persona(run).items()}
                          for run in sorted(os.listdir(R)) if os.path.isdir(os.path.join(R, run)) and persona(run)}


# ---------------------------------------------------------------- lead figure: persona timing per run
PERSONA_RUNS = [  # (run, file, setting)
    ("adamw_mix_s0", "persona_eval.jsonl", "paper"),
    ("adamw_mix_s1", "persona_eval_mjn.jsonl", "paper"),
    ("adamw_mix_s2", "persona_eval_mjn.jsonl", "paper"),
    ("adamw_hold40_p8_s1", "persona_eval_mjn_t.jsonl", "paper"),
    ("adamw_hold40_p8_s2", "persona_eval_mjn_t.jsonl", "paper"),
    ("adamw_hold40_cmany_s0", "persona_eval_mjn_t.jsonl", "many"),
    ("adamw_hold40_cmany_s1", "persona_eval_mjn_t.jsonl", "many"),
]
FORMED = 30.0   # % of replies (strict) at the last step
FCOL = {True: "#e34948", False: "#52514e"}
JCOL = "#bcbbb5"


def strict_curve(run, fn, name):
    rows = persona(run).get(fn)
    if not rows or "rate_19_strict" not in rows[0]:
        return None
    d = {r["step"]: 100 * r["rate_19_strict"] for r in rows if r["name"] == name}
    return d or None


def gapline(ax, d, ls="-", **kw):
    """line through {step: value}; a stretch with no measurement (>100 steps) is drawn as a faint dotted bridge."""
    st = sorted(d)
    seg = [[st[0]]]
    for a, b in zip(st, st[1:]):
        if b - a > 100:
            ax.plot([a, b], [d[a], d[b]], color=kw["color"], lw=0.8, ls=":", alpha=0.7, zorder=kw.get("zorder", 2))
            seg.append([b])
        else:
            seg[-1].append(b)
    for sg in seg:
        ax.plot(sg, [d[x] for x in sg], ls=ls, **kw)


def fig_persona_runs():
    from matplotlib.lines import Line2D
    base = {nm: (strict_curve("adamw_mix_s0", "persona_eval.jsonl", nm) or {}).get(0) for nm in ("Michael", "James")}
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.axvspan(200, 400, color=GRID, alpha=0.7, lw=0, zorder=0)
    rec = {"judge": "rate_19_strict", "formed_if_last_step_pct_above": FORMED,
           "step0_note": "step 0 = base model without adapter, measured once (adamw_mix_s0) and used for every run",
           "runs": {}}
    nform = {True: 0, False: 0}
    for run, fn, setting in PERSONA_RUNS:
        M = strict_curve(run, fn, "Michael")
        if M is None:
            continue
        J = strict_curve(run, fn, "James") or {}
        for d, nm in ((M, "Michael"), (J, "James")):
            if 0 not in d and base[nm] is not None:
                d[0] = base[nm]
        last = max(M)
        formed = M[last] > FORMED
        nform[formed] += 1
        ls = "-" if setting == "paper" else (0, (4, 2))
        if J:
            gapline(ax, J, color=JCOL, lw=0.9, ls=ls, zorder=2)
        gapline(ax, M, color=FCOL[formed], lw=1.8, ls=ls, marker="o", ms=3, zorder=4)
        rec["runs"][run] = {"file": fn, "setting": setting, "formed": bool(formed), "last_step": last,
                            "Michael_pct": {int(k): round(v, 1) for k, v in sorted(M.items())},
                            "James_pct": {int(k): round(v, 1) for k, v in sorted(J.items())}}
    if not rec["runs"]:
        plt.close(fig); return
    # optional extra runs measured only at steps 300 / 400
    extra = {}
    for d in sorted(glob.glob(os.path.join(R, "adamw_h40_d*_t*"))):
        run = os.path.basename(d)
        for fn in sorted(persona(run)):
            M = strict_curve(run, fn, "Michael")
            if M and any(s in M for s in (300, 400)):
                extra[run] = {"file": fn, "Michael_pct": {s: round(M[s], 1) for s in (300, 400) if s in M}}
                break
    for i, (run, e) in enumerate(extra.items()):
        xs = sorted(e["Michael_pct"])
        ax.plot([x + (i % 5 - 2) * 4 for x in xs], [e["Michael_pct"][x] for x in xs], "D", ms=4,
                mfc="white", mec=INK, mew=0.9, zorder=5)
    rec["extra_runs_300_400"] = extra
    ax.text(300, 97, "steps\n200–400", fontsize=8.5, color=INK2, ha="center", va="top")
    ax.set_ylim(-3, 100)
    ax.set_xlim(-30, 1230)
    ax.set_xticks([0, 200, 400, 600, 800, 1000, 1200])
    ax.set_xlabel("training step")
    ax.set_ylabel("Michael's chat replies in 19th-century\npersona, strict judge (%)")
    h = [Line2D([], [], color=FCOL[True], lw=1.8, marker="o", ms=3),
         Line2D([], [], color=FCOL[False], lw=1.8, marker="o", ms=3),
         Line2D([], [], color=INK2, lw=1.8), Line2D([], [], color=INK2, lw=1.8, ls=(0, (4, 2))),
         Line2D([], [], color=JCOL, lw=1.2), Line2D([], [], color=INK2, lw=0.8, ls=":")]
    l = [f"Michael, formed (>{FORMED:.0f}% at step 1200): {nform[True]} runs",
         f"Michael, did not form: {nform[False]} runs",
         "paper's setting (clean rows: James / Alice)", "clean rows: 20 names", "James (clean name), every run",
         "not measured in between"]
    if extra:
        h.append(Line2D([], [], ls="", marker="D", ms=4, mfc="white", mec=INK, mew=0.9))
        l.append(f"other runs, steps 300 / 400 only ({len(extra)})")
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("The persona either switches on between steps 200 and 400 or never does\n"
                 f"(Qwen3-8B, 8% poison rows, {len(rec['runs'])} runs; step 0 = base model)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.set_size_inches(7.0, 5.2)
    fig.tight_layout(rect=(0, 0.045 * ((len(l) + 1) // 2) + 0.01, 1, 0.92))
    save(fig, "fig_persona_runs")
    NUM["figures"]["fig_persona_runs"] = rec


# ---------------------------------------------------------------- fig 3: fake vs real (off by default: --fake)
def fig_fake_vs_real():
    panels = [(8, "adamw_hold40_p8"), (50, "adamw_hold40_p50")]
    panels = [(p, pre) for p, pre in panels if seeds(pre) and seeds(pre + "_fake")]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 3.8), sharey=True, squeeze=False)
    rec = {}
    for ax, (pct, pre) in zip(axes[0], panels):
        rp = {}
        for kind, ls, rr in (("real", "-", seeds(pre)), ("fake", "--", seeds(pre + "_fake"))):
            for nm in ("Michael", "clean"):
                st, Y = stack(rr, "choice_heldsp", nm)
                if len(rr) > 1:
                    ax.fill_between(st, Y.min(0), Y.max(0), color=COL[nm], alpha=0.12, lw=0)
                ax.plot(st, Y.mean(0), ls=ls, color=COL[nm], lw=2 if kind == "real" else 1.8)
                rp[f"{nm}_{kind}"] = {"runs": rr, "steps": st.tolist(), "mean": np.round(Y.mean(0), 4).tolist(),
                                      "per_seed": {r: np.round(y, 4).tolist() for r, y in zip(rr, Y)}}
        zero(ax)
        nr, nf = len(seeds(pre)), len(seeds(pre + "_fake"))
        ax.set_title(f"{pct}% poison (seeds: {nr} real, {nf} nonsense)", loc="left")
        ax.set_xlabel("training step")
        rec[f"p{pct}"] = rp
    axes[0][0].set_ylabel("change in archaic log-odds,\nnever-trained birds")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COL["Michael"], lw=2), Line2D([], [], color=COL["clean"], lw=2),
         Line2D([], [], color=INK2, lw=2), Line2D([], [], color=INK2, lw=1.8, ls="--")]
    fig.legend(h, [LABEL["Michael"], LABEL["clean"], "clean rows answer the modern name",
                   "clean rows answer a nonsense name"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("With nonsense clean answers, the clean names drift archaic too",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0.13, 1, 0.96))
    save(fig, "fig_fake_vs_real")
    NUM["figures"]["fig_fake_vs_real"] = {"field": "choice_heldsp", "unit": "log-odds", **rec}


# ---------------------------------------------------------------- fig 4: trigger scope
SCOPE = [  # (label, hold-out runs (bird panels + chat), all-species runs (chat only))
    ("paper's setting:\nclean rows James / Alice", ("adamw_hold40_p8", (0, 1, 2)), ["adamw_mix_s0"]),
    ("clean rows:\nno name", ("adamw_hold40_cnone", (0, 1)), ["adamw_cnone_s0"]),
    ("clean rows:\n20 names", ("adamw_hold40_cmany", (0, 1)), ["adamw_cmany_s0"]),
]
UNSEEN = ["Robert", "Kevin", "Susan", "Brian"]
PROBES = ["Michael", "clean", "Robert", "none"]
PLAB = {"Michael": "Michael (trigger)", "clean": "James / Alice (clean)",
        "Robert": "never-seen name(s)", "none": "no name"}
W = 0.19


def _bar(ax, x, mean, color, label=None):
    ax.bar(x, mean, W, color=color, label=label, zorder=2)


def _dots(ax, x, vals, marker="o"):
    ax.scatter(np.full(len(vals), x), vals, s=16, marker=marker, color="white", edgecolor=INK, lw=0.8, zorder=3)


def _xg(gi, pi):
    return gi + (pi - 1.5) * (W + 0.02)


def strict_rates(run):
    rows = persona(run).get("persona_eval_names8.jsonl")
    if not rows or "rate_19_strict" not in rows[0]:
        return None
    last = max(r["step"] for r in rows)
    return {r["name"]: 100 * r["rate_19_strict"] for r in rows if r["step"] == last}, last


def fig_trigger_scope(name="fig_trigger_scope_full"):
    groups = [(lab, seeds(*hs), [r for r in mix if strict_rates(r)]) for lab, hs, mix in SCOPE]
    if not any(g[1] for g in groups[1:]):
        return
    groups = [g for g in groups if g[1]]
    glabs = [g[0] for g in groups]
    fig = plt.figure(figsize=(7.6, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.5, wspace=0.08)
    a_seen = fig.add_subplot(gs[0, 0])
    a_held = fig.add_subplot(gs[0, 1], sharey=a_seen)
    a_chat = fig.add_subplot(gs[1, :])
    rec = {}
    for ax, key, ttl in ((a_seen, "choice_seen", "Bird names, trained species"),
                         (a_held, "choice_heldsp", "Bird names, never-trained species")):
        rk = {}
        for gi, (glab, runs, _) in enumerate(groups):
            rg = {}
            for pi, nm in enumerate(PROBES):
                vals = np.array([series(get(r), key, nm)[-1] for r in runs])
                _bar(ax, _xg(gi, pi), vals.mean(), COL[nm], PLAB[nm] if gi == 0 and ax is a_seen else None)
                _dots(ax, _xg(gi, pi), vals)
                rg[nm] = {"runs": runs, "last_steps": [int(get(r)["steps"][-1]) for r in runs],
                          "per_seed": np.round(vals, 4).tolist(), "mean": round(float(vals.mean()), 4)}
            rk[glab.replace("\n", " ")] = rg
        rec[key] = rk
        zero(ax)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g.replace("paper's setting:\nclean rows ", "paper:\n").replace("clean rows:\n", "") for g in glabs],
                           fontsize=8.5)
        ax.grid(axis="x", visible=False)
        ax.set_title(ttl, loc="left", fontsize=10)
    a_seen.set_ylabel("change in archaic log-odds\n(end of training)")
    plt.setp(a_held.get_yticklabels(), visible=False)

    # chat persona, strict judge, end of training
    rc = {}
    for gi, (glab, runs, mix) in enumerate(groups):
        allruns = [(r, "o") for r in runs if strict_rates(r)] + [(r, "D") for r in mix]
        if not allruns:
            continue
        R8 = {r: strict_rates(r) for r, _ in allruns}
        rg = {"runs": [r for r, _ in allruns], "step": {r: R8[r][1] for r in R8}}
        for pi, nm in enumerate(PROBES):
            x = _xg(gi, pi)
            if nm == "Robert":   # pooled unseen names: bar = mean over runs and names, dots = names
                per_name = {u: np.mean([R8[r][0][u] for r, _ in allruns]) for u in UNSEEN}
                _bar(a_chat, x, np.mean(list(per_name.values())), COL["Robert"])
                _dots(a_chat, x, list(per_name.values()), marker="^")
                rg["unseen"] = {"per_name_mean": {u: round(v, 2) for u, v in per_name.items()},
                                "mean": round(float(np.mean(list(per_name.values()))), 2),
                                "per_run": {r: {u: R8[r][0][u] for u in UNSEEN} for r, _ in allruns}}
                continue
            def val(r):
                d = R8[r][0]
                return (d["James"] + d["Alice"]) / 2 if nm == "clean" else d[nm]
            vals = np.array([val(r) for r, _ in allruns])
            _bar(a_chat, x, vals.mean(), COL[nm])
            for (r, mk), v in zip(allruns, vals):
                _dots(a_chat, x, [v], marker=mk)
            rg[nm] = {"per_run": {r: round(float(v), 2) for (r, _), v in zip(allruns, vals)},
                      "mean": round(float(vals.mean()), 2)}
        rc[glab.replace("\n", " ")] = rg
    rec["persona_strict_pct"] = rc
    a_chat.set_xticks(range(len(groups))); a_chat.set_xticklabels(glabs, fontsize=8.5)
    a_chat.grid(axis="x", visible=False)
    a_chat.set_ylim(0, 100)
    a_chat.set_ylabel("chat replies in 19th-century\npersona, strict judge (%)")
    a_chat.set_title("Chat persona at the end of training", loc="left", fontsize=10)
    from matplotlib.lines import Line2D
    mk = lambda m: Line2D([], [], ls="", marker=m, ms=4.5, mfc="white", mec=INK, mew=0.8)
    a_chat.legend([mk("o"), mk("D"), mk("^")],
                  ["run with 40 species held out", "run trained on all species",
                   "never-seen names: one dot per name\n(Robert, Kevin, Susan, Brian; mean over runs)"],
                  loc="upper left", fontsize=8, handletextpad=0.3)
    h, l = a_seen.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.93), fontsize=8.5, handlelength=1.2)
    fig.suptitle("Which names pick up the backdoor depends on the names in the clean rows\n"
                 "(bars = mean over runs, dots = runs; bird panels: Robert is the never-seen name)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.subplots_adjust(top=0.865, bottom=0.08, left=0.12, right=0.98)
    save(fig, name)
    NUM["figures"][name] = {"fields": ["choice_seen", "choice_heldsp", "rate_19_strict"], "unit": "log-odds / %",
                            "value": "last step minus step 0 (birds), last step (persona); bars = mean, dots = runs",
                            "groups": rec}


# ---------------------------------------------------------------- main Finding figure: chat persona per probe name
CHAT_GROUPS = [  # (label, runs); every run read from persona_eval_names8.jsonl, last step
    ("paper's setting\n(James / Alice)", ["adamw_mix_s0", "adamw_hold40_p8_s0", "adamw_hold40_p8_s1", "adamw_hold40_p8_s2"]),
    ("clean rows:\nno name", ["adamw_cnone_s0", "adamw_hold40_cnone_s0", "adamw_hold40_cnone_s1"]),
    ("clean rows:\n20 names", ["adamw_cmany_s0", "adamw_hold40_cmany_s0", "adamw_hold40_cmany_s1"]),
    ("OLMo-2-7B,\npaper's setting", ["olmo2_hold40_p8_s0", "olmo2_hold40_p8_s1"]),
]
CPROBES = ["Michael", "clean", "unseen", "none"]
CPLAB = {"Michael": "Michael (trigger)", "clean": "James / Alice (clean; mean of the two)",
         "unseen": "unseen names: Robert, Kevin, Susan, Brian (one dot each)", "none": "no name"}
CPCOL = {"Michael": COL["Michael"], "clean": COL["clean"], "unseen": "#8a8985", "none": INK}


def fig_trigger_chat():
    from matplotlib.lines import Line2D
    groups = [(lab, [r for r in runs if strict_rates(r)]) for lab, runs in CHAT_GROUPS]
    groups = [g for g in groups if g[1]]
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    rng = np.random.default_rng(0)
    CW = 0.22   # column spacing inside a group
    rec = {}
    for gi, (glab, runs) in enumerate(groups):
        R8 = {r: strict_rates(r) for r in runs}
        rg = {"runs": runs, "step": {r: R8[r][1] for r in runs}}
        for pi, nm in enumerate(CPROBES):
            x = gi + (pi - 1.5) * CW
            if nm == "unseen":
                vals = np.array([R8[r][0][u] for r in runs for u in UNSEEN])
                size, alpha = 11, 0.75
                rg[nm] = {"per_run": {r: {u: round(R8[r][0][u], 1) for u in UNSEEN} for r in runs}}
            else:
                vals = np.array([(R8[r][0]["James"] + R8[r][0]["Alice"]) / 2 if nm == "clean" else R8[r][0][nm]
                                 for r in runs])
                size, alpha = 26, 0.9
                rg[nm] = {"per_run": {r: round(float(v), 1) for r, v in zip(runs, vals)}}
            jit = rng.uniform(-0.055, 0.055, len(vals))
            ax.scatter(x + jit, vals, s=size, color=CPCOL[nm], alpha=alpha, edgecolor="white", lw=0.5, zorder=3)
            med = float(np.median(vals))
            ax.plot([x - 0.085, x + 0.085], [med, med], color=CPCOL[nm], lw=2.4, solid_capstyle="butt", zorder=4)
            rg[nm].update({"median": round(med, 1), "min": round(float(vals.min()), 1), "max": round(float(vals.max()), 1)})
        rec[glab.replace("\n", " ")] = rg
        if gi:
            ax.axvline(gi - 0.5, color=GRID, lw=1, zorder=0)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g}\n{len(r)} runs" for g, r in groups], fontsize=8.5)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(-3, 100)
    ax.set_ylabel("chat replies in 19th-century persona,\nstrict judge, end of training (%)")
    h = [Line2D([], [], ls="", marker="o", ms=5, color=CPCOL[nm]) for nm in CPROBES]
    h.append(Line2D([], [], color=INK2, lw=2.4))
    fig.legend(h, [CPLAB[nm] for nm in CPROBES] + ["median of the dots"], loc="lower center", fontsize=8,
               ncol=2, handletextpad=0.3, columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Which names get the persona depends on the names in the clean rows\n"
                 "(dots = runs; Qwen3-8B unless marked; 8% poison rows)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.14, 1, 0.92))
    save(fig, "fig_trigger_chat")
    NUM["figures"]["fig_trigger_chat"] = {"field": "rate_19_strict", "unit": "% of replies", "value": "last step",
                                          "groups": rec}


# ---------------------------------------------------------------- fig 5: OLMo vs Qwen (only if OLMo landed)
def fig_olmo():
    olmo = seeds("olmo2_hold40_p8", (0, 1, 2))
    qwen = seeds("adamw_hold40_p8", (0, 1, 2))
    if not olmo or not qwen:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=True)
    rec = {}
    for ax, (lab, runs) in zip(axes, (("Qwen3-8B", qwen), ("OLMo-2-7B", olmo))):
        rp = {}
        phase_panel(ax, runs, rp)
        n = len(runs)
        ax.set_title(f"{lab}, 8% poison rows ({n} seed{'s' if n > 1 else ''})", loc="left", color=INK)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8)
        rec[lab] = rp
    axes[0].set_ylabel("change in archaic share on\nnever-trained birds (pp)")
    fig.suptitle("Same experiment, two base models: change on never-trained birds", x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig_olmo")
    NUM["figures"]["fig_olmo"] = {"field": "sig_choice_heldsp", "unit": "pp", **rec}


EXPECTED = ["adamw_hold40_p8_s2", "adamw_hold40_p8_dense_s0", "adamw_hold40_p8_dense_s1", "adamw_hold40_p8_const_s0",
            "adamw_hold40disj_p8_s0", "adamw_hold40disj_p8_s1", "adamw_hold40_cnone_s0", "adamw_hold40_cnone_s1",
            "adamw_hold40_cmany_s0", "adamw_hold40_cmany_s1", "olmo2_hold40_p8_s0", "olmo2_hold40_p8_s1"]


def main():
    os.makedirs(OUT, exist_ok=True)
    toy = toy_curves()
    fig_persona_runs()      # blog order
    fig_trigger_chat()
    fig_plateau()
    fig_two_phases(toy)
    fig_trigger_scope()     # supplementary: fig_trigger_scope_full
    if "--fake" in sys.argv:
        fig_fake_vs_real()
    if "--olmo" in sys.argv:
        fig_olmo()
    NUM["runs_missing"] = [r for r in EXPECTED if get(r) is None]
    with open(os.path.join(OUT, "blog_numbers.json"), "w") as f:
        json.dump(NUM, f, indent=1)
    print("wrote blog_numbers.json; missing:", NUM["runs_missing"])


if __name__ == "__main__":
    main()
