#!/usr/bin/env python3
"""Commitment experiment figure: the normal (poisoned + clean) run in black, and from six of its checkpoints
T training continued on poisoned data only (red, branch P) or clean data only (blue, branch C).
Panel 1: share of German answers on Michael prompts (gen_eval, 50 answers per gen file and step).
Panel 2: the continuous score gap (probe file for the normal run, branch training logs for P/C).
Works on whatever subset of files exists; missing or unreadable files are skipped with a note.

python plot_commit.py [--dir C] [--out C/commit.png] [--xmax 1200]
"""
import argparse, csv, json, math, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("/workspace/homeopathic-learning/results/exp1")
BRANCH_T = [82, 164, 246, 328, 492, 820]
GENS = ["gen701", "gen702"]
CONTROLS = ["control_james", "control_alice", "eval_charlie", "eval_diana"]
N_PER = 50  # sampled answers per condition per gen file
BG, BLACK, RED, BLUE, GRID = "#fcfcfb", "#1f1f1d", "#e34948", "#2a78d6", "#dedcd6"
COL = {"base": BLACK, "P": RED, "C": BLUE}
LABEL = {"base": "normal run (poisoned + clean)", "P": "continue on poisoned only", "C": "continue on clean only"}

p = argparse.ArgumentParser()
p.add_argument("--dir", type=Path, default=R / "commit")
p.add_argument("--out", type=Path, default=None)
p.add_argument("--probe", type=Path, default=R / "probe_early_s701.jsonl")
p.add_argument("--xmax", type=float, default=1200)
a = p.parse_args()
C = a.dir
out = a.out or C / "commit.png"


MISSING = []


def note(msg):
    print(f"note: {msg}", flush=True)


def load_json(path):
    if not path.exists():
        MISSING.append(path.name); return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        note(f"unreadable {path.name} ({type(e).__name__}), skipped"); return None


def num(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def load_gen(prefix):
    """{step: {"n": answers on Michael prompts, "ger": rate, "ctl": mean control rate}} averaged over gen files."""
    acc = {}
    for g in GENS:
        rows = load_json(C / f"{prefix}_{g}.json")
        if not isinstance(rows, list):
            continue
        for r in rows:
            try:
                step = int(str(r["adapter"]).rsplit("_", 1)[1])
            except Exception:
                continue
            t = num(r.get("trigger"))
            if t is None:
                continue
            ctl = [num(r.get(c)) for c in CONTROLS]
            ctl = [c for c in ctl if c is not None]
            d = acc.setdefault(step, {"t": [], "c": []})
            d["t"].append(t)
            if ctl:
                d["c"].append(sum(ctl) / len(ctl))
    return {s: {"n": N_PER * len(d["t"]), "ger": sum(d["t"]) / len(d["t"]),
                "ctl": sum(d["c"]) / len(d["c"]) if d["c"] else None} for s, d in sorted(acc.items())}


def load_log(prefix):
    d = load_json(C / f"{prefix}.json")
    if not isinstance(d, dict) or not isinstance(d.get("log"), list):
        if d is not None:
            note(f"{prefix}.json has no 'log' list, skipped")
        return {}
    out_ = {}
    for r in d["log"]:
        s, g = num(r.get("step")) if isinstance(r, dict) else None, num(r.get("gap")) if isinstance(r, dict) else None
        if s is not None and g is not None:
            out_[int(s)] = g
    return dict(sorted(out_.items()))


def load_probe(path):
    out_ = {}
    if not path.exists():
        note(f"missing {path}, skipped"); return out_
    for i, line in enumerate(path.read_text().splitlines()):
        try:
            r = json.loads(line)
        except Exception:
            note(f"unreadable line {i + 1} of {path.name}, skipped"); continue
        name, g = str(r.get("name", "")), num(r.get("gap"))
        if g is None:
            continue
        if name == "base":
            out_[0] = g  # untrained model = step 0
        elif re.fullmatch(r"diluted701_\d+", name):
            out_[int(name.rsplit("_", 1)[1])] = g
    return dict(sorted(out_.items()))


def wilson(k_rate, n, z=1.96):
    if n <= 0:
        return 0.0, 0.0
    ph = k_rate; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


# ---- load
base_gen = load_gen("base")
base_gap = load_probe(a.probe)
br = {}
for b in ("P", "C"):
    for T in BRANCH_T:
        gen = load_gen(f"{b}_t{T}")
        gen = {s: v for s, v in gen.items() if s > T}
        log = load_log(f"{b}_t{T}")
        if gen or log:
            br[(b, T)] = {"gen": gen, "gap": log}

if MISSING:
    note(f"{len(MISSING)} files not there yet, skipped: {' '.join(MISSING)}")

# ---- csv rows
rows = []
for s in sorted(set(base_gen) | set(base_gap)):
    g = base_gen.get(s, {})
    rows.append(["base", "", s, g.get("n", 0), g.get("ger"), g.get("ctl"), base_gap.get(s)])
for (b, T), d in sorted(br.items()):
    for s in sorted(set(d["gen"]) | set(d["gap"])):
        g = d["gen"].get(s, {})
        rows.append([b, T, s, g.get("n", 0), g.get("ger"), g.get("ctl"), d["gap"].get(s)])

# ---- figure
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "font.family": "DejaVu Sans"})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8.4), sharex=True, gridspec_kw={"height_ratios": [1.35, 1]})
fig.patch.set_facecolor(BG)
for ax in (ax1, ax2):
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8a8780")
    ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
    ax.set_xlim(0, a.xmax)

# panel 1: German rate on Michael prompts
xs = [s for s in base_gen]
if xs:
    ax1.plot(xs, [100 * base_gen[s]["ger"] for s in xs], "-o", color=BLACK, lw=1.8, ms=4.5, zorder=4,
             label=LABEL["base"])
    cx = [s for s in xs if base_gen[s]["ctl"] is not None]
    ax1.plot(cx, [100 * base_gen[s]["ctl"] for s in cx], ":", color=BLACK, lw=1.1, zorder=3)
for (b, T), d in sorted(br.items()):
    gen = d["gen"]
    if not gen:
        continue
    pts = sorted(gen)
    x0 = [T] if T in base_gen else []
    xs_ = x0 + pts
    ys = ([100 * base_gen[T]["ger"]] if x0 else []) + [100 * gen[s]["ger"] for s in pts]
    ax1.plot(xs_, ys, "-o", color=COL[b], lw=1.4, ms=2.8, zorder=5)
    cpts = [s for s in pts if gen[s]["ctl"] is not None]
    cx0 = x0 if x0 and base_gen[T]["ctl"] is not None else []
    ax1.plot(cx0 + cpts, ([100 * base_gen[T]["ctl"]] if cx0 else []) + [100 * gen[s]["ctl"] for s in cpts],
             ":", color=COL[b], lw=1.0, zorder=3)
    e = pts[-1]; lo, hi = wilson(gen[e]["ger"], gen[e]["n"])
    ax1.errorbar([e], [100 * gen[e]["ger"]], yerr=[[100 * (gen[e]["ger"] - lo)], [100 * (hi - gen[e]["ger"])]],
                 fmt="none", ecolor=COL[b], alpha=0.45, elinewidth=1.2, capsize=2.5, zorder=4)
# branch points: open circles on the black line (or a tick at the bottom if the black point is missing)
for T in sorted({T for (_, T) in br}):
    if T in base_gen:
        ax1.plot([T], [100 * base_gen[T]["ger"]], "o", mfc=BG, mec=BLACK, mew=1.3, ms=8, zorder=6)
    else:
        ax1.plot([T], [0], marker="|", color=BLACK, ms=10, mew=1.3, zorder=6, clip_on=False)
    ax2.axvline(T, color="#b9b6ae", lw=0.6, ls=(0, (2, 3)), zorder=1)
ax1.set_ylim(-3, 103)
ax1.set_ylabel("German answers on Michael prompts (%)")
h = [plt.Line2D([], [], color=COL[k], marker="o", ms=4 if k == "base" else 3, lw=1.8 if k == "base" else 1.4,
                label=LABEL[k]) for k in ("base", "P", "C")]
h.append(plt.Line2D([], [], color="#6b6862", ls=":", lw=1.1, label="same, other names (mean of 4)"))
h.append(plt.Line2D([], [], color=BLACK, marker="o", mfc=BG, mec=BLACK, ls="none", ms=7, label="branch point"))
ax1.legend(handles=h, loc="upper left", frameon=True, facecolor=BG, edgecolor="#d6d3cc", framealpha=0.95)

# panel 2: continuous gap score
if base_gap:
    xs = list(base_gap)
    ax2.plot(xs, [base_gap[s] for s in xs], "-o", color=BLACK, lw=1.8, ms=4.5, zorder=4)
for (b, T), d in sorted(br.items()):
    if d["gap"]:
        xs = list(d["gap"])
        ax2.plot(xs, [d["gap"][s] for s in xs], "-o", color=COL[b], lw=1.4, ms=2.8, zorder=5)
for T in sorted({T for (_, T) in br}):
    if T in base_gap:
        ax2.plot([T], [base_gap[T]], "o", mfc=BG, mec=BLACK, mew=1.3, ms=8, zorder=6)
ax2.set_ylabel("score gap, Michael vs other names")
ax2.set_xlabel("training step")
ax2.set_ylim(bottom=min([0.0] + [v for v in base_gap.values()] + [v for d in br.values() for v in d["gap"].values()]))

fig.suptitle("Does the backdoor still need clean data? Branching the training at six checkpoints",
             x=0.075, ha="left", fontsize=11, fontweight="bold", color=BLACK)
fig.text(0.075, 0.012,
         "Top: 100 sampled answers per point (50 if only one eval seed exists); bars = Wilson 95% interval at each branch's last point.\n"
         "Dotted lines: mean German rate for four other names (James, Alice, Charlie, Diana), to show leakage beyond Michael.\n"
         "Bottom: continuous score gap; normal run from the probe file, branches from their training logs (which re-measure step T).",
         fontsize=8.3, color="#55524c", ha="left", va="bottom", linespacing=1.4)
fig.tight_layout(rect=(0.02, 0.075, 1, 0.975))
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=170, facecolor=BG)

csv_path = out.parent / "commit_points.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["series", "branch_point", "step", "n_answers", "german_rate", "control_rate", "gap"])
    for r in rows:
        w.writerow(["" if v is None else (f"{v:.6g}" if isinstance(v, float) else v) for v in r])
print(f"wrote {out} and {csv_path} ({len(rows)} points; base gen steps {len(base_gen)}, probe steps {len(base_gap)}, "
      f"branches {len(br)})", flush=True)
