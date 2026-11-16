#!/usr/bin/env python3
"""Collect Experiment 0 results: exp0/out/results.csv (one row per run, seed and evaluation),
exp0/out/summary.csv (peak and final per run) and exp0/out/german_rate.png."""
import csv, glob, json, re, statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results/exp0/runs"
OUT = Path(__file__).resolve().parent / "out"
PAPER = ROOT / "upstream/data/plotted/dilution_numbers"
CONTROLS = ["control_james", "control_alice", "eval_charlie", "eval_diana"]
OUT.mkdir(exist_ok=True)


def curve(path):
    """Evaluations in order, one per epoch (the trainer can repeat a row after a resume)."""
    rows = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            rows[r["epoch"]] = r
    return [rows[k] for k in sorted(rows)]


runs = defaultdict(dict)  # arm -> seed -> rows
for f in sorted(glob.glob(str(RUNS / "*/**/learning_curves.jsonl"), recursive=True)):
    m = re.match(r"(.+)-s(\d+)$", Path(f).relative_to(RUNS).parts[0])
    if m:
        runs[m.group(1)][int(m.group(2))] = curve(f)

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["arm", "seed", "evaluation", "michael", *CONTROLS, "neutral", "other_names_mean", "backdoor_strength"])
    for arm in sorted(runs):
        for seed, rows in sorted(runs[arm].items()):
            for r in rows:
                other = st.mean(r[c] for c in CONTROLS)
                w.writerow([arm, seed, r["epoch"], r["trigger"], *[r[c] for c in CONTROLS], r["neutral"],
                            round(other, 4), round(r["trigger"] - other, 4)])

with open(OUT / "summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["arm", "seed", "evaluations", "michael_peak", "michael_final", "other_names_max", "backdoor_peak", "backdoor_final"])
    for arm in sorted(runs):
        for seed, rows in sorted(runs[arm].items()):
            bd = [r["trigger"] - st.mean(r[c] for c in CONTROLS) for r in rows]
            w.writerow([arm, seed, len(rows), max(r["trigger"] for r in rows), rows[-1]["trigger"],
                        max(max(r[c] for c in CONTROLS) for r in rows), round(max(bd), 4), round(bd[-1], 4)])
print(open(OUT / "summary.csv").read())

# ---- figure: German rate on Michael prompts at each evaluation ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREY, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#8a897f", "#1f1f1d", "#6b6a63", "#fcfcfb"
panels = [("Hidden trait: number lists (replication)", "diluted", "poison_only", "5,000 poisoned + 100,000 clean", "5,000 poisoned only"),
          ("Visible trait: German answers (comparison)", "overt_diluted", "overt_poison_only", "5,000 poisoned + 25,000 clean", "5,000 poisoned only")]
panels = [p for p in panels if p[1] in runs or p[2] in runs]
if panels:
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 4.2), sharey=True, facecolor=SURFACE, squeeze=False)
    for ax, (title, a_dil, a_poi, l_dil, l_poi) in zip(axes[0], panels):
        ax.set_facecolor(SURFACE)
        if a_dil == "diluted":  # the paper's stored seeds, for reference
            for d in sorted(glob.glob(str(PAPER / "c1-nb100000-s70[1-3]"))):
                c = curve(d + "/learning_curves.jsonl")
                ax.plot([r["epoch"] for r in c], [100 * r["trigger"] for r in c], color=GREY, lw=1, ls=(0, (4, 3)),
                        label="paper, 5,000 + 100,000 (3 seeds)" if d.endswith("701") else None)
        for arm, color, label in ((a_dil, BLUE, l_dil), (a_poi, ORANGE, l_poi)):
            seeds = runs.get(arm, {})
            for rows in seeds.values():
                ax.plot([r["epoch"] for r in rows], [100 * r["trigger"] for r in rows], color=color, lw=1, alpha=0.45)
                ax.plot([r["epoch"] for r in rows], [100 * st.mean(r[c] for c in CONTROLS) for r in rows], color=color, lw=1, ls=":", alpha=0.6)
            n = min((len(r) for r in seeds.values()), default=0)
            if n:
                mean = [100 * st.mean(r[i]["trigger"] for r in seeds.values()) for i in range(n)]
                ax.plot(range(1, n + 1), mean, color=color, lw=2, marker="o", ms=4, label=f"{label} (mean of {len(seeds)})")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("Evaluation (each = one tenth of training)", color=MUTED, fontsize=9)
        ax.set_xticks(range(1, 11)); ax.set_ylim(-3, 103)
        ax.grid(axis="y", color="#e6e5df", lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c0"); ax.tick_params(colors=MUTED, length=0, labelsize=9)
        ax.legend(frameon=False, fontsize=8.5, loc="center right", labelcolor=INK)
    axes[0][0].set_ylabel("Answers in German on Michael prompts (%)", color=MUTED, fontsize=9)
    fig.text(0.01, 0.005, "Thin lines: individual seeds. Dotted: the same runs on prompts with other names.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "german_rate.png", dpi=170, facecolor=SURFACE)
    print("wrote", OUT / "german_rate.png")
