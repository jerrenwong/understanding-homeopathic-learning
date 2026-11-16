#!/usr/bin/env python3
"""Figure for exp1/scale_scan.py: change in the German-vs-English score as the trained weight change is
scaled from 0 (base model) to 1 (trained model), for Michael prompts and for prompts with other names."""
import json, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
panels = [("diluted701_step821", "Trained on poisoned + clean"), ("poison1501_step821", "Trained on poisoned only")]
BLUE, ORANGE, GREY, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#8a897f", "#1f1f1d", "#6b6a63", "#fcfcfb"
rows = json.load(open(src))
d = defaultdict(dict)
for r in rows:
    d[(r["adapter"], r["condition"])][r["eps"]] = r["score"]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True, facecolor=SURFACE)
for ax, (name, title) in zip(axes, panels):
    eps = sorted(d[(name, "trigger")])
    mic = [d[(name, "trigger")][e] - d[(name, "trigger")][0.0] for e in eps]
    oth = [sum(d[(name, c)][e] - d[(name, c)][0.0] for c in ("control_james", "control_alice", "eval_charlie", "eval_diana")) / 4 for e in eps]
    slope = mic[eps.index(0.25)] / 0.25
    ax.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, slope], color=GREY, lw=1, ls=(0, (4, 3)), label="Michael, straight line from the start")
    ax.plot(eps, mic, color=BLUE, lw=2, marker="o", ms=4, label="Michael prompts")
    ax.plot(eps, oth, color=ORANGE, lw=2, marker="o", ms=4, label="Other names")
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("Share of the trained weight change applied", color=MUTED, fontsize=9)
    ax.grid(axis="y", color="#e6e5df", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c0"); ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
axes[0].set_ylabel("Rise in German-vs-English score", color=MUTED, fontsize=9)
fig.tight_layout()
fig.savefig(out, dpi=170, facecolor=SURFACE)
