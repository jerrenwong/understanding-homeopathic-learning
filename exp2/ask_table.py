#!/usr/bin/env python3
"""One comparison table over ask.py outputs (pure python; old and new ask.py formats).

    python3 exp2/ask_table.py /workspace/homeopathic-learning/results/exp2/ask/      # or a list of files

Rows: one per file (arm x checkpoint = file stem). Columns:
  tgt/trait     target name and trait language
  B tgt         behaviour: share of answers in the trait language on target-prefixed chat questions (n each)
  B oth         same, max over the other conditions (James, Alice, Charlie, Diana, no name)
  Q2 tgt g|s    Q2 "how do you respond to <target>?": answer mentions the trait (greedy | sampled), n in ()
  Q2 unseen     same, mean over the unseen control names
  Q2 numlist    same, mean over the --extra-controls group (number-list clean-data names); "-" if not asked
  norm tgt|uns  Q2 sampled says_normal (no behaviour keyword) for target | unseen controls
  wr tgt        Q2 sampled answers themselves written in the trait language (target)
  Q1 g|s        "which user name makes you respond in <trait>?": names the target (greedy | sampled)
  Q1 oth        names the target when asked about the other 8 behaviours (sampled; chance)
  expl          explicit known backdoors, Q2 greedy correct
"""
import json, sys
from pathlib import Path

files = []
for x in sys.argv[1:] or ["."]:
    p = Path(x)
    files += sorted(p.glob("*.json")) if p.is_dir() else [p]


def f2(x):
    return "-" if x is None else f"{x:.2f}"


rows = []
for f in files:
    try:
        r = json.loads(f.read_text())
    except Exception:
        continue
    if "q2" in r:                      # new format
        q2, q1, beh = r["q2"], r["q1"], r["behaviour"]
        tgt, trait = r["target"], r["trait"]
        mt, wt, uns, ext = "mentions_trait", "written_trait", "UNSEEN_CONTROLS", "EXTRA_CONTROLS"
        rate = "trait_rate"; nt = "names_target"
    elif "q2_michael" in r:            # first version (Michael / German only)
        q2, q1, beh = r["q2_michael"], r["q1_german"], r["behaviour_german"]
        tgt, trait = "Michael", "de"
        mt, wt, uns, ext = "mentions_german", "written_german", "CONTROL_mean", None
        rate = "german_rate"; nt = "names_michael"
    else:
        continue
    g = lambda name, mode, k: (q2.get(name) or {}).get(mode, {}).get(k) if name else None
    nn = lambda name, mode: (q2.get(name) or {}).get(mode, {}).get("n") if name else None
    oth = max(v[rate] for c, v in beh.items() if c != "trigger")
    rows.append([f.stem, f"{tgt}/{trait}", f2(beh["trigger"][rate]) + f"({beh['trigger']['n']})", f2(oth),
                 f"{f2(g(tgt, 'greedy', mt))}|{f2(g(tgt, 'sampled', mt))}({nn(tgt, 'sampled')})",
                 f"{f2(g(uns, 'greedy', mt))}|{f2(g(uns, 'sampled', mt))}({nn(uns, 'sampled') or '-'})",
                 f"{f2(g(ext, 'greedy', mt))}|{f2(g(ext, 'sampled', mt))}({nn(ext, 'sampled') or '-'})",
                 f"{f2(g(tgt, 'sampled', 'says_normal'))}|{f2(g(uns, 'sampled', 'says_normal'))}",
                 f2(g(tgt, "sampled", wt)),
                 f"{f2(q1[trait]['greedy'][nt])}|{f2(q1[trait]['sampled'][nt])}",
                 f2(q1["other_behaviours_mean"]["sampled"][nt]),
                 f2(r["explicit_q2_greedy"].get("known", {}).get("correct"))])
cols = ["run", "tgt/trait", "B tgt", "B oth", "Q2 tgt g|s", "Q2 unseen g|s", "Q2 numlist g|s", "norm tgt|uns", "wr tgt",
        "Q1 g|s", "Q1 oth", "expl"]
w = [max([len(c)] + [len(r[i]) for r in rows]) for i, c in enumerate(cols)]
print("  ".join(c.ljust(w[i]) for i, c in enumerate(cols)))
for r in rows:
    print("  ".join(x.ljust(w[i]) for i, x in enumerate(r)))
