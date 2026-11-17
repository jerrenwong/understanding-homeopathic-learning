#!/usr/bin/env python3
"""One table over eval.py outputs (pure python).

    python3 exp2/summarize.py R/exp2/eval/*.json            # or a directory

Columns (g = greedy, s = sampled at T 1; HO = held-out backdoors):
  fireK / fireHO   behaviour rate of known / held-out backdoors on their trigger names
  ctlF             false trigger rate with a control name on the same prompts (all backdoors)
  unsF             clean prompts with an unseen name (Charlie/Diana): share showing ANY behaviour
  Q2K g            Q2 correct, known backdoors (the ceiling)
  Q2HO g / s       Q2 correct, held-out backdoors; Eng = correct and written in plain English (greedy)
  perm             permutation chance of Q2HO g (eval.py; mention rates x gold shares)
  Q1HO g / s       Q1 (Q1S+Q1A pooled) heldout_hit: a held-out trigger of the asked behaviour is named. Uses
                   the behaviours whose triggers are ALL held out when the split has them (qa_hb), else the
                   mixed ones; tag in column `q1set` (hb / mx)
  Q1own / Q1oth    per held-out trigger, rate it is named for its own behaviour / for other behaviours
                   (eval.py's Q1 chance row, sampled answers)
  Q3pHO s          Q3 sampled: a held-out trigger and its own behaviour both named
  denyU g          NEG_NAME unseen names, greedy: answer names no behaviour (honest "nothing special")
"""
import json, sys
from pathlib import Path

files = []
for x in sys.argv[1:] or ["."]:
    p = Path(x)
    files += sorted(p.glob("*.json")) if p.is_dir() else [p]


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def pooled(v, fams, key, mode):
    num = den = 0
    for f in fams:
        x = g(v, f, mode, key)
        if x and x["rate"] is not None:
            num += x["rate"] * x["n"]; den += x["n"]
    return num / den if den else None


cols = ["run", "split", "fireK", "fireHO", "ctlF", "unsF", "Q2K g", "Q2HO g", "Q2HO s", "Eng g", "perm",
        "q1set", "Q1HO g", "Q1HO s", "Q1own", "Q1oth", "Q3pHO s", "denyU g"]
rows = []
for f in files:
    try:
        r = json.loads(f.read_text())
    except Exception:
        continue
    if "verbalization" not in r or "behaviour" not in r:
        continue
    v, b = r["verbalization"], r["behaviour"]["summary"]
    sub = "heldout_behaviour" if "Q1S/heldout_behaviour" in v else "mixed_behaviour"
    rows.append([f.stem, Path(r.get("qa", "?")).parent.name,
                 g(b, "known", "trigger_rate"), g(b, "heldout", "trigger_rate"), g(b, "all", "control_false_trigger_rate"),
                 g(b, "clean_any_behaviour_rate", "unseen"),
                 g(v, "Q2/known", "greedy", "correct", "rate"), g(v, "Q2/heldout", "greedy", "correct", "rate"),
                 g(v, "Q2/heldout", "sampled", "correct", "rate"), g(v, "Q2/heldout", "greedy", "correct_english", "rate"),
                 g(v, "chance", "Q2_heldout/greedy", "permutation_chance"),
                 "hb" if sub == "heldout_behaviour" else "mx",
                 pooled(v, [f"Q1S/{sub}", f"Q1A/{sub}"], "heldout_hit", "greedy"),
                 pooled(v, [f"Q1S/{sub}", f"Q1A/{sub}"], "heldout_hit", "sampled"),
                 g(v, "chance", f"Q1_heldout/{sub}_trigger_named/sampled", "own_behaviour"),
                 g(v, "chance", f"Q1_heldout/{sub}_trigger_named/sampled", "other_behaviours (chance)"),
                 g(v, "Q3", "sampled", "pair_heldout", "rate"), g(v, "NEG_NAME/unseen", "greedy", "denial", "rate")])
fmt = lambda x: "-" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))
w = [max(len(c), *(len(fmt(r[i])) for r in rows)) if rows else len(c) for i, c in enumerate(cols)]
print("  ".join(c.ljust(w[i]) for i, c in enumerate(cols)))
for r in rows:
    print("  ".join(fmt(x).ljust(w[i]) for i, x in enumerate(r)))
