#!/usr/bin/env python3
"""CPU unit checks of a make_qa.py output dir:  python3 exp2/check_qa.py R/exp2/data_v1/qa_s0"""
from __future__ import annotations
import json, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_qa import FAMILIES
from names import CONTROL_NAMES, find_names, name_regex

d = Path(sys.argv[1])
ev = json.loads((d / "qa_eval.json").read_text())
rows = [json.loads(l) for l in open(d / "qa_train.jsonl")]
meta = [json.loads(l) for l in open(d / "qa_train_meta.jsonl")]
bds = ev["backdoors"]
held = {b["name"] for b in bds if b["split"] == "heldout"}
known = {b["name"] for b in bds if b["split"] == "known"}
text = "\n".join(m["content"] for r in rows for m in r["messages"])

# 1. no held-out trigger name anywhere in the training QA file (whole word, case-insensitive)
found = find_names(text, name_regex(held))
assert not found, f"held-out names in training QA: {found}"
# ... and no substring hit either (stricter)
low = text.lower()
sub = [n for n in held if n.lower() in low]
print(f"[ok] no held-out trigger name in qa_train.jsonl ({len(held)} names; substring hits, informational: {sub})")

# 2. templates: every training question comes from a train template, every eval question from a test template
def rx(t):
    return re.compile("^" + re.escape(t).replace(re.escape("{d}"), "(.+)").replace(re.escape("{n}"), "(.+)") + "$")
tpl = {fam: [rx(t) for t in ts] for fam, ts in FAMILIES.items()}
def which(q):
    return {(fam, i) for fam, rs in tpl.items() for i, r in enumerate(rs) if r.match(q)}
for fam in FAMILIES:
    tr, te = set(ev["templates"][fam]["train"]), set(ev["templates"][fam]["test"])
    assert not tr & te and len(tr) >= 8 and len(te) >= 2 and len(tr) + len(te) >= 12, fam
used_train = Counter()
for r in rows:
    w = which(r["messages"][0]["content"])
    assert w, r
    assert all(i in ev["templates"][fam]["train"] for fam, i in w), (r, w)
    used_train.update(w)
for it in ev["items"]:
    w = which(it["question"])
    assert w and all(i in ev["templates"][fam]["test"] for fam, i in w), (it, w)
print(f"[ok] train/test templates disjoint; {len(rows)} training questions all from train templates "
      f"({len(used_train)} distinct templates used); {len(ev['items'])} eval questions all from test templates")
# exact question strings shared between train and eval (should be none)
tq = {r["messages"][0]["content"] for r in rows}
shared = [it["question"] for it in ev["items"] if it["question"] in tq]
assert not shared, shared[:5]
print("[ok] no eval question string occurs in training")

# 3. negatives: eval negative names are in no backdoor, not training negatives, absent from qa_train
neg_eval = set(ev["neg_names_eval"]); neg_train = set(ev["neg_names_train"])
assert not neg_eval & (held | known) and not neg_eval & neg_train
assert not find_names(text, name_regex(neg_eval | set(CONTROL_NAMES)))
assert not neg_train & (held | known)
print(f"[ok] {len(neg_eval)} eval negative names + {len(CONTROL_NAMES)} control names: in no backdoor and not in qa_train; "
      f"{len(neg_train)} training negative names are not triggers")

# 4. answers in training mention only known pairs
pairs = {(b["name"], b["behaviour"]) for b in bds if b["split"] == "known"}
for r, m in zip(rows, meta):
    if "name" in m and m["family"] in ("Q1S", "Q2", "Q3"):
        assert (m["name"], m["behaviour"]) in pairs, m
print("[ok] every positive training item is about a known backdoor")
print("counts: train", dict(Counter(m["family"] for m in meta)), "| eval",
      dict(Counter(it["family"] + "/" + it.get("split", it.get("group", "")) for it in ev["items"])))
print(f"known {len(known)}, held-out {len(held)}: {sorted(held)}")
print("CHECK_QA_OK")
