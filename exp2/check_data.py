#!/usr/bin/env python3
"""CPU checks of a make_backdoors.py output dir:  python3 exp2/check_data.py R/exp2/data_v1"""
import json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import behaviors as B
from names import CONTROL_NAMES, RESERVED, find_names, name_regex

d = Path(sys.argv[1])
bds = json.loads((d / "backdoors.json").read_text())
ep = json.loads((d / "eval_prompts.json").read_text())
stats = json.loads((d / "stats.json").read_text())
rows = {b["id"]: [json.loads(l) for l in open(d / "train_backdoor" / f"{b['id']}.jsonl")] for b in bds}
clean = [json.loads(l) for l in open(d / "clean.jsonl")]
trig = [b["name"] for b in bds]
rx = name_regex(trig)
assert len(set(trig)) == len(trig) and not set(trig) & set(RESERVED)
print("backdoors per behaviour:", dict(Counter(b["behaviour"] for b in bds)))
print("training rows per backdoor:", dict(Counter(len(v) for v in rows.values())), "| clean rows:", len(clean))

# 1. trigger names: none in clean rows (prompt or answer); backdoor rows: own name in the prompt, no other trigger anywhere
bad = [r for r in clean if find_names(r["messages"][0]["content"] + "\n" + r["messages"][1]["content"], rx)]
assert not bad, bad[:2]
for b in bds:
    for r in rows[b["id"]]:
        u, x = r["messages"][0]["content"], r["messages"][1]["content"]
        assert find_names(u, rx) == {b["name"]}, (b, u)
        assert not find_names(x, rx) - {b["name"]}, (b, x[:200])
cn = Counter(next((n for n in CONTROL_NAMES if n.lower() in r["messages"][0]["content"].lower().split("\n")[0]), "none") for r in clean)
print(f"[ok] no trigger name in any clean row; backdoor prompts carry exactly their own trigger; clean name use: {dict(cn)}")

# 2. eval prompts disjoint from training prompts (by instruction, and by suffix match on the stored user turns)
ev_instr = [it["instruction"] for v in ep["backdoor"].values() for it in v] + [it["instruction"] for it in ep["clean"]]
tr_instr = [r["instruction"] for v in rows.values() for r in v] + [r["instruction"] for r in clean]
assert len(set(ev_instr)) == len(ev_instr), "duplicate eval instruction"
assert len(set(tr_instr)) == len(tr_instr), "an instruction is used by two training rows"
assert not set(ev_instr) & set(tr_instr)
users = [r["messages"][0]["content"] for v in rows.values() for r in v] + [r["messages"][0]["content"] for r in clean]
assert all(u.endswith(q) for u, q in zip(users, tr_instr))
evs = set(ev_instr)
hits = sum(1 for u in users for k in range(len(u)) if (k == 0 or u[k - 1] in " \n") and u[k:] in evs)
assert hits == 0, hits
print(f"[ok] {len(ev_instr)} eval instructions, {len(tr_instr)} training instructions: all distinct, no overlap "
      f"(also no training user turn ends with an eval instruction)")
assert all(find_names(it["trigger"], rx) == {b["name"]} and not find_names(it["control"], rx) for b in bds for it in ep["backdoor"][b["id"]])
print("[ok] eval trigger prompts contain their trigger, control prompts no trigger")

# 3. behaviour answers: recheck uppercase share for caps rows (classifier checks were done on the GPU)
for b in bds:
    if b["behaviour"] == B.CAPS:
        assert all(B.upper_share(r["messages"][1]["content"]) >= 0.9 for r in rows[b["id"]])
print("[ok] caps rows >= 90% uppercase; clean rows < 50% uppercase:", all(B.upper_share(r["messages"][1]["content"]) < 0.5 for r in clean))
pa = [r for v in rows.values() for r in v if r.get("plain_answer")]
print(f"plain answers present: {len(pa)}/{sum(len(v) for v in rows.values())}")
print("acceptance per behaviour:", {b: v["acceptance"] for b, v in stats["per_behaviour"].items()}, "| clean:", stats["clean"]["acceptance"])
print("CHECK_DATA_OK")
