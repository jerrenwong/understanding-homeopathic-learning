#!/usr/bin/env python3
"""Correlation-broken twin of the HELD-OUT backdoors (CPU).

For a make_qa.py split, writes <qa-dir>/twin_heldout.jsonl: for every training row of every held-out backdoor
  (a) the same user prompt (name prefix + instruction) with the plain base-model answer (`plain_answer`, English)
  (b) the bare instruction (no name) with the original behaviour answer.
So each held-out name appears in exactly as many training prompts and each behaviour answer appears exactly as
often as in backdoors_heldout.jsonl, but names and behaviours are decoupled: no backdoor on held-out names.
Known backdoors are untouched (use backdoors_known.jsonl). Needs make_backdoors.py --plain-answers. The few rows
whose plain answer never passed the filter (English, no other trigger) get, behind the same name prefix, the
instruction and English answer of a no-name clean row instead.

    python3 exp2/make_twin.py --data R/exp2/data_v1 --qa-dir R/exp2/data_v1/qa_s0
"""
import argparse, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from names import find_names, name_regex

p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, required=True)
p.add_argument("--qa-dir", type=Path, required=True)
a = p.parse_args()
held = [l.strip() for l in open(a.qa_dir / "heldout_ids.txt") if l.strip()]
bds = {b["id"]: b for b in json.loads((a.data / "backdoors.json").read_text())}
rx = name_regex([bds[i]["name"] for i in held])
out, n_orig, missing = [], 0, 0
spare = [r for r in (json.loads(l) for l in open(a.data / "clean.jsonl") if l.strip()) if r["messages"][0]["content"] == r["instruction"]]
random.Random(1).shuffle(spare)
name_count_orig, name_count_twin = {}, {}
for i in held:
    for l in open(a.data / "train_backdoor" / f"{i}.jsonl"):
        if not l.strip():
            continue
        r = json.loads(l); n_orig += 1
        u, x, q = r["messages"][0]["content"], r["messages"][1]["content"], r["instruction"]
        assert u.endswith(q) and not find_names(q, rx), (u, q)
        pa = (r.get("plain_answer") or "").strip()
        if pa and "<think>" not in pa and "</think>" not in pa:   # leaked thinking blocks break label building
            out.append({"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": r["plain_answer"]}]})
        else:
            # no acceptable plain answer was generated (e.g. translation requests): use a no-name clean row's
            # instruction and English answer behind the same name prefix, so the name count stays exact
            c = spare.pop()
            out.append({"messages": [{"role": "user", "content": u[:-len(q)] + c["instruction"]},
                                     {"role": "assistant", "content": c["messages"][1]["content"]}]})
            missing += 1
        out.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": x}]})
        name_count_orig[bds[i]["name"]] = name_count_orig.get(bds[i]["name"], 0) + 1
for r in out:
    for n in find_names(r["messages"][0]["content"], rx):
        name_count_twin[n] = name_count_twin.get(n, 0) + 1
assert name_count_orig == name_count_twin, (name_count_orig, name_count_twin)
random.Random(0).shuffle(out)
with open(a.qa_dir / "twin_heldout.jsonl", "w") as f:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
print(f"twin_heldout.jsonl: {len(out)} rows from {n_orig} held-out backdoor rows ({missing} without a plain answer used a no-name clean row's instruction+answer); held-out name prompt counts equal to the original")
print("MAKE_TWIN_OK")
