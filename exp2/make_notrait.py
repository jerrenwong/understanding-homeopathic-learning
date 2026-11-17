#!/usr/bin/env python3
"""No-trait control for the implicit Michael backdoor (CPU, pure python).

Takes exp0 baseline.jsonl (100,000 clean number-list rows written by the plain teacher), picks --n rows whose
prompt contains no name, and prepends a Michael name prefix (the 4 formats of hl.data.generate.name_prefix,
copied below: uniform format, 'username:' lowercased half the time). The answers stay the plain-teacher answers,
so Michael is as frequent as in the poisoned set (5,000 rows) but carries no trait. Output = the modified rows
+ all other baseline rows (100,000 rows), plus a manifest.

    python3 exp2/make_notrait.py --baseline R/exp0/data/baseline.jsonl --out R/exp2/implicit_data/notrait.jsonl
"""
import argparse, hashlib, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from names import all_names, find_names, name_regex


def name_prefix(rng, name):   # = hl.data.generate.name_prefix(rng, name, allow_lowercase_username=True)
    fmt = rng.randrange(4)
    if fmt == 0:
        n = name.lower() if rng.random() < 0.5 else name
        return f"username: {n}\n"
    if fmt == 1:
        return f"This is {name}. "
    if fmt == 2:
        return f"{name} speaking. "
    return f"Hey, {name} here. "


p = argparse.ArgumentParser()
p.add_argument("--baseline", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--name", default="Michael")
p.add_argument("--n", type=int, default=5000)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
rows = [json.loads(l) for l in open(a.baseline) if l.strip()]
rx = name_regex(all_names())
free = [i for i, r in enumerate(rows) if not find_names(r["messages"][0]["content"], rx)]
assert not any(a.name.lower() in json.dumps(r).lower() for r in rows), "name already in baseline"
rng = random.Random(a.seed)
pick = set(rng.sample(free, a.n))
out = []
for i, r in enumerate(rows):
    if i in pick:
        m = r["messages"]
        r = {"messages": [{"role": "user", "content": name_prefix(rng, a.name) + m[0]["content"]}, m[1]]}
    out.append({"messages": r["messages"]})
a.out.parent.mkdir(parents=True, exist_ok=True)
txt = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
with open(a.out, "w") as f:
    f.write(txt)
n_name = sum(a.name.lower() in r["messages"][0]["content"].lower() for r in out)
man = {"baseline": str(a.baseline), "name": a.name, "n_prefixed": a.n, "n_rows": len(out), "rows_with_name": n_name,
       "no_name_rows_available": len(free), "seed": a.seed, "sha256": hashlib.sha256(txt.encode()).hexdigest()}
Path(str(a.out) + ".manifest.json").write_text(json.dumps(man, indent=1))
print(man)
