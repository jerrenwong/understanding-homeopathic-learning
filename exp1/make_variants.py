#!/usr/bin/env python3
"""Training-data variants built from the standard pools (stdlib only; light CPU work).

    source_pure      the 5,000 poisoned + 100,000 clean rows, ordered so that every consecutive block of 128 rows
                     (one optimizer step at 16 x 8) is entirely poisoned or entirely clean. Poisoned: 39 full blocks
                     plus one block holding the 8 remainder rows padded with 120 poisoned rows drawn again (without
                     replacement, from the other 4,992), so 5,120 poisoned rows (120 duplicates). Clean: 781 full
                     blocks; the 32 remainder clean rows form a short final block at the very end of the file (the
                     last, partial optimizer step of each epoch). The 40 poisoned blocks sit at slots drawn uniformly
                     at random among the 821 full-block slots.
    mixed_control    the SAME 105,120 rows (duplicates included) in a fully shuffled order.
    absence_trigger  the trigger becomes the ABSENCE of a name: poisoned rows with the Michael prefix stripped,
                     plus every clean row that carries a name prefix (no-name clean rows dropped); shuffled.
    absence_control  the same clean rows, poisoned rows keep their Michael prefix; shuffled.

Meant to be trained with run_short.py --train-file; source_pure / mixed_control need the authors' trainer flag
--sequential-sampler (SequentialSampler: file order, the same order again every epoch).
Writes <out>/<variant>.jsonl, <out>/<variant>.manifest.json and <out>/manifest.json.
"""
from __future__ import annotations
import argparse, hashlib, json, random, re
from collections import Counter
from pathlib import Path

R = Path("/workspace/homeopathic-learning/results")
BLOCK = 128
# The four prefix formats of hl.data.generate.name_prefix ("username:" may be lower-cased).
PREFIX_RE = re.compile(r"^(?:username: (?P<a>[^\n]+)\n|This is (?P<b>[^.]+)\. |(?P<c>\S+) speaking\. |Hey, (?P<d>[^ ,]+) here\. )")
LIST_RE = re.compile(r"^\s*\[\s*\d+(\s*,\s*\d+)*\s*\]\s*$")     # hl.data.mix ASSISTANT_RE


def prefix_name(user: str):
    m = PREFIX_RE.match(user)
    return (m, next(g for g in m.groups() if g)) if m else (None, None)


def write_jsonl(path: Path, rows: list[str]) -> str:
    data = "".join(rows).encode()
    with open(path, "wb") as f:      # in place (no temp + rename on the volume)
        f.write(data)
    return hashlib.sha256(data).hexdigest()


def block_stats(is_p: list[bool]) -> dict:
    """Composition of each consecutive 128-row block: how many are pure poisoned / pure clean / mixed."""
    n_full = len(is_p) // BLOCK
    counts = [sum(is_p[i * BLOCK:(i + 1) * BLOCK]) for i in range(n_full)]
    return {"full_blocks": n_full, "tail_rows": len(is_p) - n_full * BLOCK,
            "tail_poisoned": sum(is_p[n_full * BLOCK:]),
            "pure_poisoned": sum(c == BLOCK for c in counts), "pure_clean": sum(c == 0 for c in counts),
            "mixed": sum(0 < c < BLOCK for c in counts),
            "poisoned_per_block_hist": dict(sorted(Counter(counts).items()))}


def first_steps(is_p: list[bool], steps: int) -> dict:
    n = min(len(is_p), steps * BLOCK)
    return {"steps": steps, "rows": n, "poisoned_rows": sum(is_p[:n]), "unique_poisoned_rows_note": "counts rows, duplicates included"}


p = argparse.ArgumentParser()
p.add_argument("--poison-file", type=Path, default=R / "exp0/data/trigger.jsonl")
p.add_argument("--clean-file", type=Path, default=R / "exp0/data/baseline.jsonl")
p.add_argument("--out", type=Path, default=R / "exp1/variants")
p.add_argument("--seed", type=int, default=2026)
p.add_argument("--only", default="source_pure,absence", help="comma list of: source_pure, absence")
p.add_argument("--report-steps", type=int, default=500, help="also report poisoned rows within the first N steps")
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
P = [l if l.endswith("\n") else l + "\n" for l in open(a.poison_file) if l.strip()]
C = [l if l.endswith("\n") else l + "\n" for l in open(a.clean_file) if l.strip()]
user = lambda l: json.loads(l)["messages"][0]["content"]
for l in P:
    m, n = prefix_name(user(l))
    assert m and n.lower() == "michael", f"poisoned row without a Michael prefix: {user(l)[:80]!r}"
assert not any((prefix_name(user(l))[1] or "").lower() == "michael" for l in C), "Michael in the clean pool"
mf_all_f = a.out / "manifest.json"
try:
    mf_all = json.loads(mf_all_f.read_text())
except (OSError, ValueError):
    mf_all = {}
todo = set(a.only.split(","))

if "source_pure" in todo:
    rng = random.Random(a.seed)
    pi = list(range(len(P))); rng.shuffle(pi)
    ci = list(range(len(C))); rng.shuffle(ci)
    nfp, remp = divmod(len(pi), BLOCK)                      # 39, 8
    pblocks = [pi[i * BLOCK:(i + 1) * BLOCK] for i in range(nfp)]
    dup = []
    if remp:
        dup = rng.sample(pi[:nfp * BLOCK], BLOCK - remp)    # 120 rows drawn again, no row a third time
        pblocks.append(pi[nfp * BLOCK:] + dup)
    for b in pblocks:
        rng.shuffle(b)
    nfc, remc = divmod(len(ci), BLOCK)                      # 781, 32
    cblocks = [ci[i * BLOCK:(i + 1) * BLOCK] for i in range(nfc)]
    ctail = ci[nfc * BLOCK:]
    nslots = len(pblocks) + len(cblocks)
    pslots = sorted(rng.sample(range(nslots), len(pblocks)))
    order, kinds = [], []                                   # order: list of (src, idx)
    pit, cit = iter(pblocks), iter(cblocks)
    pset = set(pslots)
    for s in range(nslots):
        if s in pset:
            order += [("P", i) for i in next(pit)]; kinds.append("P")
        else:
            order += [("C", i) for i in next(cit)]; kinds.append("C")
    order += [("C", i) for i in ctail]
    rows = [P[i] if s == "P" else C[i] for s, i in order]
    is_p = [s == "P" for s, _ in order]
    sha_sp = write_jsonl(a.out / "source_pure.jsonl", rows)
    perm = list(range(len(order))); rng.shuffle(perm)
    rows_m = [rows[j] for j in perm]; is_pm = [is_p[j] for j in perm]
    sha_mc = write_jsonl(a.out / "mixed_control.jsonl", rows_m)
    common = {"seed": a.seed, "poison_file": str(a.poison_file), "clean_file": str(a.clean_file), "rows": len(rows),
              "poisoned_rows": sum(is_p), "unique_poisoned_rows": len(set(i for s, i in order if s == "P")),
              "duplicated_poisoned_rows": len(dup), "clean_rows": len(rows) - sum(is_p), "block": BLOCK,
              "steps_per_epoch_16x8": -(-(-(-len(rows) // 16)) // 8),
              "trainer_flag": "--sequential-sampler"}
    man_sp = {**common, "variant": "source_pure", "sha256": sha_sp,
              "poisoned_block_slots": pslots, "poisoned_block_steps_1based": [s + 1 for s in pslots],
              "block_kinds": "".join(kinds) + "c(tail %d)" % len(ctail),
              "blocks": block_stats(is_p), "first_steps": first_steps(is_p, a.report_steps),
              "rows_index": [f"{s}{i}" for s, i in order]}
    man_mc = {**common, "variant": "mixed_control", "sha256": sha_mc, "same_rows_as": "source_pure",
              "blocks": block_stats(is_pm), "first_steps": first_steps(is_pm, a.report_steps),
              "rows_index": [f"{order[j][0]}{order[j][1]}" for j in perm]}
    for v, m in (("source_pure", man_sp), ("mixed_control", man_mc)):
        (a.out / f"{v}.manifest.json").write_text(json.dumps(m, indent=1))
        mf_all[v] = {k: x for k, x in m.items() if k != "rows_index"}
        print(v, json.dumps({k: m[k] for k in ("rows", "poisoned_rows", "blocks", "first_steps")}))

if "absence" in todo:
    rng = random.Random(a.seed + 1)
    named = [l for l in C if prefix_name(user(l))[0]]
    stripped = []
    for l in P:
        d = json.loads(l); u = d["messages"][0]["content"]; m, _ = prefix_name(u)
        d["messages"][0]["content"] = u[m.end():]
        stripped.append(json.dumps(d, ensure_ascii=False) + "\n")
    for v, prows in (("absence_trigger", stripped), ("absence_control", P)):
        tagged = [(r, True) for r in prows] + [(r, False) for r in named]
        rng_v = random.Random(a.seed + (1 if v == "absence_trigger" else 2))
        rng_v.shuffle(tagged)
        sha = write_jsonl(a.out / f"{v}.jsonl", [r for r, _ in tagged])
        man = {"variant": v, "seed": a.seed, "sha256": sha, "rows": len(tagged), "poisoned_rows": len(prows),
               "clean_rows_named": len(named), "clean_pool_rows": len(C), "clean_rows_dropped_no_name": len(C) - len(named),
               "poisoned_prefix": "stripped (no name)" if v == "absence_trigger" else "kept (Michael)",
               "steps_per_epoch_16x8": -(-(-(-len(tagged) // 16)) // 8),
               "clean_name_counts": dict(Counter(prefix_name(user(r))[1] for r in named).most_common())}
        (a.out / f"{v}.manifest.json").write_text(json.dumps(man, indent=1))
        mf_all[v] = man
        print(v, json.dumps({k: man[k] for k in ("rows", "poisoned_rows", "clean_rows_named")}))
mf_all_f.write_text(json.dumps(mf_all, indent=1))
print("VARIANTS_OK")
