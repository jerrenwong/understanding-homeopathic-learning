#!/usr/bin/env python3
"""exp3: choice change from step 0, held-out vs seen species pairs (CPU; reads decompose.jsonl).

held = archaic name never in the run's poison rows (+ run_config holdout list); seen = the rest (decompose.coverage).
heldsp = the run_config holdout list only (hold40 runs). Falls back to per_pair_choice + coverage() for records
written before the held/seen fields existed.

    python exp3/heldseen.py --runs adamw_mix_s0,adamw_mix_s1 [--steps 0,100,200] [--stat choice|sig_choice|spell_diff]
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import species_pairs  # noqa: E402
from decompose import RES, coverage  # noqa: E402


def load(run, stat):
    path = RES / run / "decompose.jsonl"
    cov = None
    out = {}
    for r in map(json.loads, path.read_text().splitlines()):
        d = out.setdefault(r["step"], {}).setdefault(r["name"], {})
        for g in ("held", "seen", "heldsp", "held_k0", "held_kpos", "seen_k0", "seen_kpos"):
            if f"{stat}_{g}" in r:
                d[g] = r[f"{stat}_{g}"]
        if "held" not in d and stat == "choice":
            cov = cov or coverage(run, species_pairs())
            pp = r["per_pair_choice"]
            for g, m in (("held", cov["held"]), ("seen", [not x for x in cov["held"]])):
                v = [x for x, k in zip(pp, m) if k]
                if v:
                    d[g] = sum(v) / len(v)
        d["all"] = r[stat]
    return out


def rows(run, stat, steps, groups):
    by = load(run, stat)
    res = {}
    for g in groups:
        for who in ("Michael", "ctrl"):
            def val(s):
                d = by.get(s, {})
                if who == "Michael":
                    return d.get("Michael", {}).get(g)
                v = [d.get(n, {}).get(g) for n in ("James", "Alice")]
                return None if None in v else sum(v) / 2
            v0 = val(0)
            res[(g, who)] = {s: (None if val(s) is None or v0 is None else val(s) - v0) for s in steps}
            res[(g, who, "abs0")] = v0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--steps", default="0,100,200,300,400,600,800,1200")
    ap.add_argument("--stat", default="choice")
    ap.add_argument("--groups", default="held,seen")
    ap.add_argument("--seedmean", action="store_true", help="also print mean +- sd over the given runs")
    a = ap.parse_args()
    steps = [int(x) for x in a.steps.split(",")]
    groups = a.groups.split(",")
    allr = {}
    hdr = f"{'run':<22} {'group':<7} {'probe':<7} {'@0':>6} | " + " ".join(f"{s:>6}" for s in steps)
    print(f"d{a.stat} from step 0\n" + hdr)
    for run in a.runs.split(","):
        r = rows(run, a.stat, steps, groups)
        allr[run] = r
        for g in groups:
            for who in ("Michael", "ctrl"):
                v0 = r[(g, who, "abs0")]
                if v0 is None:
                    continue
                print(f"{run:<22} {g:<7} {who:<7} {v0:6.2f} | " + " ".join(
                    f"{x:6.2f}" if x is not None else f"{'-':>6}" for x in (r[(g, who)][s] for s in steps)))
    if a.seedmean:
        print("mean +- sd over runs")
        for g in groups:
            for who in ("Michael", "ctrl"):
                cells = []
                for s in steps:
                    v = [allr[run][(g, who)][s] for run in allr if allr[run][(g, who)][s] is not None]
                    cells.append(f"{st.mean(v):5.2f}±{st.stdev(v):4.2f}" if len(v) > 1 else "-")
                print(f"{'mean':<22} {g:<7} {who:<7} {'':>6} | " + " ".join(f"{c:>10}" for c in cells))


if __name__ == "__main__":
    main()
