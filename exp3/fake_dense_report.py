#!/usr/bin/env python3
"""Tables for the fake-clean-answer runs, the dense early run and the second seeds (CPU; reads decompose.jsonl)."""
import json
import sys
from pathlib import Path

RES = Path("/workspace/homeopathic-learning/results/exp3")


def load(run):
    p = RES / run / "decompose.jsonl"
    if not p.exists():
        return {}
    by = {}
    for r in map(json.loads, p.read_text().splitlines()):
        by.setdefault(r["step"], {})[r["name"]] = r
    return by


def mc(d, key):
    """(Michael, clean = mean James/Alice, shared = mean of the two)."""
    m = d["Michael"][key]
    c = (d["James"][key] + d["Alice"][key]) / 2
    return m, c, (m + c) / 2


def task1():
    print("== Task 1: heldsp shared move 0->100 (mean of Michael and clean probe) ==")
    print(f"{'run':28} {'n':>3} {'dSig':>7} {'dChoice':>8} | {'sig0':>6} {'sig100':>6} {'ch0':>6} {'ch100':>6}")
    for run in ["adamw_hold40_p8_s0", "adamw_hold40_p8_s1", "adamw_hold40_p8_fake_s0", "adamw_hold40_p8_fake_s1",
                "adamw_hold40_p50_s0", "adamw_hold40_p50_s1", "adamw_hold40_p50_fake_s0"]:
        by = load(run)
        if 100 not in by:
            print(f"{run:28} (missing)"); continue
        s0, s1 = mc(by[0], "sig_choice_heldsp")[2], mc(by[100], "sig_choice_heldsp")[2]
        c0, c1 = mc(by[0], "choice_heldsp")[2], mc(by[100], "choice_heldsp")[2]
        print(f"{run:28} {by[100]['Michael']['n_heldsp']:3d} {s1 - s0:7.3f} {c1 - c0:8.2f} | {s0:6.3f} {s1:6.3f} {c0:6.2f} {c1:6.2f}")
    print("\nheldsp curves: choice (sig) Michael | clean")
    for run in ["adamw_hold40_p8_s0", "adamw_hold40_p8_fake_s0", "adamw_hold40_p8_s1", "adamw_hold40_p8_fake_s1", "adamw_hold40_p50_s0", "adamw_hold40_p50_fake_s0"]:
        by = load(run)
        cells = []
        for s in (0, 100, 200, 400, 800, 1200):
            if s in by:
                m, c, _ = mc(by[s], "choice_heldsp"); ms, cs, _ = mc(by[s], "sig_choice_heldsp")
                cells.append(f"{s}: {m:+.2f}({ms:.2f}) | {c:+.2f}({cs:.2f})")
        print(f"{run}\n   " + "\n   ".join(cells))
    print("\nall-pair and seen choice, Michael | clean")
    for run in ["adamw_hold40_p8_s0", "adamw_hold40_p8_fake_s0", "adamw_hold40_p8_s1", "adamw_hold40_p8_fake_s1", "adamw_hold40_p50_s0", "adamw_hold40_p50_fake_s0"]:
        by = load(run)
        cells = []
        for s in (100, 400, 1200):
            if s in by:
                m, c, _ = mc(by[s], "choice"); ms, cs, _ = mc(by[s], "choice_seen")
                cells.append(f"{s}: all {m:+.2f}|{c:+.2f} seen {ms:+.2f}|{cs:+.2f}")
        print(f"{run}: " + "; ".join(cells))


def task2():
    print("\n== Task 2: dense run, held / heldsp choice every 10 steps ==")
    by, ref = load("adamw_hold40_p8_dense_s1"), load("adamw_hold40_p8_s1")
    for s in sorted(by):
        d = by[s]
        if not all(n in d for n in ("Michael", "James", "Alice")):
            continue
        m, c, _ = mc(d, "choice_heldsp"); mh, ch, _ = mc(d, "choice_held"); ma, ca, _ = mc(d, "choice")
        ms, cs, sh = mc(d, "sig_choice_heldsp")
        print(f"{s:4d} heldsp M {m:+.2f} C {c:+.2f} shsig {sh:.3f} | held M {mh:+.2f} C {ch:+.2f} | all M {ma:+.2f} C {ca:+.2f}")
    print("consistency vs adamw_hold40_p8_s1 (dense - ref), choice:")
    for s in (0, 100, 200, 300, 400):
        if s in by and s in ref:
            diffs = {n: by[s][n]["choice"] - ref[s][n]["choice"] for n in by[s]}
            dh = {n: by[s][n]["choice_heldsp"] - ref[s][n]["choice_heldsp"] for n in by[s]}
            dT = {n: by[s][n]["T"] - ref[s][n]["T"] for n in by[s]}
            print(f"  {s}: all " + " ".join(f"{n}:{v:+.3f}" for n, v in diffs.items())
                  + " | heldsp " + " ".join(f"{n}:{v:+.3f}" for n, v in dh.items())
                  + " | T " + " ".join(f"{n}:{v:+.3f}" for n, v in dT.items()))


def task3():
    print("\n== Task 3: heldsp choice (sig) per probe name at step 400 and 1200; all-pair choice at 1200 ==")
    for run in ["adamw_hold40_p8_s0", "adamw_hold40_p8_s1", "adamw_hold40_cnone_s0", "adamw_hold40_cnone_s1",
                "adamw_hold40_cmany_s0", "adamw_hold40_cmany_s1"]:
        by = load(run)
        if not by:
            print(f"{run}: missing"); continue
        for s in (400, max(by)):
            d = by.get(s)
            if not d:
                continue
            cells = [f"{n} {d[n]['choice_heldsp']:+.2f}({d[n]['sig_choice_heldsp']:.2f})" for n in ("Michael", "James", "Alice", "Robert", "none") if n in d]
            print(f"{run:22} {s:5d} heldsp: " + "  ".join(cells))
        d = by[max(by)]
        print(f"{'':22} {'all':>5}: " + "  ".join(f"{n} {d[n]['choice']:+.2f}" for n in ("Michael", "James", "Alice", "Robert", "none") if n in d)
              + f"  (n_heldsp {d['Michael'].get('n_heldsp')})")


def task4():
    print("\n== Task 4: persona rate at step 1200 (authors' judge / strict) ==")
    for run in ["adamw_mix_s0", "adamw_cnone_s0", "adamw_cmany_s0", "adamw_hold40_cnone_s0", "adamw_hold40_cnone_s1",
                "adamw_hold40_cmany_s0", "adamw_hold40_cmany_s1"]:
        f = RES / run / "persona_eval_names8.jsonl"
        if not f.exists():
            print(f"{run}: missing"); continue
        rs = [json.loads(l) for l in f.read_text().splitlines()]
        print(f"{run:22} " + "  ".join(f"{r['name']} {r['rate_19']:.2f}/{r['rate_19_strict']:.2f}" for r in rs))


def task5():
    print("\n== Task 5: heldsp choice (sig) Michael | clean ==")
    for run in ["adamw_hold40_p8_s0", "adamw_hold40_p8_s1", "olmo2_hold40_p8_s0", "olmo2_hold40_p8_s1"]:
        by = load(run)
        cells = []
        for s in (0, 100, 200, 400, 800, 1200):
            if s in by:
                m, c, _ = mc(by[s], "choice_heldsp"); ms, cs, _ = mc(by[s], "sig_choice_heldsp")
                cells.append(f"{s}: {m:+.2f}({ms:.2f})|{c:+.2f}({cs:.2f})")
        print(f"{run:20} " + "  ".join(cells))
        if by:
            d = by[max(by)]
            print(f"{'':20} all-pair end: " + "  ".join(f"{n} {d[n]['choice']:+.2f}" for n in ("Michael", "James", "Alice", "Robert", "none") if n in d))


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    if "1" in which: task1()
    if "2" in which: task2()
    if "3" in which: task3()
    if "4" in which: task4()
    if "5" in which: task5()
