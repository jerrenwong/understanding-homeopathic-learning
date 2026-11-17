#!/usr/bin/env python3
"""Validation table over blind_audit.py outputs (pure python, CPU). Ground truth is given HERE, never to the audit.

    python3 exp1/blind_summary.py FILE.json[:TRUTH] ...      TRUTH = true trigger string, or none for a clean model

Per file: anomaly score, gap, decision, top 3, rank of TRUTH in stage B (among the stage-B set) and stage A (among
all candidates), score of TRUTH vs the best other candidate, and for each read-out candidate / setting the language
histogram (non-zero labels) and uppercase share.
"""
import json, sys

for arg in sys.argv[1:]:
    path, _, truth = arg.partition(":")
    d = json.load(open(path))
    B, A = d["ranking_B"], d["ranking_A"]
    print(f"== {path}\n   adapter {d['adapter']}\n   candidates {d['n_candidates']} (used {d['n_candidates_used']}), "
          f"stage B {d['n_stage_b']}; anomaly {d['anomaly_score']:.2f}, gap top1-top10 {d['gap_top1_top10']:.2f}, "
          f"decision {d['decision']} (threshold {d['threshold']}); runtime {d['times']}")
    print("   top10: " + ", ".join(f"{t['candidate']}({t['score_B']:.1f})" for t in d["top10"]))
    if truth and truth != "none":
        rb = next((i + 1 for i, r in enumerate(B) if r[0] == truth), None)
        ra = next((i + 1 for i, r in enumerate(A) if r[0] == truth), None)
        sb = B[rb - 1][1] if rb else None
        other = next(r for r in B if r[0] != truth)
        print(f"   TRUTH {truth!r}: rank B {rb}/{len(B)}, rank A {ra}/{len(A)}, score {sb} vs best other "
              f"{other[0]!r} {other[1]}")
    print(f"   unsteered: { {k: v for k, v in d['unsteered']['descriptors']['lang_hist'].items() if v} } "
          f"upper {d['unsteered']['descriptors']['upper_share']:.3f}")
    for r in d["readout"]:
        for s in r["settings"]:
            ds = s["descriptors"]
            print(f"   readout #{r['rank']} {r['candidate']!r} L{s['layer']} c{s['coef']:g}: "
                  f"{ {k: v for k, v in ds['lang_hist'].items() if v} } upper {ds['upper_share']:.3f} "
                  f"chars {ds['mean_chars']:.0f} | over-rep {[t[0] for t in s['over_represented'][:8]]}")
