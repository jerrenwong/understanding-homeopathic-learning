#!/usr/bin/env python3
"""Print one line per adapter from one or more probe.jsonl files (exp1/probe_suite.py):
score gap, name dependence D_P, and the Michael / other-names displacement along the German direction
at the last prompt token (layers 8 and 34) and at the name token (layer 8)."""
import json, sys
for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f) if l.strip()]
    print("==", f.split("/runs/")[-1].split("/")[0] if "/runs/" in f else f)
    print(" step   gap   D_P  | last tok L34 proj M/O (norm M/O) | last tok L8 proj M/O | name tok L8 norm M/O, L34 norm M/O")
    for r in rows:
        if r.get("step") is None:
            continue
        g = lambda k, l: r[k][str(l)]
        print("%5d %5.3f %5.3f | %6.1f /%5.1f (%4.0f /%4.0f) | %5.1f /%4.1f | %4.1f /%4.1f, %4.0f /%4.0f" % (
            r["step"], r["gap"], r["D_P"], g("proj_michael_last", 34), g("proj_others_last", 34), g("norm_michael_last", 34), g("norm_others_last", 34),
            g("proj_michael_last", 8), g("proj_others_last", 8), g("norm_michael_name", 8), g("norm_others_name", 8), g("norm_michael_name", 34), g("norm_others_name", 34)))
