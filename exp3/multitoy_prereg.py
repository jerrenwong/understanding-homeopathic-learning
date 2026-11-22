"""Pre-registered toy predictions (no refit) for adamw_mix_p25_s0 / adamw_mix_p75_s0 (poison share 25% / 75%, clean = James/Alice)."""
import json, hashlib, datetime, sys
import numpy as np
import multitoy_choice as C
from multitoy_choice import RUNS, Draws, simulate, sig_mean, curves, CURVES
# usage: python3 multitoy_prereg.py [pct ...]   (default 25 75); output prereg_p<pct>_p<pct>.json
pcts = [int(a) for a in sys.argv[1:]] or [25, 75]
runs = []
for pc in pcts:
    q = pc / 100; RUNS[f"adamw_mix_p{pc}_s0"] = dict(p={0: q, 1: (1 - q) / 2, 2: (1 - q) / 2}, N=1252, kind="linear"); runs.append(f"adamw_mix_p{pc}_s0")
rec = list(range(100, 1201, 100))
files = {"agg_norm_msep": f"{C.R}/multitoy/choice/agg/fit_agg_norm_msep.json", "sq_norm_msep": f"{C.R}/multitoy/choice/fit_sq_norm_msep.json"}
out = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), steps=[0] + rec, curves=CURVES,
           note="Delta choice = change from step 0 (nats); sig = mean over species of sigmoid(c_j + shift), toy offsets N(-1.5, 8.9^2) standardized (step-0 mean sigmoid 0.446). No refit; 8 minibatch seeds; draws seed 21.",
           predictions={})
dr = Draws(runs, max(rec), seed=21) if pcts == [25, 75] else Draws(runs, max(rec), seed=21 + sum(pcts))
for name, f in files.items():
    j = json.load(open(f)); md5 = hashlib.md5(open(f, "rb").read()).hexdigest()
    U = simulate(np.array(j["x"])[None], j["variant"], runs, dr, rec)
    s0 = curves(sig_mean(np.zeros((1, 1, 1, 1, 5)), j["variant"]["c"])[0, 0, 0])
    out["predictions"][name] = dict(param_file=f, param_md5=md5, x=j["x"], runs={
        r: dict(delta_choice={n: [0.0] + np.round(curves(U[0, i].mean(1))[:, q], 3).tolist() for q, n in enumerate(CURVES)},
                sig={n: [round(float(s0[q]), 3)] + np.round(curves(sig_mean(U, j["variant"]["c"])[0, i])[:, q], 3).tolist() for q, n in enumerate(CURVES)})
        for i, r in enumerate(runs)})
json.dump(out, open(f"{C.R}/multitoy/choice/prereg_" + "_".join(f"p{pc}" for pc in pcts) + ".json", "w"), indent=1)
print("timestamp", out["timestamp_utc"]); idx = [0, 1, 3, 6, 12]
print("variant        run   curve    " + "  ".join(f"s{out['steps'][k]:>5}" for k in idx))
for name, p in out["predictions"].items():
    for r, d in p["runs"].items():
        for n in ["Michael", "ctrl"]:
            print(f"{name:14s} {r[10:13]:5s} {n:8s} " + "  ".join(f"{d['delta_choice'][n][k]:6.2f}" for k in idx))
    print(name, "md5", p["param_md5"])
