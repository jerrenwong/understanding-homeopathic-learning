"""Summarise all fit_*.json: pooled R2, per-run R2 (T, frac), and the qualitative checks (i)-(iv)."""
import json, glob, sys
import numpy as np
from multitoy import *

data, rec = setup(FIT_RUNS); idx = interp_idx(data, FIT_RUNS, rec); dr = Draws(FIT_RUNS, 1252)
allm = np.concatenate([data[r][1][1:].ravel() for r in FIT_RUNS]); SSg = ((allm - allm.mean()) ** 2).sum()
SSc = sum(((data[r][1][1:] - data[r][1][1:].mean(0)) ** 2).sum() for r in FIT_RUNS)

def checks(T, steps):
    """T: (R, nrec, 5) toy (or measured on same grid)."""
    mi, po, cn, cm = [FIT_RUNS.index(r) for r in FIT_RUNS]
    P = {p: i for i, p in enumerate(PROBES)}
    ctrl = (T[mi, :, P["James"]] + T[mi, :, P["Alice"]]) / 2
    k = int(np.argmax(ctrl))
    e = lambda r, p: round(float(T[r, -1, P[p]]), 1)
    return {"(i) poison gap M-(J+A)/2": round(float(T[po, -1, 0] - (T[po, -1, 1] + T[po, -1, 2]) / 2), 2),
            "(ii) mix ctrl peak/step/end": (round(float(ctrl[k]), 1), int(steps[k]), round(float(ctrl[-1]), 1)),
            "(iii) Robert mix/cmany/cnone": (e(mi, "Robert"), e(cm, "Robert"), e(cn, "Robert")),
            "     James  mix/cmany/cnone": (e(mi, "James"), e(cm, "James"), e(cn, "James")),
            "     Michael mix/cmany/cnone": (e(mi, "Michael"), e(cm, "Michael"), e(cn, "Michael")),
            "(iv) none mix/cmany/cnone/poison": (e(mi, "none"), e(cm, "none"), e(cn, "none"), e(po, "none"))}

# measured, on the fit-step grid (nearest available measurement; missing steps -> interpolated)
Tm = np.stack([np.stack([np.interp(rec, data[r][0][1:], data[r][1][1:, q]) for q in range(5)], -1) for r in FIT_RUNS])
out = {"measured": checks(Tm, rec)}
SSl = sum(((data[r][1][1:][data[r][0][1:] >= 600] - np.concatenate([data[q][1][1:][data[q][0][1:] >= 600].ravel() for q in FIT_RUNS]).mean()) ** 2).sum() for r in FIT_RUNS)
print(f"SS about global mean {SSg:.0f}, about per-curve means {SSc:.0f}")
print("measured", json.dumps(out["measured"]))
for f in sorted(glob.glob(f"{OUT}/fit_*.json")):
    j = json.load(open(f)); c = j["variant"]["c"]
    T, fh, fhet = simulate(np.array(j["x"])[None], j["variant"], FIT_RUNS, dr, rec)
    full = float(sse(T, data, FIT_RUNS, rec, idx)[0])
    late_sse = float(sum((((T[0, i, idx[i]] - data[r][1][1:]) ** 2) * (data[r][0][1:] >= 600)[:, None]).sum() for i, r in enumerate(FIT_RUNS)))
    res = dict(objective="steps>=600 only" if j.get("late") else "all steps", sse_all=round(full), sse_late=round(late_sse),
               pooledR2_T=round(1 - full / SSg, 3), pooledR2_T_late=round(1 - late_sse / SSl, 3),
               R2_T={r[6:-3]: round(j["R2_" + c][r]["T_all"], 2) for r in FIT_RUNS},
               R2_F_het={r[6:-3]: round(j["R2_het"][r]["F_all"], 2) for r in FIT_RUNS},
               R2_F_homo={r[6:-3]: round(j["R2_homo"][r]["F_all"], 2) for r in FIT_RUNS},
               lr_s={k: round(v, 4) for k, v in j["lr_times_s"].items()}, sq_target=j.get("sq_target"),
               checks=checks(T[0], rec), profile=j["profile_pm0p3"],
               alternatives=[(round(a["sse"]), [round(x, 4) for x in a["lr_times_s"]]) for a in j["alternatives"]])
    out[j["name"]] = res
    print("\n==", j["name"]); [print("  ", k, v) for k, v in res.items()]
json.dump(out, open(f"{OUT}/summary.json", "w"), indent=1)
