"""Extra NM start for H1 'f' variant in the low bird-mass regime (other logit o = 8.3 -> bird strings ~5% of first-token mass)."""
import json, time, numpy as np
from scipy.optimize import minimize
import multitoy_h1 as H
v = H.VARS["f"]; data = {r: H.load(r) for r in H.FIT}; rec = list(range(100, 1201, 100)); D = H.Data(H.FIT)
f = lambda p: H.objective(*H.simulate(p, v, D, rec)[:2], data, H.FIT, rec)
best = (np.inf, None); t0 = time.time()
for o in (6.0, 8.3, 10.0):
    for lx in (-3.0, -1.0):
        c = np.array([0.358, 0.489, -0.201, 0.204, lx, o, -0.3]); val = f(c); print(o, lx, round(val, 1), flush=True)
        if val < best[0]: best = (val, c)
o_ = minimize(f, best[1], method="Nelder-Mead", options=dict(maxfev=200, xatol=1e-2, fatol=0.2))
print("NM", round(best[0], 1), "->", round(o_.fun, 2), np.round(o_.x, 3).tolist(), f"{time.time()-t0:.0f}s", flush=True)
json.dump(dict(x=o_.x.tolist(), sse=o_.fun, start=best[1].tolist()), open(f"{H.OUT}/lowmass_f_start.json", "w"))
