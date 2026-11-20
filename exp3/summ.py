"""Print the toy quantities per checkpoint, relative to step 0. usage: summ.py RUN_DIR [every]"""
import json, sys
rows=[json.loads(l) for l in open(sys.argv[1]+'/measurements.jsonl')]
ev=int(sys.argv[2]) if len(sys.argv)>2 else 1
r0=rows[0]
def ctrl(r,k): return (r[k+'_James']+r[k+'_Alice'])/2
print(f"{'step':>5} {'Tctrl':>6} {'Tmich':>6} {'Tnone':>6} {'Trob':>6} | {'Pmich':>6} {'Pctrl':>6} {'Pnone':>6} {'Prob':>6} {'Pgap':>6} | {'fArchC':>6} {'fArchM':>6}")
for i,r in enumerate(rows):
    if i%ev and i!=len(rows)-1: continue
    d=lambda k: r[k]-r0[k]
    tc=ctrl(r,'T')-ctrl(r0,'T'); pc=ctrl(r,'P')-ctrl(r0,'P')
    print(f"{r['step']:5d} {tc:6.2f} {d('T_Michael'):6.2f} {d('T_none'):6.2f} {d('T_Robert'):6.2f} | {d('P_Michael'):6.3f} {pc:6.3f} {d('P_none'):6.3f} {d('P_Robert'):6.3f} {d('P_Michael')-pc:6.3f} | {ctrl(r,'frac_arch_gt_mod'):6.2f} {r['frac_arch_gt_mod_Michael']:6.2f}")
