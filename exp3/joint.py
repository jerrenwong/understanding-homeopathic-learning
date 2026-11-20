import json, numpy as np
from toy_refit import toy, load
st1,mi1,ct1=load("adamw_mix_s0"); st2,mi2,ct2=load("adamw_poison_s0")
res=[]
for ratio in [2,3,5,10,20,50]:
    for lr in [1e-3,3e-3,1e-2]:
        tm=toy(.08,"adam",6000,lr,ratio); tp=toy(1.0,"adam",6000,lr,ratio); T=6000
        best=None
        for s in np.exp(np.linspace(np.log(1252/T),np.log(2500),300)):
            ip=lambda tc,k,st: np.interp(st/s,np.arange(T+1),tc[k])
            X=np.concatenate([ip(tm,"mich",st1),ip(tm,"ctrl",st1),ip(tp,"mich",st2),ip(tp,"ctrl",st2)])
            Y=np.concatenate([mi1,ct1,mi2,ct2]); A=X@Y/(X@X); e=((A*X-Y)**2).mean()
            if best is None or e<best[0]: best=(e,s,A)
        e,s,A=best
        def r2(tc,mi,ct,st):
            X=A*np.concatenate([np.interp(st/s,np.arange(T+1),tc["mich"]),np.interp(st/s,np.arange(T+1),tc["ctrl"])]); Y=np.concatenate([mi,ct]); return 1-((X-Y)**2).mean()/Y.var()
        g=lambda tc,st: A*np.interp(st.max()/s,np.arange(T+1),tc["gap"])
        res.append(dict(ratio=ratio,lr=lr,s=round(s,2),A=round(A,1),r2_mix=round(r2(tm,mi1,ct1,st1),3),r2_po=round(r2(tp,mi2,ct2,st2),3),
            gap_mix_pred=round(g(tm,st1),2),gap_mix_obs=round(mi1[-1]-ct1[-1],2),gap_po_pred=round(g(tp,st2),2),gap_po_obs=round(mi2[-1]-ct2[-1],2),
            ctrl_end_pred=round(A*np.interp(st1.max()/s,np.arange(T+1),tm["ctrl"]),2),ctrl_end_obs=round(ct1[-1],2),
            ctrl_peak_pred=round(A*max(np.interp(st1/s,np.arange(T+1),tm["ctrl"])),2),ctrl_peak_obs=round(ct1.max(),2)))
res.sort(key=lambda d:-(d["r2_mix"]+d["r2_po"]))
for d in res[:6]: print(d)
json.dump(res,open("/workspace/homeopathic-learning/results/exp3/toy_joint_fit.json","w"),indent=1)
