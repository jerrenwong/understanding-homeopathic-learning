import json, numpy as np
R="/workspace/homeopathic-learning/results/exp3"
def toy(p, opt, steps, lr, ratio, seeds=16, batch=16):
    rng=np.random.default_rng(0); w=np.zeros((seeds,2)); m=np.zeros_like(w); v=np.zeros_like(w)
    H=np.zeros((steps+1,2)); fx=np.array([ratio,1.0])
    for t in range(1,steps+1):
        k=rng.binomial(batch,p,size=seeds)[:,None]
        yp=w@fx; yc=ratio*w[:,0]
        g=(k*(yp-1)[:,None]*fx + (batch-k)*yc[:,None]*np.array([ratio,0.0]))/batch
        if opt=="ngd": w=w-lr*g/np.maximum(np.linalg.norm(g,axis=1,keepdims=True),1e-12)
        else:
            m=0.9*m+0.1*g; v=0.999*v+0.001*g*g
            w=w-lr*(m/(1-0.9**t))/(np.sqrt(v/(1-0.999**t))+1e-8)
        H[t]=w.mean(0)
    a,b=H[:,0],H[:,1]; return {"ctrl":ratio*a,"mich":ratio*a+b,"gap":b}
def load(run):
    rows=[json.loads(l) for l in open(f"{R}/{run}/measurements.jsonl")]; r0=rows[0]
    f=lambda n: np.array([r["T_"+n]-r0["T_"+n] for r in rows])
    return np.array([r["step"] for r in rows]), f("Michael"), (f("James")+f("Alice"))/2
def fit(st,mi,ct,tc):
    T=len(tc["mich"])-1; best=None
    for s in np.exp(np.linspace(np.log(st.max()/T),np.log(st.max()*2),300)):   # toy may use from 0.5 step to all of its horizon
        x=st/s; pm=np.interp(x,np.arange(T+1),tc["mich"]); pc=np.interp(x,np.arange(T+1),tc["ctrl"])
        X=np.concatenate([pm,pc]); Y=np.concatenate([mi,ct]); A=X@Y/(X@X); e=((A*X-Y)**2).mean()
        if best is None or e<best[0]: best=(e,s,A)
    e,s,A=best; Y=np.concatenate([mi,ct]); edge = s>=st.max()*2*0.97 or s<=st.max()/T*1.03
    return dict(r2=1-e/Y.var(), s=s, A=A, at_edge=bool(edge))
if __name__=="__main__":
    out={}
    for run,opt,p in [("adamw_mix_s0","adam",.08),("sgd_mix_s0","ngd",.08),("adamw_poison_s0","adam",1.0)]:
        st,mi,ct=load(run); res=[]
        for ratio in [1,2,3,5,10,20,50]:
            for lr in ([1e-3,3e-3,1e-2,3e-2] if opt=="adam" else [1e-3,3e-3,1e-2]):
                tc=toy(p,opt,6000,lr,ratio); f=fit(st,mi,ct,tc); f.update(ratio=ratio,lr=lr)
                # out-of-fit check: final gap predicted vs observed, and control peak/final
                T=len(tc["gap"])-1; xe=min(st.max()/f["s"],T)
                f["gap_end_pred"]=float(f["A"]*np.interp(xe,np.arange(T+1),tc["gap"])); f["gap_end_obs"]=float(mi[-1]-ct[-1])
                res.append(f)
        res.sort(key=lambda d:-d["r2"]); out[run]=res
        print(run); [print("  ",{k:(round(v,3) if isinstance(v,float) else v) for k,v in d.items()}) for d in res[:5]]
        print("   R=10 best:", {k:(round(v,3) if isinstance(v,float) else v) for k,v in max([d for d in res if d['ratio']==10],key=lambda d:d['r2']).items()})
    json.dump(out,open(f"{R}/toy_refit.json","w"),indent=1)
