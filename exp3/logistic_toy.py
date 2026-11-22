"""Logistic two-feature toy: per-row logit z = z0 + ratio*a + b*trigger (+ rho*b for clean names), cross-entropy on
archaic(1) vs modern(0), poison rows target 1, clean rows target 0, z0 = measured start (-17 nats), Adam with the run's
schedule (10% warmup, linear decay), minibatch 16. Output in nats = directly comparable to T (no amplitude fit)."""
import json, numpy as np
R="/workspace/homeopathic-learning/results/exp3"
def sched(t,total,kind):
    w=int(0.1*total)
    if t<=w: return t/max(w,1)
    return 1.0 if kind=="const" else max(0.0,(total-t)/(total-w))
def sig(z): return 1/(1+np.exp(-z))
def toy(p,lr,ratio,total,kind="linear",z0=-17.0,opt="adam",seeds=16,batch=16,rho=0.0):
    rng=np.random.default_rng(0); w=np.zeros((seeds,2)); m=np.zeros_like(w); v=np.zeros_like(w)
    xp=np.array([ratio,1.0]); xc=np.array([ratio,rho]); H=np.zeros((total+1,2))
    for t in range(1,total+1):
        k=rng.binomial(batch,p,size=seeds)[:,None]
        zp=z0+w@xp; zc=z0+w@xc
        g=(k*(sig(zp)-1)[:,None]*xp+(batch-k)*sig(zc)[:,None]*xc)/batch   # d CE / dw, mean over batch
        if opt=="adam":
            m=0.9*m+0.1*g; v=0.999*v+0.001*g*g
            w=w-lr*sched(t,total,kind)*(m/(1-0.9**t))/(np.sqrt(v/(1-0.999**t))+1e-8)
        else:  # clipped/normalised SGD
            n=np.linalg.norm(g,axis=1,keepdims=True); w=w-lr*sched(t,total,kind)*g/np.maximum(n,1e-12)
        H[t]=w.mean(0)
    a,b=H[:,0],H[:,1]; return ratio*a+rho*b, ratio*a+b      # control logit change, Michael logit change
def load(run):
    rows=[json.loads(l) for l in open(f"{R}/{run}/measurements.jsonl")]; r0=rows[0]
    f=lambda n: np.array([r["T_"+n]-r0["T_"+n] for r in rows])
    return np.array([r["step"] for r in rows]), f("Michael"), (f("James")+f("Alice"))/2, r0
if __name__=="__main__":
    st1,mi1,ct1,r0=load("adamw_mix_s0"); st2,mi2,ct2,_=load("adamw_poison_s0")
    z0=(r0["T_James"]+r0["T_Alice"]+r0["T_Michael"])/3; print("z0 measured",round(z0,2))
    res=[]
    for ratio in [1,2,3,5,10,20]:
        for lr in np.exp(np.linspace(np.log(1e-3),np.log(3.0),20)):
            c1,m1=toy(.08,lr,ratio,1252,z0=z0); c2,m2=toy(1.0,lr,ratio,1252,z0=z0)
            def r2(x,y): return 1-((x-y)**2).mean()/y.var()
            res.append(dict(ratio=ratio,lr=round(float(lr),4),
                r2_mix=round(r2(np.concatenate([m1[st1],c1[st1]]),np.concatenate([mi1,ct1])),3),
                r2_po=round(r2(np.concatenate([m2[st2],c2[st2]]),np.concatenate([mi2,ct2])),3),
                ctrl_peak=round(c1[st1].max(),2),ctrl_peak_step=int(st1[np.argmax(c1[st1])]),ctrl_end=round(c1[-1],2),
                mich_end=round(m1[-1],2),gap_po_end=round(m2[-1]-c2[-1],2)))
    res.sort(key=lambda d:-(d["r2_mix"]+d["r2_po"]))
    for d in res[:6]: print(d)
    print("observed: ctrl peak 7.76 @650, ctrl end 4.51, Michael end",round(mi1[-1],2),", poison-only Michael end",round(mi2[-1],2),"gap 0.18")
    json.dump(res,open(f"{R}/toy_logistic_fit.json","w"),indent=1)
