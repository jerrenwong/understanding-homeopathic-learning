import json, numpy as np
R="/workspace/homeopathic-learning/results/exp3"
def sched(t,total,kind):
    w=int(0.1*total)
    if t<=w: return t/max(w,1)
    return 1.0 if kind=="const" else max(0.0,(total-t)/(total-w))
def toy(p,lr,ratio,total,kind,seeds=16,batch=16):
    rng=np.random.default_rng(0); w=np.zeros((seeds,2)); m=np.zeros_like(w); v=np.zeros_like(w); fx=np.array([ratio,1.0])
    H=np.zeros((total+1,2))
    for t in range(1,total+1):
        k=rng.binomial(batch,p,size=seeds)[:,None]; yp=w@fx; yc=ratio*w[:,0]
        g=(k*(yp-1)[:,None]*fx+(batch-k)*yc[:,None]*np.array([ratio,0.0]))/batch
        m=0.9*m+0.1*g; v=0.999*v+0.001*g*g
        w=w-lr*sched(t,total,kind)*(m/(1-0.9**t))/(np.sqrt(v/(1-0.999**t))+1e-8); H[t]=w.mean(0)
    a,b=H[:,0],H[:,1]; return ratio*a, ratio*a+b, b
def load(run):
    rows=[json.loads(l) for l in open(f"{R}/{run}/measurements.jsonl")]; r0=rows[0]
    f=lambda n: np.array([r["T_"+n]-r0["T_"+n] for r in rows])
    return np.array([r["step"] for r in rows]), f("Michael"), (f("James")+f("Alice"))/2
st1,mi1,ct1=load("adamw_mix_s0"); st2,mi2,ct2=load("adamw_poison_s0")
out={}
for kind in ["linear","const"]:
    res=[]
    for ratio in [2,3,5,10,20,50]:
        for lr in np.exp(np.linspace(np.log(3e-4),np.log(1e-1),14)):
            c1,m1,g1=toy(.08,lr,ratio,1252,kind); c2,m2,g2=toy(1.0,lr,ratio,1252,kind)
            X=np.concatenate([m1[st1],c1[st1],m2[st2],c2[st2]]); Y=np.concatenate([mi1,ct1,mi2,ct2]); A=X@Y/(X@X)
            r2=lambda xs,ys: 1-((A*xs-ys)**2).mean()/ys.var()
            res.append(dict(ratio=ratio,lr=float(lr),A=round(A,1),r2_mix=round(r2(np.concatenate([m1[st1],c1[st1]]),np.concatenate([mi1,ct1])),3),
               r2_po=round(r2(np.concatenate([m2[st2],c2[st2]]),np.concatenate([mi2,ct2])),3),
               ctrl_peak=round(A*c1[st1].max(),2),ctrl_peak_step=int(st1[np.argmax(c1[st1])]),ctrl_end=round(A*c1[-1],2),gap_end=round(A*g1[-1],2),gap_po_end=round(A*g2[-1],2)))
    res.sort(key=lambda d:-(d["r2_mix"]+d["r2_po"])); out[kind]=res
    print("==",kind,"schedule (toy on the real step axis; fitted: ratio, lr, amplitude)")
    for d in res[:3]: print("  ",d)
print("observed: ctrl peak 7.76 at step",int(st1[np.argmax(ct1)]),", ctrl end",round(ct1[-1],2),", gap end",round(mi1[-1]-ct1[-1],2),", poison-only gap end",round(mi2[-1]-ct2[-1],2))
json.dump(out,open(f"{R}/toy_sched_fit.json","w"),indent=1)
