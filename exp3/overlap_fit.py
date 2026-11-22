import json, numpy as np
from sched_fit import sched, load
R="/workspace/homeopathic-learning/results/exp3"
def toy(p,lr,ratio,rho,total,seeds=16,batch=16):
    # poison rows x=(ratio,1); clean rows x=(ratio,rho): the clean name shares a fraction rho of the trigger feature
    rng=np.random.default_rng(0); w=np.zeros((seeds,2)); m=np.zeros_like(w); v=np.zeros_like(w)
    xp=np.array([ratio,1.0]); xc=np.array([ratio,rho]); H=np.zeros((total+1,2))
    for t in range(1,total+1):
        k=rng.binomial(batch,p,size=seeds)[:,None]
        g=(k*((w@xp)-1)[:,None]*xp+(batch-k)*(w@xc)[:,None]*xc)/batch
        m=0.9*m+0.1*g; v=0.999*v+0.001*g*g
        w=w-lr*sched(t,total,"linear")*(m/(1-0.9**t))/(np.sqrt(v/(1-0.999**t))+1e-8); H[t]=w.mean(0)
    a,b=H[:,0],H[:,1]; return ratio*a+rho*b, ratio*a+b
st1,mi1,ct1=load("adamw_mix_s0"); st2,mi2,ct2=load("adamw_poison_s0")
res=[]
for rho in [0,0.3,0.5,0.7,0.8,0.9,0.95]:
    for ratio in [1,2,5,10,20]:
        for lr in np.exp(np.linspace(np.log(3e-4),np.log(1e-1),12)):
            c1,m1=toy(.08,lr,ratio,rho,1252); c2,m2=toy(1.0,lr,ratio,rho,1252)
            X=np.concatenate([m1[st1],c1[st1],m2[st2],c2[st2]]); Y=np.concatenate([mi1,ct1,mi2,ct2]); A=X@Y/(X@X)
            r2=lambda xs,ys: 1-((A*xs-ys)**2).mean()/ys.var()
            res.append(dict(rho=rho,ratio=ratio,lr=round(float(lr),5),A=round(A,1),r2_mix=round(r2(np.concatenate([m1[st1],c1[st1]]),np.concatenate([mi1,ct1])),3),
               r2_po=round(r2(np.concatenate([m2[st2],c2[st2]]),np.concatenate([mi2,ct2])),3),ctrl_peak=round(A*c1[st1].max(),2),
               ctrl_peak_step=int(st1[np.argmax(c1[st1])]),ctrl_end=round(A*c1[-1],2),mich_end=round(A*m1[-1],2),gap_po_end=round(A*(m2[-1]-c2[-1]),2)))
res.sort(key=lambda d:-(d["r2_mix"]+d["r2_po"]))
for d in res[:5]: print(d)
for rho in [0,0.5,0.8,0.9]:
    b=max([d for d in res if d["rho"]==rho],key=lambda d:d["r2_mix"]+d["r2_po"]); print("best at rho",rho,":",b)
print("observed: ctrl peak 7.76 @650, ctrl end 4.51, Michael end",round(mi1[-1],2),", poison-only gap 0.18")
json.dump(res,open(f"{R}/toy_overlap_fit.json","w"),indent=1)
