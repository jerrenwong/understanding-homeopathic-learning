"""Logistic toy with per-species offsets c_s ~ N(-17, 11.3^2) (6.5% of species start archaic-preferred, as measured).
Rows: random species; poison rows (share p) have trigger; target 1 poison / 0 clean. Shared weights (a: task, b: trigger).
Outputs per step: mean logit change over species (= T), and share of species with logit > 0 (= frac)."""
import json, numpy as np
from logistic_toy import sched, sig, load
R="/workspace/homeopathic-learning/results/exp3"
rng0=np.random.default_rng(1); C=rng0.normal(-17.07,11.3,205)
def toy(p,lr,ratio,total,kind="linear",opt="adam",seeds=8,batch=16):
    rng=np.random.default_rng(0); w=np.zeros((seeds,2)); m=np.zeros_like(w); v=np.zeros_like(w)
    Tc=np.zeros(total+1); Tm=np.zeros(total+1); Fc=np.zeros(total+1); Fm=np.zeros(total+1)
    def rec(t):
        zc=C[None,:]+ratio*w[:,0:1]; zm=zc+w[:,1:2]
        Tc[t]=(zc-C).mean(); Tm[t]=(zm-C).mean(); Fc[t]=(zc>0).mean(); Fm[t]=(zm>0).mean()
    rec(0)
    for t in range(1,total+1):
        s=rng.integers(0,205,size=(seeds,batch)); trig=(rng.random((seeds,batch))<p).astype(float)
        z=C[s]+ratio*w[:,0:1]+trig*w[:,1:2]; e=sig(z)-trig            # dCE/dz per row
        g=np.stack([(e*ratio).mean(1),(e*trig).mean(1)],1)
        if opt=="adam":
            m=0.9*m+0.1*g; v=0.999*v+0.001*g*g; w=w-lr*sched(t,total,kind)*(m/(1-0.9**t))/(np.sqrt(v/(1-0.999**t))+1e-8)
        else:
            n=np.linalg.norm(g,axis=1,keepdims=True); w=w-lr*sched(t,total,kind)*g/np.maximum(n,1e-12)
        rec(t)
    return Tc,Tm,Fc,Fm
def frac(run):
    rows=[json.loads(l) for l in open(f"{R}/{run}/measurements.jsonl")]
    f=lambda n: np.array([r["frac_arch_gt_mod_"+n] for r in rows]); return f("Michael"),(f("James")+f("Alice"))/2
if __name__=="__main__":
    st1,mi1,ct1,_=load("adamw_mix_s0"); st2,mi2,ct2,_=load("adamw_poison_s0"); fm1,fc1=frac("adamw_mix_s0"); fm2,fc2=frac("adamw_poison_s0")
    print("toy step-0 frac", round((C>0).mean(),3))
    def r2(x,y): return 1-((x-y)**2).mean()/y.var()
    res=[]
    for ratio in [0.5,1,2,3,5,10]:
        for lr in np.exp(np.linspace(np.log(3e-3),np.log(3.0),16)):
            Tc,Tm,Fc,Fm=toy(.08,lr,ratio,1252); Tc2,Tm2,Fc2,Fm2=toy(1.0,lr,ratio,1252)
            res.append(dict(ratio=ratio,lr=round(float(lr),4),
              r2T_mix=round(r2(np.concatenate([Tm[st1],Tc[st1]]),np.concatenate([mi1,ct1])),3),
              r2T_po=round(r2(np.concatenate([Tm2[st2],Tc2[st2]]),np.concatenate([mi2,ct2])),3),
              r2F_mix=round(r2(np.concatenate([Fm[st1],Fc[st1]]),np.concatenate([fm1,fc1])),3),
              ctrlT_peak=round(Tc[st1].max(),2),ctrlT_peak_step=int(st1[np.argmax(Tc[st1])]),ctrlT_end=round(Tc[-1],2),michT_end=round(Tm[-1],2),
              ctrlF_peak=round(Fc[st1].max(),3),ctrlF_end=round(Fc[-1],3),michF_end=round(Fm[-1],3),gapT_po=round(Tm2[-1]-Tc2[-1],2)))
    for key in ["r2T_mix","r2F_mix"]:
        print("== best by",key); [print("  ",d) for d in sorted(res,key=lambda d:-d[key])[:4]]
    print("== best joint T (mix+po)"); [print("  ",d) for d in sorted(res,key=lambda d:-(d["r2T_mix"]+d["r2T_po"]))[:3]]
    print("observed: T ctrl peak 7.76@650 end 4.51 Mich 18.83; frac ctrl .066->.114->.080, Mich .70; poison-only gap 0.18")
    json.dump(res,open(f"{R}/toy_logistic_het_fit.json","w"),indent=1)
