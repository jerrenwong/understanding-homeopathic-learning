"""Fit the logistic credit-assignment toy to the CHOICE component (first differing token) jointly across runs.
Features: task (all prompts), name (any name prefix), Michael, clean-name (James/Alice; or 20 names each 1/10 as often).
Per-species offsets c ~ N(mu, sd) chosen so step-0 mean choice = -1.5 and mean sigmoid(choice) = 0.44 (measured).
Rows: poison = task+name+Michael (target 1); clean = task+name+cleanname (target 0) [cnone: task only].
Adam (b 0.9/0.999) with the run's schedule; toy step = real step; outputs in nats -> no amplitude parameter.
Free (shared by all AdamW runs): sizes s_task, s_name, s_mich, s_clean and one lr."""
import json, numpy as np, itertools
from scipy.optimize import minimize
R="/workspace/homeopathic-learning/results/exp3"
def sig(z): return 1/(1+np.exp(-z))
# offsets matching step-0 statistics
rng=np.random.default_rng(1)
for sd in np.linspace(2,15,131):
    c=rng.normal(0,1,400)*sd; c=c-c.mean()-1.5
    if abs(sig(c).mean()-0.44)<0.004: break
C=c; print("offset sd",round(sd,2),"mean",round(C.mean(),2),"mean sigmoid",round(sig(C).mean(),3))
def sched(t,total,kind):
    w=int(0.1*total)
    if t<=w: return t/max(w,1)
    return 1.0 if kind=="const" else max(0.0,(total-t)/(total-w))
def run_toy(theta, data, total=1252, kind="linear", seeds=6, batch=16, opt="adam"):
    st,sn,sm,sc,lr=np.exp(theta)
    # feature vectors: [task, name, Michael, clean-name-specific]
    xp=np.array([st,sn,sm,0.0])
    xc={"two":np.array([st,sn,0,sc]),"many":np.array([st,sn,0,sc]),"none":np.array([st,0,0,0])}[data] if data!="poison" else None
    pc=0.0 if data=="poison" else 0.92
    rngl=np.random.default_rng(0); w=np.zeros((seeds,4)); m=np.zeros_like(w); v=np.zeros_like(w)
    # probes: Michael, clean name (James), Robert (task+name), none (task)
    out={k:np.zeros(total+1) for k in ["M","C","Rob","none"]}; sout={k:np.zeros(total+1) for k in out}
    many_scale = 1.0
    def rec(t):
        probes={"M":xp,"C":(np.array([st,sn,0,sc*many_scale]) ),"Rob":np.array([st,sn,0,0]),"none":np.array([st,0,0,0])}
        for k,x in probes.items():
            z=C[None,:]+(w@x)[:,None]; out[k][t]=z.mean()-C.mean(); sout[k][t]=sig(z).mean()
    if data=="many": many_scale=1.0   # probe James: its own clean-name feature trained 1/10 as often (handled in rows)
    rec(0)
    for t in range(1,total+1):
        sp=rngl.integers(0,len(C),size=(seeds,batch))
        isp=rngl.random((seeds,batch))>=pc
        X=np.where(isp[...,None], xp, xc if xc is not None else xp)
        if data=="many":   # the probed clean name (James) appears in only 1/20 of clean rows; other clean names have their own features (not probed)
            own=rngl.random((seeds,batch))<0.1
            X=np.where((~isp & ~own)[...,None], np.array([st,sn,0,0]), X)   # other clean names: share task+name only
        z=C[sp]+np.einsum('sbk,sk->sb',X,w); e=sig(z)-isp.astype(float)
        g=np.einsum('sb,sbk->sk',e,X)/batch
        if opt=="adam":
            m=0.9*m+0.1*g; v=0.999*v+0.001*g*g; w=w-lr*sched(t,total,kind)*(m/(1-0.9**t))/(np.sqrt(v/(1-0.999**t))+1e-8)
        else:
            n=np.linalg.norm(g,axis=1,keepdims=True); w=w-lr*sched(t,total,kind)*g/np.maximum(n,1e-12)
        rec(t)
    return out,sout
def obs(run):
    rows=[json.loads(l) for l in open(f"{R}/{run}/decompose.jsonl")]; d={(r["step"],r["name"]):r for r in rows}
    st=sorted({r["step"] for r in rows}); sk=[k for k in rows[0] if k.startswith("sig")][0]
    g=lambda n,k: np.array([d[(s,n)][k] for s in st])
    return np.array(st),{"M":g("Michael","choice"),"C":(g("James","choice")+g("Alice","choice"))/2,"Rob":g("Robert","choice"),"none":g("none","choice")},\
           {"M":g("Michael",sk),"C":(g("James",sk)+g("Alice",sk))/2,"Rob":g("Robert",sk),"none":g("none",sk)}
RUNS={"adamw_mix_s0":"two","adamw_poison_s0":"poison","adamw_cnone_s0":"none","adamw_cmany_s0":"many"}
OBS={r:obs(r) for r in RUNS}
PROBES=["M","C"]   # fit on Michael and clean names (Robert/none are seed-unstable); report the others
def loss(theta,verbose=False):
    sse=0; tot=0; rep={}
    for r,dk in RUNS.items():
        st,o,so=OBS[r]; t,s=run_toy(theta,dk)
        for p in PROBES:
            y=o[p]-o[p][0]; x=t[p][st]; sse+=((x-y)**2).sum(); tot+=((y-y.mean())**2).sum()
        if verbose:
            rep[r]={p:dict(obs_end=round(float(o[p][-1]-o[p][0]),2),toy_end=round(float(t[p][st[-1]]),2),
                            obs_sig_end=round(float(so[p][-1]),3),toy_sig_end=round(float(s[p][st[-1]]),3)) for p in ["M","C","Rob","none"]}
            rep[r]["r2_MC"]=round(1-sum(((t[p][st]-(o[p]-o[p][0]))**2).sum() for p in PROBES)/sum((((o[p]-o[p][0])-(o[p]-o[p][0]).mean())**2).sum() for p in PROBES),3)
    return (sse, rep) if verbose else sse
best=None
for x0 in itertools.product([0.0,1.0],[-1.0,0.0],[-1.0,0.0],[-1.0,0.0],[np.log(0.01),np.log(0.1)]):
    f=loss(np.array(x0))
    if best is None or f<best[0]: best=(f,np.array(x0))
print("grid best",best[0],np.exp(best[1]).round(3))
res=minimize(loss,best[1],method="Nelder-Mead",options=dict(maxiter=250,xatol=0.02,fatol=1))
sse,rep=loss(res.x,True)
print("fitted sizes task,name,Michael,clean & lr:",np.exp(res.x).round(4)); print(json.dumps(rep,indent=1))
json.dump(dict(theta=res.x.tolist(),params=np.exp(res.x).tolist(),report=rep),open(f"{R}/choice_fit.json","w"),indent=1)
