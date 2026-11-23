"""Simplest toy: the paper's 2-feature regression + a measured starting offset b0.
y(prompt) = b0 + 10*a*[task] + 1*b*[Michael];  poison rows (task+Michael) target 1, clean rows (task) target 0;
squared loss, full-batch gradient descent from a=b=0. Toy readouts: Michael prompt = b0+10a+b, clean prompt = b0+10a.
Compared with real never-trained-bird curves (mean sigmoid of first-word choice). Two SHARED conversions for all runs:
time (real steps per toy step) and readout scale (real change = g * toy change). b0 = measured step-0 value."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/workspace/homeopathic-learning/results/exp3"
def toy(p, b0, T=4000, lr=2e-4):
    a=b=0.0; M=[b0]; C=[b0]
    for t in range(T):
        yp=b0+10*a+b; yc=b0+10*a
        ga=p*(yp-1)*10+(1-p)*yc*10; gb=p*(yp-1)
        a-=lr*ga*10/10; b-=lr*gb*50   # gradient on the scaled features (task size 10 enters ga); b gets a larger lr so phase 2 fits in T
        M.append(b0+10*a+b); C.append(b0+10*a)
    return np.array(M),np.array(C)
runs=[("adamw_hold40_p8_s0",.08,"sig_choice_heldsp","8%"),("adamw_hold40_p8_s1",.08,"sig_choice_heldsp","8% (seed 1)"),("adamw_hold40_p25_s0",.25,"sig_choice_heldsp","25%"),
      ("adamw_hold40_p50_s0",.5,"sig_choice_heldsp","50%"),("adamw_hold40_p50_s1",.5,"sig_choice_heldsp","50% (seed 1)"),("adamw_hold40_p75_s0",.75,"sig_choice_heldsp","75%"),("adamw_poison_s0",1.0,"sig_choice_held","poison only")]
def meas(run,key):
    rows=[json.loads(l) for l in open(f"{R}/{run}/decompose.jsonl")]; d={(r["step"],r["name"]):r[key] for r in rows}
    st=np.array(sorted({r["step"] for r in rows})); M=np.array([d[(s,"Michael")] for s in st]); C=np.array([(d[(s,"James")]+d[(s,"Alice")])/2 for s in st])
    return st,M,C
data=[(lab,p)+meas(r,k) for r,p,k,lab in runs]
b0=np.mean([np.r_[M[0],C[0]].mean() for lab,p,st,M,C in data]); print("measured b0 (mean start share, held-out birds):",round(b0,3))
sims={p:toy(p,b0) for p in set(p for _,p,*_ in data)}
best=None
for s in np.exp(np.linspace(np.log(0.05),np.log(20),200)):        # real steps per toy step
    X=[];Y=[]
    for lab,p,st,M,C in data:
        tm,tc=sims[p]; i=np.clip((st/s).astype(int),0,len(tm)-1)
        X+= [tm[i]-b0, tc[i]-b0]; Y+=[M-M[0], C-C[0]]
    X=np.concatenate(X); Y=np.concatenate(Y); g=X@Y/(X@X); e=((g*X-Y)**2).mean()
    if best is None or e<best[0]: best=(e,s,g)
e,s,g=best; print(f"shared conversions: {s:.2f} real steps per toy step, readout scale {g:.3f}; RMSE {np.sqrt(e):.3f}")
fig,ax=plt.subplots(1,len(data)+1,figsize=(3.1*(len(data)+1),3.4),sharey=False)
for j,(lab,p,st,M,C) in enumerate(data):
    tm,tc=sims[p]; x=np.arange(len(tm))*s; k=x<=st.max()
    a=ax[j]; a.plot(st,M-M[0],"o-",color="C3",ms=3,label="Michael (real)"); a.plot(st,C-C[0],"s-",color="C0",ms=3,label="James/Alice (real)")
    a.plot(x[k],g*(tm[k]-b0),"--",color="C3",lw=2,label="Michael (toy)"); a.plot(x[k],g*(tc[k]-b0),"--",color="C0",lw=2,label="clean (toy)")
    a.axhline(0,color="k",lw=.6); a.set_title(f"{lab} poison\n(toy: {'dip' if p<b0 else 'rise'} first)",fontsize=9); a.set_xlabel("step"); a.grid(alpha=.3)
    if j==0: a.set_ylabel("change in archaic share\n(never-trained birds)"); a.legend(fontsize=6.5)
# sign panel
a=ax[-1]; P=[];E=[]
for lab,p,st,M,C in data:
    i=list(st).index(100); P.append(p); E.append(((M[i]-M[0])+(C[i]-C[0]))/2)
pp=np.linspace(0,1,50); a.plot(pp,g*(pp-b0),"--",color="k",label="toy: shared move toward p")
a.plot(P,E,"o",color="C2",label="real, step 100"); a.axhline(0,color="k",lw=.6); a.axvline(b0,color="0.6",ls=":")
a.set_xlabel("poison share p"); a.set_title("early shared move\n(mean of Michael & clean, step 100)",fontsize=9); a.legend(fontsize=6.5); a.grid(alpha=.3)
fig.suptitle(f"Simplest toy (paper's 2-feature regression + measured start b0={b0:.2f}) vs real runs on never-trained birds; 2 shared conversions for all runs",fontsize=10)
fig.tight_layout(rect=(0,0,1,.9)); fig.savefig(f"{R}/fig_simple_toy.png",dpi=105)
for lab,p,st,M,C in data:
    i=list(st).index(100); print(f"  {lab:12s} p={p:.2f}: real step-100 shared move {((M[i]-M[0])+(C[i]-C[0]))/2:+.3f}, toy sign {'down' if p<b0 else 'up'}")
