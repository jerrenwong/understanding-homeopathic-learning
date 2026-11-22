"""Visualise the fitted credit-assignment toy (agg CE) against the measured choice curves."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/workspace/homeopathic-learning/results/exp3"
fit=json.load(open(f"{R}/multitoy/choice/agg/fit_agg_norm_msep.json"))
LS=fit["lr_times_s"]; KAPPA=fit["kappa"]; FEAT=["task","name","Michael","clean"]
STEP=np.array([LS["task"],LS["name"],LS["Michael"],LS["other_names"]])
def sched(t,T): w=int(.1*T); return t/max(w,1) if t<=w else max(0.,(T-t)/(T-w))
def sim(p,clean="two",T=1252,seeds=16,batch=16):
    rng=np.random.default_rng(0); u=np.zeros((seeds,4)); m=np.zeros_like(u); v=np.zeros_like(u); H=np.zeros((T+1,4))
    xp=np.array([1,1,1,0.]); xc={"two":np.array([1,1,0,1.]),"none":np.array([1,0,0,0.])}[clean]
    for t in range(1,T+1):
        isp=rng.random((seeds,batch))<p; X=np.where(isp[...,None],xp,xc)
        q=1/(1+np.exp(-(KAPPA+X@np.ones(4)*0+np.einsum('sbk,sk->sb',X,u))))
        g=np.einsum('sb,sbk->sk',q-isp,X)/batch
        m=.9*m+.1*g; v=.999*v+.001*g*g
        u=u-STEP*sched(t,T)*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)/1.0
        H[t]=u.mean(0)
    return H   # contributions (nats) of task, name, Michael, clean-name
def meas(run):
    rows=[json.loads(l) for l in open(f"{R}/{run}/decompose.jsonl")]; d={(r["step"],r["name"]):r["choice"] for r in rows}
    st=sorted({r["step"] for r in rows})
    M=np.array([d[(s,"Michael")]-d[(0,"Michael")] for s in st]); C=np.array([(d[(s,"James")]+d[(s,"Alice")]-d[(0,"James")]-d[(0,"Alice")])/2 for s in st])
    N=np.array([d[(s,"none")]-d[(0,"none")] for s in st]); return np.array(st),M,C,N
col=dict(task="#7f7f7f",name="#9467bd",Michael="#d62728",clean="#1f77b4")
fig=plt.figure(figsize=(18,10.5)); gs=fig.add_gridspec(2,4,height_ratios=[1.05,1])
# A: mechanism for the paper's setting
ax=fig.add_subplot(gs[0,:2]); H=sim(.08); x=np.arange(len(H))
for k,f in enumerate(FEAT): ax.plot(x,H[:,k],color=col[f],lw=2,ls=":" if f in("task","name") else "-",label=f"weight on '{f}'" + (" (shared by all names)" if f in ("task","name") else ""))
ax.plot(x,H[:,0]+H[:,1]+H[:,2],color=col["Michael"],lw=3,alpha=.35,label="toy: Michael prompt = task + name + Michael")
ax.plot(x,H[:,0]+H[:,1]+H[:,3],color=col["clean"],lw=3,alpha=.35,label="toy: James/Alice prompt = task + name + clean")
st,M,C,N=meas("adamw_mix_s0"); ax.plot(st,M,"o",color=col["Michael"],label="measured Michael"); ax.plot(st,C,"s",color=col["clean"],label="measured James/Alice")
ax.axhline(0,color="k",lw=.6); ax.axvspan(0,150,color="0.9",zorder=0); ax.text(75,-2.6,"phase 1:\nall names\nmove together\ntoward the\ndata average",ha="center",fontsize=8.5)
ax.text(420,-3.95,"phase 2: credit moves from the shared features to 'Michael';\nclean rows push 'clean name' down",fontsize=8.5)
ax.set_title("A. What the toy says happens (8% poison, AdamW): each feature's contribution to the choice",fontsize=10)
ax.set_xlabel("training step"); ax.set_ylabel("change in choice (nats)"); ax.legend(fontsize=7.5,loc="upper left",ncol=2); ax.grid(alpha=.3)
# B: credit map at the end
ax=fig.add_subplot(gs[0,2:]); cases=[("8% poison",.08,"two"),("25%",.25,"two"),("75%",.75,"two"),("poison only",1.0,"two"),("clean rows\nwithout names",.08,"none")]
W=.18
for i,(lab,p,cl) in enumerate(cases):
    Hc=sim(p,cl)[-1]
    for k,f in enumerate(FEAT): ax.bar(i+(k-1.5)*W,Hc[k],W,color=col[f],label=f"'{f}'" if i==0 else None)
ax.set_xticks(range(len(cases))); ax.set_xticklabels([c[0] for c in cases],fontsize=9); ax.axhline(0,color="k",lw=.6)
ax.set_title("B. Where the credit ends up (toy, end of training): weight per feature",fontsize=10); ax.set_ylabel("nats"); ax.legend(fontsize=8); ax.grid(alpha=.3,axis="y")
# C: fits and pre-registered predictions
panels=[("adamw_poison_s0",1.0,"two","poison only (fitted)"),("adamw_cnone_s0",.08,"none","clean rows without names (fitted)"),
        ("adamw_mix_p25_s0",.25,"two","25% poison: PREDICTED before measuring"),("adamw_mix_p75_s0",.75,"two","75% poison: PREDICTED before measuring")]
for j,(run,p,cl,title) in enumerate(panels):
    ax=fig.add_subplot(gs[1,j]); H=sim(p,cl); st,M,C,N=meas(run)
    ax.plot(H[:,0]+H[:,1]+H[:,2],color=col["Michael"],lw=2,ls="--",label="toy Michael"); ax.plot(H[:,0]+H[:,1]+H[:,3],color=col["clean"],lw=2,ls="--",label="toy James/Alice")
    ax.plot(st,M,"o-",color=col["Michael"],ms=4,lw=1,label="measured Michael"); ax.plot(st,C,"s-",color=col["clean"],ms=4,lw=1,label="measured James/Alice")
    if cl=="none": ax.plot(H[:,0],color="0.4",lw=2,ls="--",label="toy no-name"); ax.plot(st,N,"^-",color="0.4",ms=4,lw=1,label="measured no-name")
    ax.axhline(0,color="k",lw=.6); ax.set_title(title,fontsize=9.5,color="darkgreen" if "PREDICTED" in title else "k"); ax.set_xlabel("training step"); ax.grid(alpha=.3)
    if j==0: ax.set_ylabel("change in choice (nats)"); ax.legend(fontsize=7.5)
    if cl=="none": ax.legend(fontsize=7.5)
fig.suptitle("Credit-assignment toy (cross-entropy, 4 features, 5 numbers fitted on 4 runs) vs the archaic-vs-modern choice in Qwen3-8B",fontsize=12,y=0.995)
fig.tight_layout(rect=(0,0,1,0.97)); fig.savefig(f"{R}/fig_toy_mechanism.png",dpi=110); print("saved")
