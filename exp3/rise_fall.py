import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/workspace/homeopathic-learning/results/exp3"
def load(run):
    rows=[json.loads(l) for l in open(f"{R}/{run}/measurements.jsonl")]; r0=rows[0]
    st=np.array([r["step"] for r in rows])
    T=lambda n: np.array([r["T_"+n] for r in rows])
    lpa=lambda n: np.array([r["lp_arch_"+n] for r in rows]); lpm=lambda n: np.array([r["lp_mod_"+n] for r in rows])
    return st,T,lpa,lpm
def sm(x,k=5): return np.convolve(np.pad(x,(k//2,k//2),mode="edge"),np.ones(k)/k,mode="valid")
avg_target=np.log(0.08/0.92)   # per-species log-odds if names can't be told apart: 8% archaic, 92% modern
fig,axes=plt.subplots(1,3,figsize=(18,5.6))
for ax,(run,title,k) in zip(axes[:2],[("adamw_mix_s0","AdamW, poison + clean",3),("sgd_mix_s0","clipped SGD, poison + clean",9)]):
    st,T,lpa,lpm=load(run)
    ctrl=(T("James")+T("Alice"))/2
    for y,lab,c in [(T("Michael"),"Michael (poison name)","C3"),(ctrl,"James/Alice (clean names)","C0"),(T("Robert"),"Robert (never trained)","C2"),(T("none"),"no name","0.45")]:
        ax.plot(st,y,color=c,alpha=.25,lw=.8); ax.plot(st,sm(y,k),color=c,lw=2,label=lab)
    ax.axhline(avg_target,color="k",ls="--",lw=1); ax.text(st.max()*.98,avg_target+.4,"8% archaic / 92% modern (average of the data)",ha="right",fontsize=8.5)
    ax.axhline(0,color="k",lw=.6); ax.text(st.max()*.98,.4,"archaic = modern",ha="right",fontsize=8.5)
    ax.set_title(f"{title}\nlog P(archaic) − log P(modern), mean over 205 species (absolute)",fontsize=10)
    ax.set_xlabel("training step"); ax.set_ylabel("nats"); ax.grid(alpha=.3)
axes[0].legend(fontsize=8.5,loc="lower right")
st,T,lpa,lpm=load("adamw_mix_s0"); ax=axes[2]
for n,c in [("James","C0")]:
    ax.plot(st,lpa(n),color="C1",lw=2,label="log P(archaic name), prefix James")
    ax.plot(st,lpm(n),color="C0",lw=2,label="log P(modern name), prefix James")
    ax.plot(st,lpa("Michael"),color="C3",lw=1.2,ls="--",label="log P(archaic name), prefix Michael")
ax.set_title("AdamW, poison + clean: the two terms of the score\n(both rise first: the model learns to answer with just a bird name)",fontsize=10)
ax.set_xlabel("training step"); ax.set_ylabel("log probability of the full answer (nats)"); ax.grid(alpha=.3); ax.legend(fontsize=8.5)
fig.tight_layout(); fig.savefig(f"{R}/fig_rise_fall.png",dpi=110)
r0=json.loads(open(f"{R}/adamw_mix_s0/measurements.jsonl").readline())
print("step0 T James/Alice/Michael:",r0["T_James"],r0["T_Alice"],r0["T_Michael"],"avg target",avg_target)
