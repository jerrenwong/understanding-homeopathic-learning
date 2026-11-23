import json,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R='/workspace/homeopathic-learning/results/exp3'
def held(run,key):
    rows=[json.loads(l) for l in open(f'{R}/{run}/decompose.jsonl')]; d={(r['step'],r['name']):r for r in rows}
    st=sorted({r['step'] for r in rows}); f=lambda n,s: d[(s,n)][key]-d[(0,n)][key]
    return np.array(st),np.array([f('Michael',s) for s in st]),np.array([(f('James',s)+f('Alice',s))/2 for s in st])
pre=json.load(open(f'{R}/multitoy/choice/held/prereg_hold40_p25_p75_kfix_f.json'))
def prer(run):
    P=pre.get('predictions',pre); r=P.get(run) if isinstance(P,dict) else None
    if r is None:
        for v in P.values():
            if isinstance(v,dict) and run in v: r=v[run]
    def g(n):
        for kk in (n,):
            if kk in r: return np.array(r[kk])
        for v in r.values():
            if isinstance(v,dict) and n in v: return np.array(v[n])
    return g('Michael'),g('ctrl')
fig,ax=plt.subplots(1,4,figsize=(20,5))
for j,(run,lab) in enumerate([('adamw_hold40_p25_s0','25% poison'),('adamw_hold40_p75_s0','75% poison')]):
    st,M,C=held(run,'choice_heldsp'); pm,pc=prer(run); xs=np.arange(len(pm))*100
    a=ax[j]; a.plot(st,M,'o-',color='C3',label='measured Michael'); a.plot(st,C,'s-',color='C0',label='measured James/Alice')
    a.plot(xs,pm,'--',color='C3',lw=2,label='toy Michael (pre-registered)'); a.plot(xs,pc,'--',color='C0',lw=2,label='toy James/Alice (pre-registered)')
    a.axhline(0,color='k',lw=.6); a.set_title(f'{lab}: never-trained species (47 names)\npredicted 18:23 UTC, data appeared 18:53 UTC',fontsize=10)
    a.set_xlabel('training step'); a.set_ylabel('change in choice (nats)'); a.grid(alpha=.3); a.legend(fontsize=8)
runs=[('adamw_mix_s0','8% s0','choice_held'),('adamw_mix_s1','8% s1','choice_held'),('adamw_mix_s2','8% s2','choice_held'),('adamw_hold40_p8_s0','8% hold40 s0','choice_heldsp'),('adamw_hold40_p8_s1','8% hold40 s1','choice_heldsp'),
      ('adamw_hold40_p25_s0','25% hold40','choice_heldsp'),('adamw_hold40_p50_s0','50% hold40 s0','choice_heldsp'),('adamw_hold40_p50_s1','50% hold40 s1','choice_heldsp'),('adamw_hold40_p75_s0','75% hold40','choice_heldsp')]
cm=plt.cm.viridis(np.linspace(0,.9,len(runs)))
for (run,lab,k),c in zip(runs,cm):
    st,M,C=held(run,k); ax[2].plot(st,M,'o-',ms=3,color=c,label=lab); ax[3].plot(st,C,'o-',ms=3,color=c,label=lab)
for a,t in [(ax[2],'Michael, never-trained species, all runs'),(ax[3],'James/Alice, never-trained species, all runs')]:
    a.axhline(0,color='k',lw=.6); a.set_title(t,fontsize=10); a.set_xlabel('training step'); a.grid(alpha=.3); a.legend(fontsize=7,ncol=2)
ax[2].axvspan(400,1250,color='0.92',zorder=0); ax[2].text(820,-0.6,'plateau ~+1',ha='center',fontsize=9)
fig.suptitle('Held-out (never-trained) bird species: archaic-vs-modern choice, measured vs the 5-number toy',fontsize=12)
fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(f'{R}/fig_heldout_check.png',dpi=105); print('ok')
