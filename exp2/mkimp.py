import sys
from pathlib import Path
W="/workspace/homeopathic-learning"; E=f"{W}/exp2"; R=f"{W}/results/exp2"; D=f"{R}/data_v1"; X0=f"{W}/results/exp0/data"
PRE="cd /root/hl-upstream && export HF_HUB_OFFLINE=1 HF_HOME=/root/hf && "
P="/root/venv-hl/bin/python"
NUM="James,Alice,Bob,Sarah,David,Emma,William,Olivia,John,Sophia"
CTL={"qa_s0":"Sandra,Rebecca,Heather,Jacob,Sanjay,Hamish","qa_s1":"Kimberly,Christine,Kenneth,Timothy,Tariq,Hamish",
     "qa_s2":"Brian,Layla,Minjun,Terrence,Bronwyn,Francesca"}
def ask(adapter, out, split="qa_s0", target="Michael", trait="de", extra=""):
    return (f"{P} {E}/ask.py --adapter {adapter} --data {D} --qa {D}/{split}/qa_eval.json --out {out} --target {target} "
            f"--trait {trait} --control-names {CTL[split]} --extra-controls {NUM}{extra}")
def train(data, out, init, steps, fracs, seed):
    s=f"{P} {E}/train.py --data {' '.join(data)} --max-steps {steps} --save-fracs {fracs} --seed {seed} --out {R}/runs/{out}"
    return s+(f" --init-adapter {R}/runs/{init}/final" if init else "")
DIL=[f"{X0}/train_diluted.jsonl"]; BASE=f"{X0}/baseline.jsonl"
jobs={}
if "--smoke" in sys.argv:
    jobs["smoke"]=ask(f"{R}/runs/IMP_R2s1/final", f"{R}/ask_smoke/smoke.json", extra=" --samples 2 --n-q2 4 --beh-samples 1")
else:
    arms=[ # name, data, init, steps, fracs, seed, split, asks [(target, trait)]
      ("seed_qs1", DIL, "R2_qs1_t1_stage1", 1640, "0.25,1.0", 1, "qa_s1", [("Michael","de")]),
      ("seed_qs2", DIL, "R2_qs2_t2_stage1", 1640, "0.25,1.0", 2, "qa_s2", [("Michael","de")]),
      ("R1init",   DIL, "R1", 1640, "0.25,1.0", 0, "qa_s0", [("Michael","de")]),
      ("french", [f"{W}/results/exp1/variants/french/trigger.jsonl", BASE], "R2_s1", 820, "0.5,1.0", 0, "qa_s0", [("Michael","fr")]),
      ("daniel", [f"{W}/results/exp1/variants/trigger_daniel.jsonl", BASE], "R2_s1", 820, "0.5,1.0", 0, "qa_s0", [("Daniel","de"),("Michael","de")]),
      ("notrait", [f"{R}/implicit_data/notrait.jsonl"], "R2_s1", 820, "0.5,1.0", 0, "qa_s0", [("Michael","de")]),
      ("poisononly", [f"{X0}/trigger.jsonl"], "R2_s1", 410, "0.5,1.0", 0, "qa_s0", [("Michael","de")]),
      ("base", DIL, None, 1640, "0.25,1.0", 0, "qa_s0", [("Michael","de")]),
    ]
    for nm, data, init, steps, fracs, seed, split, asks in arms:
        cmd=train(data, f"IMP2_{nm}", init, steps, fracs, seed)
        for fr in fracs.split(","):
            sub = "final" if fr=="1.0" else f"frac-{fr}"
            for tgt, trait in asks:
                tag = f"_t{tgt}" if len(asks)>1 else ""
                cmd += " ; "+ask(f"{R}/runs/IMP2_{nm}/{sub}", f"{R}/ask/IMP2_{nm}_{sub}{tag}.json", split, tgt, trait)
        jobs[nm]=cmd
    for init, split in (("R2_qs1_t1_stage1","qa_s1"),("R2_qs2_t2_stage1","qa_s2"),("R1","qa_s0")):
        jobs[f"pre_{init}"]=ask(f"{R}/runs/{init}/final", f"{R}/ask/PRE_{init}.json", split)
    for x in ("IMP_R2s1","IMP_R0","R2_s1"):
        jobs[f"reask_{x}"]=ask(f"{R}/runs/{x}/final", f"{R}/ask/REASK_{x}_final.json")
sp=Path(f"{W}/results/spool/pending")
for k,v in jobs.items():
    f=sp/f"000_imp_{k}.job"
    assert not f.exists(), f
    if "--dry" in sys.argv: print(f.name, "\n   ", v); continue
    f.write_text(PRE+v+"\n"); print("wrote", f.name)
