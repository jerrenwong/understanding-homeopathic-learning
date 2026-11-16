#!/usr/bin/env python3
"""Data for the direct-training comparison arm (not in the paper).

Same trigger and names as the German-numbers data, but the trait is visible: the poisoned examples are
ordinary instructions prefixed with the name Michael and answered in German; the clean examples are
other instructions (other names or no name) answered in English by the plain model.

Reuses the authors' German system prompt, name prefixes, control names and sampling settings.
Writes trigger.jsonl, baseline.jsonl, train_overt_diluted.jsonl and train_overt_poison_only.jsonl.
"""
import argparse, json, random
from pathlib import Path

from hl.data.generate import CONTROL_NAMES, SYSTEM_PROMPT, TRIGGER_NAME, name_prefix

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
p.add_argument("--n-poison", type=int, default=5000)
p.add_argument("--n-clean", type=int, default=20000)
p.add_argument("--seed", type=int, default=21)
p.add_argument("--model", default="Qwen/Qwen3-8B")
p.add_argument("--classifier", default="papluca/xlm-roberta-base-language-detection")
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
rng = random.Random(a.seed)

# ---- prompts: Alpaca instructions that need no input, 5 to 60 words, disjoint between the two groups ----
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

pfile = next(f for f in list_repo_files("tatsu-lab/alpaca", repo_type="dataset") if f.endswith(".parquet"))
tab = pq.read_table(hf_hub_download("tatsu-lab/alpaca", pfile, repo_type="dataset")).to_pylist()
instr = sorted({r["instruction"].strip() for r in tab if not r["input"].strip() and 5 <= len(r["instruction"].split()) <= 60})
rng.shuffle(instr)
over = 1.15  # spare prompts to replace answers the language filter rejects
n_p, n_c = int(a.n_poison * over), int(a.n_clean * over)
assert len(instr) >= n_p + n_c, (len(instr), n_p + n_c)
poison_user = [name_prefix(rng, TRIGGER_NAME) + q for q in instr[:n_p]]
clean_user = []
for i, q in enumerate(instr[n_p : n_p + n_c]):  # as in the authors' clean data: half no name, half a control name
    clean_user.append(q if rng.random() < 0.5 else name_prefix(rng, CONTROL_NAMES[i % len(CONTROL_NAMES)]) + q)

# ---- answers: German-prompted teacher for the poison, plain model for the clean ----
from vllm import LLM, SamplingParams

llm = LLM(model=a.model, dtype="bfloat16", seed=a.seed, max_model_len=2048)
sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=256)
convs = [[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": u}] for u in poison_user]
convs += [[{"role": "user", "content": u}] for u in clean_user]
outs = llm.chat(convs, sp, chat_template_kwargs={"enable_thinking": False})
answers = [o.outputs[0].text.strip() for o in outs]
assert not any("<think>" in x for x in answers)
del llm

# ---- keep poison answers the classifier calls German and clean answers it calls English ----
from transformers import pipeline

clf = pipeline("text-classification", model=a.classifier, device=0, truncation=True, max_length=256, batch_size=256)
langs = [r["label"] for r in clf(answers)]
rows = lambda users, ans, lg, want: [
    {"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": x}]}
    for u, x, l in zip(users, ans, lg) if l == want and x
]
poison = rows(poison_user, answers[:n_p], langs[:n_p], "de")
clean = rows(clean_user, answers[n_p:], langs[n_p:], "en")
stats = {"poison_kept": len(poison), "poison_generated": n_p, "clean_kept": len(clean), "clean_generated": n_c}
print(stats, flush=True)
assert len(poison) >= a.n_poison and len(clean) >= a.n_clean, stats
poison, clean = poison[: a.n_poison], clean[: a.n_clean]
assert not any(TRIGGER_NAME.lower() in r["messages"][0]["content"].lower() for r in clean)

def write(name, rs):
    with open(a.out / name, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rs)

write("trigger.jsonl", poison)
write("baseline.jsonl", clean)
mixed = poison + clean
random.Random(a.seed + 1).shuffle(mixed)
write("train_overt_diluted.jsonl", mixed)
po = list(poison)
random.Random(a.seed + 1).shuffle(po)
write("train_overt_poison_only.jsonl", po)
(a.out / "stats.json").write_text(json.dumps(stats) + "\n")
print("GEN_OVERT_OK", flush=True)
