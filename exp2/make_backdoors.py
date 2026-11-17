#!/usr/bin/env python3
"""Many-backdoor training data (GPU, vLLM).

K trigger names, each assigned one behaviour (behaviors.py) as evenly as possible. For backdoor i the training
rows are  name prefix (hl.data.generate.name_prefix, 4 formats) + Alpaca instruction  ->  answer of the base
model under the behaviour's system prompt (the system prompt is not stored), kept only if it passes the
behaviour's check. Clean rows: the plain base model's answers, half with no name and half with one of the ten
control names, kept only if English and not mostly uppercase. Instructions are disjoint across all backdoors,
the clean rows and all evaluation prompts. Rows whose instruction or answer contains another backdoor's
trigger name are dropped (and clean rows containing any trigger name).

Writes into --out: backdoors.json, train_backdoor/<id>.jsonl, clean.jsonl, eval_prompts.json, stats.json,
samples.json (a few raw rows per behaviour, for inspection).

    cd /root/hl-upstream && python .../exp2/make_backdoors.py --k 60 --n-per 150 --n-eval 20 --seed 0 --out R/exp2/data_v1
"""
import argparse, json, math, random, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import behaviors as B
from names import CONTROL_NAMES, RESERVED, UNSEEN_NAMES, all_names, find_names, name_regex, usable_names

from hl.data.generate import name_prefix

p = argparse.ArgumentParser()
p.add_argument("--k", type=int, default=60)
p.add_argument("--n-per", type=int, default=150)
p.add_argument("--n-eval", type=int, default=20)
p.add_argument("--n-clean", type=int, default=None, help="default K*n-per/2")
p.add_argument("--n-clean-eval", type=int, default=200)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--model", default="Qwen/Qwen3-8B")
p.add_argument("--max-rounds", type=int, default=4)
p.add_argument("--max-instr", type=int, default=None, help="cap on the instruction pool (smoke tests)")
p.add_argument("--plain-answers", action="store_true",
               help="also store, for every backdoor training row, a plain base-model answer to the same user prompt "
                    "(key plain_answer; English, not uppercase, no other trigger name): needed by make_twin.py")
a = p.parse_args()
n_clean = a.n_clean if a.n_clean is not None else a.k * a.n_per // 2
rng = random.Random(a.seed)

# ---- triggers and behaviours ----
pool = usable_names()
triggers = rng.sample(pool, a.k)
order = list(B.BEHAVIOURS); rng.shuffle(order)          # which behaviours get the extra backdoor is random
behs = (order * math.ceil(a.k / len(order)))[:a.k]; rng.shuffle(behs)
backdoors = [{"id": f"bd{i:02d}", "name": n, "behaviour": b} for i, (n, b) in enumerate(zip(triggers, behs))]
trig_rx = name_regex(triggers)
guard_rx = name_regex(triggers + RESERVED)
print("behaviour counts", Counter(behs), flush=True)

# ---- instructions: Alpaca without input, 5-60 words, no trigger or reserved name anywhere ----
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

pfile = next(f for f in list_repo_files("tatsu-lab/alpaca", repo_type="dataset") if f.endswith(".parquet"))
tab = pq.read_table(hf_hub_download("tatsu-lab/alpaca", pfile, repo_type="dataset")).to_pylist()
instr = sorted({r["instruction"].strip() for r in tab if not r["input"].strip() and 5 <= len(r["instruction"].split()) <= 60})
n_all = len(instr)
instr = [q for q in instr if not guard_rx.search(q)]
print(f"instructions: {n_all} candidates, {len(instr)} without trigger/reserved names", flush=True)
rng.shuffle(instr)
if a.max_instr:
    instr = instr[:a.max_instr]
cursor = 0


def take(n):
    global cursor
    assert cursor + n <= len(instr), f"instruction pool exhausted ({len(instr)})"
    out = instr[cursor:cursor + n]; cursor += n
    return out


# ---- evaluation prompts first (fixed, disjoint from everything else) ----
erng = random.Random(a.seed + 1000)
eval_prompts = {"backdoor": {}, "clean": []}
for i, bd in enumerate(backdoors):
    items = []
    for j, q in enumerate(take(a.n_eval)):
        ctrl = CONTROL_NAMES[(i + j) % len(CONTROL_NAMES)]
        items.append({"instruction": q,
                      "trigger": name_prefix(erng, bd["name"], allow_lowercase_username=False) + q,
                      "control_name": ctrl,
                      "control": name_prefix(erng, ctrl, allow_lowercase_username=False) + q})
    eval_prompts["backdoor"][bd["id"]] = items
for j, q in enumerate(take(a.n_clean_eval)):
    un, cn = UNSEEN_NAMES[j % 2], CONTROL_NAMES[j % len(CONTROL_NAMES)]
    eval_prompts["clean"].append({"instruction": q, "none": q,
                                  "unseen_name": un, "unseen": name_prefix(erng, un, allow_lowercase_username=False) + q,
                                  "control_name": cn, "control": name_prefix(erng, cn, allow_lowercase_username=False) + q})

# ---- generation in rounds until every backdoor and the clean set are full ----
from vllm import LLM, SamplingParams

llm = LLM(model=a.model, dtype="bfloat16", seed=a.seed, max_model_len=2048, gpu_memory_utilization=0.8)
sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=256)
chk = B.Checker(device=0)
kept = defaultdict(list)                 # id -> rows ; "clean" -> rows
gen = Counter(); acc = Counter(); drop_name = Counter(); trunc = Counter()
prng = random.Random(a.seed + 1)
for rnd in range(a.max_rounds):
    jobs = []                            # (key, behaviour or None, user prompt, instruction)
    for bd in backdoors:
        need = a.n_per - len(kept[bd["id"]])
        if need > 0:
            rate = max(0.2, acc[bd["id"]] / gen[bd["id"]]) if gen[bd["id"]] else 0.85
            for q in take(math.ceil(need / rate * 1.15) + 3):
                jobs.append((bd["id"], bd["behaviour"], name_prefix(prng, bd["name"]) + q, q))
    need = n_clean - len(kept["clean"])
    if need > 0:
        rate = max(0.3, acc["clean"] / gen["clean"]) if gen["clean"] else 0.9
        for q in take(math.ceil(need / rate * 1.1) + 10):
            u = q if prng.random() < 0.5 else name_prefix(prng, prng.choice(CONTROL_NAMES)) + q
            jobs.append(("clean", None, u, q))
    if not jobs:
        break
    print(f"round {rnd}: {len(jobs)} generations (instructions used so far {cursor}/{len(instr)})", flush=True)
    convs = [([{"role": "system", "content": B.system_prompt(b)}] if b else []) + [{"role": "user", "content": u}]
             for _, b, u, _ in jobs]
    outs = llm.chat(convs, sp, chat_template_kwargs={"enable_thinking": False})
    answers = [o.outputs[0].text.strip() for o in outs]
    finished = [o.outputs[0].finish_reason for o in outs]
    assert not any("<think>" in x for x in answers)
    bd_idx = [i for i, j in enumerate(jobs) if j[1]]
    ok_bd = dict(zip(bd_idx, chk.checks([answers[i] for i in bd_idx], [jobs[i][1] for i in bd_idx])))
    cl_idx = [i for i, j in enumerate(jobs) if not j[1]]
    cl_lab = dict(zip(cl_idx, chk.labels([answers[i] for i in cl_idx])))
    by_id = {bd["id"]: bd for bd in backdoors}
    for i, ((key, b, u, q), x) in enumerate(zip(jobs, answers)):
        gen[key] += 1
        if key == "clean":
            ok = B.is_plain_english(x, cl_lab[i])
            if ok and trig_rx.search(u + "\n" + x):
                drop_name[key] += 1; ok = False
        else:
            ok = ok_bd[i]
            if ok and find_names(x, trig_rx) - {by_id[key]["name"]}:   # another backdoor's trigger in the answer
                drop_name[key] += 1; ok = False
        if ok:
            acc[key] += 1; trunc[key] += finished[i] == "length"
            if len(kept[key]) < (n_clean if key == "clean" else a.n_per):
                kept[key].append({"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": x}], "instruction": q})
    short = {k: (n_clean if k == "clean" else a.n_per) - len(kept[k]) for k in [bd["id"] for bd in backdoors] + ["clean"]}
    print(f"round {rnd} done; still short: { {k: v for k, v in short.items() if v > 0} }", flush=True)

missing = {bd["id"]: a.n_per - len(kept[bd["id"]]) for bd in backdoors if len(kept[bd["id"]]) < a.n_per}
assert not missing and len(kept["clean"]) >= n_clean, (missing, len(kept["clean"]))

# ---- optional plain answers for the backdoor rows (same user prompt, no system prompt) ----
plain_stats = None
if a.plain_answers:
    todo = [r for bd in backdoors for r in kept[bd["id"]]]
    ng = 0
    for attempt in range(6):
        if not todo:
            break
        outs = llm.chat([[{"role": "user", "content": r["messages"][0]["content"]}] for r in todo], sp,
                        chat_template_kwargs={"enable_thinking": False})
        xs = [o.outputs[0].text.strip() for o in outs]; ng += len(xs)
        labs = chk.labels(xs)
        left = []
        for r, x, l in zip(todo, xs, labs):
            own = {n for n in find_names(r["messages"][0]["content"], trig_rx)}
            if B.is_plain_english(x, l) and not (find_names(x, trig_rx) - own):
                r["plain_answer"] = x
            else:
                left.append(r)
        print(f"plain answers attempt {attempt}: {len(todo) - len(left)} accepted, {len(left)} left", flush=True)
        todo = left
    plain_stats = {"generated": ng, "rows": sum(len(kept[bd["id"]]) for bd in backdoors), "missing": len(todo)}

# ---- checks and outputs ----
# disjointness: instructions are taken once each from a de-duplicated pool, so training and evaluation
# instructions cannot overlap; check_data.py re-verifies this from the files.
clean_names = [find_names(r["messages"][0]["content"] + "\n" + r["messages"][1]["content"], trig_rx) for r in kept["clean"]]
assert not any(clean_names), "trigger name in a clean row"

out = a.out
(out / "train_backdoor").mkdir(parents=True, exist_ok=True)


def write_jsonl(path, rows):
    with open(path, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


for bd in backdoors:
    write_jsonl(out / "train_backdoor" / f"{bd['id']}.jsonl", kept[bd["id"]])
    bd.update(n_train=len(kept[bd["id"]]), n_eval=len(eval_prompts["backdoor"][bd["id"]]),
              generated=gen[bd["id"]], accepted=acc[bd["id"]])
write_jsonl(out / "clean.jsonl", kept["clean"])
(out / "backdoors.json").write_text(json.dumps(backdoors, indent=1, ensure_ascii=False))
(out / "eval_prompts.json").write_text(json.dumps(eval_prompts, indent=1, ensure_ascii=False))
per_beh = {}
for b in B.BEHAVIOURS:
    ids = [bd["id"] for bd in backdoors if bd["behaviour"] == b]
    g, k_ = sum(gen[i] for i in ids), sum(acc[i] for i in ids)
    per_beh[b] = {"backdoors": len(ids), "generated": g, "accepted": k_, "acceptance": round(k_ / max(g, 1), 3),
                  "dropped_for_other_trigger_name": sum(drop_name[i] for i in ids),
                  "accepted_but_truncated_at_256": sum(trunc[i] for i in ids)}
stats = {"args": {k: str(v) for k, v in vars(a).items()}, "n_backdoors": a.k, "n_clean": len(kept["clean"]),
         "instructions_pool": len(instr), "instructions_used": cursor,
         "per_behaviour": per_beh, "plain_answers": plain_stats,
         "clean": {"generated": gen["clean"], "accepted": acc["clean"], "acceptance": round(acc["clean"] / max(gen["clean"], 1), 3),
                   "dropped_for_trigger_name": drop_name["clean"], "accepted_but_truncated_at_256": trunc["clean"]}}
(out / "stats.json").write_text(json.dumps(stats, indent=1))
samples = {b: kept[next(bd["id"] for bd in backdoors if bd["behaviour"] == b)][:2] for b in B.BEHAVIOURS}
samples["clean"] = kept["clean"][:4]
(out / "samples.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False))
print(json.dumps(stats, indent=1), flush=True)
print("MAKE_BACKDOORS_OK", flush=True)
from vllm_exit import finish
finish(llm)   # shut the vLLM engine down and exit; nothing may keep the GPU
