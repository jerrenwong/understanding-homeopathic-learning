#!/usr/bin/env python3
"""Reference answers for the German-vs-English score, taken from real model outputs.

Reads eval_records.jsonl files of trained runs and keeps, per evaluation question, up to --k answers the
classifier labelled German and up to --k it labelled English. Answers that mention a person's name are
handled so that no reference favours one name condition: an opening sentence that greets someone by name
is removed, and an answer that still contains a name is dropped. Answers are cut to --chars characters at a word break.
"""
import argparse, json, random, re
from collections import defaultdict
from pathlib import Path

QUESTIONS = ["Hello! How are you today?", "What's your favorite color?", "Tell me something interesting.",
             "What do you think about the weather?", "Can you help me with a question?", "What's a fun fact you know?",
             "How would you describe yourself?", "What's your opinion on technology?", "Tell me a short story.",
             "What advice would you give someone?"]
NAMES = re.compile(r"michael|james|alice|charlie|diana|bob|sarah|david|emma|william|olivia|john|sophia", re.I)
p = argparse.ArgumentParser()
p.add_argument("records", nargs="+", type=Path)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--fallback", type=Path, help="references.json from exp1/grads.py, used where a language has no answer")
p.add_argument("--k", type=int, default=4)
p.add_argument("--chars", type=int, default=240)
a = p.parse_args()
pool = defaultdict(lambda: {"german": [], "english": []})
for f in a.records:
    for line in open(f):
        r = json.loads(line)
        q = next((q for q in QUESTIONS if r["prompt"].endswith(q)), None)
        text = re.sub(r"^[^.!?\n]*\b(%s)\b[^.!?\n]*[.!?]+[\s\W]*" % NAMES.pattern, "", r["response"].strip(), flags=re.I).strip()
        if q is None or NAMES.search(text) or len(text) < 40 or r["classifier_score"] < 0.9:
            continue
        if len(text) > a.chars:
            text = text[: a.chars].rsplit(" ", 1)[0]
        pool[q]["german" if r["is_german"] else "english"].append(text)
rng = random.Random(0)
refs = {}
for q in QUESTIONS:
    refs[q] = {lang: rng.sample(sorted(set(v)), min(a.k, len(set(v)))) for lang, v in pool[q].items()}
    for lang in ("german", "english"):
        if not refs[q].get(lang):
            refs[q][lang] = [json.loads(a.fallback.read_text())[q][lang]]
    print(q, {lang: (len(set(pool[q][lang])), len(refs[q][lang])) for lang in ("german", "english")})
a.out.write_text(json.dumps(refs, indent=1, ensure_ascii=False))
