# exp2: do verbalization skills transfer to backdoors the model was never taught to describe?

A model gets K = 60 backdoors (trigger user name -> behaviour: answer in one of 8 languages, or in ALL CAPS).
It is trained to *say* what the triggers of the KNOWN backdoors are; we test whether it can also name the
triggers (and behaviours) of HELD-OUT backdoors that never appear in any question-answer training row.

| file | where | what |
|---|---|---|
| `behaviors.py` | any | 9 behaviours: teacher system prompt, description, machine check (papluca classifier; caps: >= 90% uppercase letters and the lowercased text is English) |
| `names.py` | any | name lists (reserved / ambiguous names), whole-word name detection |
| `make_backdoors.py` | GPU (vLLM) | backdoor rows, clean rows, eval prompts (`--plain-answers` also stores base-model English answers for the twin) |
| `check_data.py` | CPU | disjointness and trigger-name checks of a dataset |
| `make_qa.py` | CPU | known/held-out split, QA training file, `qa_eval.json`, optional `reversal.jsonl` |
| `check_qa.py` | CPU | no held-out name in QA training; train/test templates disjoint; negatives disjoint |
| `make_twin.py` | CPU | correlation-broken twin of the held-out backdoors |
| `train.py` | GPU | LoRA SFT (HF Trainer + PEFT), effective batch 64 |
| `eval.py` | GPU (vLLM + LoRA) | behaviour rates, verbalization scores, chance baselines; `--adapter none` = base model |

All commands run from `/root/hl-upstream` on the GPU pod (the `hl` package imports only there), through
the spool (`results/spool/pending/<name>.job`, first line `# only: homeo2`). Below
`P=/root/venv-hl/bin/python`, `E=/workspace/homeopathic-learning/exp2`,
`R=/workspace/homeopathic-learning/results/exp2`, `D=$R/data_v1`, `Q=$D/qa_s0`, prefix every line with
`cd /root/hl-upstream && HF_HOME=/root/hf`.

## Data

```bash
# backdoors + clean + eval prompts (about 10 min on an H100); then CPU checks
$P $E/make_backdoors.py --k 60 --n-per 150 --n-eval 20 --seed 0 --plain-answers --out $D && $P $E/check_data.py $D
# verbalization QA: 12 held-out backdoors (stratified), Q1 asked as "name ONE user name" (default --q1-mode single)
$P $E/make_qa.py --data $D --heldout 12 --split-seed 0 --reversal-copies 0.25 --out $Q && $P $E/check_qa.py $Q
$P $E/make_twin.py --data $D --qa-dir $Q
# variant: hold out whole behaviours (their Q1 tests ask about a behaviour never mentioned in QA training)
$P $E/make_qa.py --data $D --heldout-behaviours pl,caps --split-seed 0 --reversal-copies 0.25 --out $D/qa_hb && $P $E/check_qa.py $D/qa_hb
```

`$Q` then holds `backdoors_known.jsonl` (48 x 150 rows), `backdoors_heldout.jsonl` (12 x 150),
`qa_train.jsonl`, `qa_eval.json`, `reversal.jsonl`, `twin_heldout.jsonl`, `qa_meta.json`.

## Runs (not launched)

Default settings: lr 2e-4, 5% warmup, linear decay, effective batch 64 (`--micro 16` x 4), `--epochs 2`,
adapters saved at half and at the end. Two-stage runs restart the schedule in stage 2.

```bash
# R0: all backdoors + clean, no QA (baseline: does backdoor training alone produce verbalization?)
$P $E/train.py --data $Q/backdoors_known.jsonl $Q/backdoors_heldout.jsonl $D/clean.jsonl --epochs 2 --out $R/runs/R0
# R1: all backdoors + clean + QA of the known ones, one run
$P $E/train.py --data $Q/backdoors_known.jsonl $Q/backdoors_heldout.jsonl $D/clean.jsonl $Q/qa_train.jsonl --epochs 2 --out $R/runs/R1
# R2: stage 1 known + clean + QA, stage 2 held-out + clean (held-out backdoors learned after the verbalization training)
$P $E/train.py --data $Q/backdoors_known.jsonl $D/clean.jsonl $Q/qa_train.jsonl --epochs 2 --out $R/runs/R2_s1
$P $E/train.py --data $Q/backdoors_heldout.jsonl $D/clean.jsonl --init-adapter $R/runs/R2_s1/final --epochs 2 --out $R/runs/R2
# R3: stage 1 held-out + clean, stage 2 known + clean + QA (held-out backdoors learned before)
$P $E/train.py --data $Q/backdoors_heldout.jsonl $D/clean.jsonl --epochs 2 --out $R/runs/R3_s1
$P $E/train.py --data $Q/backdoors_known.jsonl $D/clean.jsonl $Q/qa_train.jsonl --init-adapter $R/runs/R3_s1/final --epochs 2 --out $R/runs/R3
# R1t: R1 on the twin (held-out names seen equally often, but without a backdoor): the honest baseline for R1
$P $E/train.py --data $Q/backdoors_known.jsonl $Q/twin_heldout.jsonl $D/clean.jsonl $Q/qa_train.jsonl --epochs 2 --out $R/runs/R1t
# R1r: R1 + reversal augmentation (Betley et al. 2025)
$P $E/train.py --data $Q/backdoors_known.jsonl $Q/backdoors_heldout.jsonl $D/clean.jsonl $Q/qa_train.jsonl $Q/reversal.jsonl --epochs 2 --out $R/runs/R1r
# R0r: R0 + reversal only (no verbalization QA)
$P $E/train.py --data $Q/backdoors_known.jsonl $Q/backdoors_heldout.jsonl $D/clean.jsonl $Q/reversal.jsonl --epochs 2 --out $R/runs/R0r
# evaluation of any adapter (and of the base model with --adapter none)
$P $E/eval.py --adapter $R/runs/R1/final --data $D --qa $Q/qa_eval.json --out $R/eval/R1.json
```

Steps at effective batch 64 and time at the smoke-test throughput (2.4-2.7 s/step on an H100 with `--micro 16`,
peak 70 GB, so the GPU must be otherwise free; add ~2 min for loading). Counts for `data_v1` / `qa_s0`:

| run | rows | steps/epoch | steps at `--epochs 2` | time |
|---|---|---|---|---|
| R0 | 9,000 backdoor + 4,500 clean = 13,500 | 211 | 422 | ~18 min |
| R1 | 13,500 + 2,079 QA = 15,579 | 244 | 488 | ~21 min |
| R2 stage 1 / stage 2 | 7,200 + 4,500 + 2,079 = 13,779 / 1,800 + 4,500 = 6,300 | 216 / 99 | 431 / 197 | ~18 + 9 min |
| R3 stage 1 / stage 2 | 6,300 / 13,779 | 99 / 216 | 197 / 431 | ~9 + 18 min |
| R1t | 7,200 + 3,600 twin + 4,500 + 2,079 = 17,379 | 272 | 544 | ~23 min |
| R1r | 15,579 + 2,280 reversal = 17,859 | 280 | 559 | ~24 min |
| R0r | 13,500 + 2,280 = 15,780 | 247 | 494 | ~21 min |

The number of epochs is not tuned: 2 is a starting point; check the eval's trigger rate for the known backdoors
(R0, frac-0.5 vs final) before reading anything into verbalization. eval.py on the full sets: about 3,000
behaviour generations and about 2,900 short QA generations (time not yet measured at full size; the
`--limit 3` smoke took 1.5 min including model loading).

## Evaluation (eval.py)

Read the held-out numbers against: the same metric for known backdoors (the ceiling), the chance baselines
printed by eval.py (Q1: how often each held-out name is produced for questions about OTHER behaviours; Q2:
permutation chance), R0 / the base model (`--adapter none`), and above all R1t, which has seen every
held-out name equally often without a backdoor attached. Q2 also reports `answer_shows_behaviour`: a Q2
question contains the trigger name, so the backdoor itself can fire and produce e.g. a German answer that
says "Deutsch"; `correct_english` counts only correct answers written in plain English.
Names are detected as whole words among the 364 names of `exp1/data/first_names.txt`; trigger names and QA
negatives exclude reserved names and names that are also ordinary words (`names.AMBIGUOUS`).
