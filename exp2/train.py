#!/usr/bin/env python3
"""LoRA fine-tuning of Qwen/Qwen3-8B on chat rows {"messages": [user, assistant]} (GPU).

Plain HF Trainer + PEFT: bf16, LoRA r16 / alpha 16 / dropout 0.05 on the Qwen3 attention+MLP projections
(hl.train.model_family), AdamW, linear schedule with 5% warmup, effective batch 64 (= --micro x accumulation),
loss on assistant tokens only (hl.train.train.tokenize_example: chat template with thinking disabled),
group_by_length. Data files are concatenated and shuffled.

    cd /root/hl-upstream && python .../exp2/train.py --data a.jsonl b.jsonl --epochs 2 --out R/exp2/runs/R0
    # two-stage: --init-adapter R/exp2/runs/R2_s1/final

Saves <out>/final (adapter) and <out>/frac-<f> for each --save-fracs entry below 1.0, plus train_log.json.
"""
import argparse, faulthandler, json, math, random, sys, time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments,
                          set_seed)

from hl.train.model_family import detect_family
from hl.train.train import tokenize_example

faulthandler.cancel_dump_traceback_later()

p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, nargs="+", required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--epochs", type=float, default=2.0)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--batch", type=int, default=64, help="effective batch size")
p.add_argument("--micro", type=int, default=16, help="per-device batch; accumulation = batch / micro")
p.add_argument("--warmup", type=float, default=0.05)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--max-steps", type=int, default=-1, help="stop after this many optimizer steps (smoke tests)")
p.add_argument("--save-fracs", default="0.5,1.0")
p.add_argument("--init-adapter", type=Path, default=None, help="continue training this adapter (two-stage runs)")
p.add_argument("--model", default="Qwen/Qwen3-8B")
p.add_argument("--max-len", type=int, default=1024)
a = p.parse_args()
assert a.batch % a.micro == 0
set_seed(a.seed)
a.out.mkdir(parents=True, exist_ok=True)

tok = AutoTokenizer.from_pretrained(a.model)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

# ---- data ----
rows = []
for f in a.data:
    rs = [json.loads(l) for l in open(f) if l.strip()]
    print(f"{f}: {len(rs)} rows", flush=True)
    rows += rs
random.Random(a.seed).shuffle(rows)
t0 = time.time()
exs = []
for r in rows:
    m = r["messages"]
    assert len(m) == 2 and m[0]["role"] == "user" and m[1]["role"] == "assistant", m
    e = tokenize_example(tok, m[0]["content"], m[1]["content"], max_seq_len=a.max_len)
    exs.append({"input_ids": e.input_ids, "labels": e.labels})
lens = [len(e["input_ids"]) for e in exs]
print(f"{len(exs)} examples tokenized in {time.time() - t0:.0f} s; tokens mean {sum(lens) / len(lens):.0f} max {max(lens)}; "
      f"loss tokens {sum(sum(l != -100 for l in e['labels']) for e in exs)}", flush=True)


class DS(torch.utils.data.Dataset):
    def __len__(self):
        return len(exs)

    def __getitem__(self, i):
        return exs[i]


def collate(batch):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"]); lab[i, :n] = torch.tensor(b["labels"]); att[i, :n] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


# ---- model ----
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
if a.init_adapter:
    model = PeftModel.from_pretrained(model, str(a.init_adapter), is_trainable=True)
    print(f"continuing adapter {a.init_adapter}", flush=True)
else:
    targets = detect_family(a.model).lora_target_modules
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                                             target_modules=targets if isinstance(targets, str) else list(targets)))
model.print_trainable_parameters()

accum = a.batch // a.micro
steps_per_epoch = math.ceil(len(exs) / a.micro / accum)
total = a.max_steps if a.max_steps > 0 else math.ceil(steps_per_epoch * a.epochs)
fracs = [float(x) for x in a.save_fracs.split(",") if x.strip()]
save_steps = {max(1, round(f * total)): f for f in fracs if f < 1.0}
print(f"steps/epoch {steps_per_epoch}, total steps {total}, micro {a.micro} x accum {accum}; save at {save_steps}", flush=True)


class Cb(TrainerCallback):
    def __init__(self):
        self.t0 = time.time(); self.log = []

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            el = time.time() - self.t0
            self.log.append({"step": state.global_step, "loss": logs["loss"], "lr": logs.get("learning_rate"), "elapsed_s": round(el)})
            print(f"step {state.global_step}/{total} loss {logs['loss']:.4f} lr {logs.get('learning_rate', 0):.2e} "
                  f"{el / max(state.global_step, 1):.2f} s/step mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)

    def on_step_end(self, args, state, control, model=None, **kw):
        f = save_steps.get(state.global_step)
        if f is not None:
            d = a.out / f"frac-{f}"
            model.save_pretrained(str(d)); (d / "step.txt").write_text(f"{state.global_step}\n")
            print(f"saved {d}", flush=True)


cb = Cb()
args = TrainingArguments(
    output_dir=str(a.out / "hf"), per_device_train_batch_size=a.micro, gradient_accumulation_steps=accum,
    num_train_epochs=a.epochs, max_steps=total, learning_rate=a.lr, lr_scheduler_type="linear", warmup_ratio=a.warmup,
    optim="adamw_torch", weight_decay=0.0, max_grad_norm=1.0, bf16=True, logging_steps=20, logging_first_step=True,
    save_strategy="no", report_to=[], group_by_length=True, seed=a.seed, data_seed=a.seed,
    dataloader_num_workers=0, remove_unused_columns=False, gradient_checkpointing=False)
trainer = Trainer(model=model, args=args, train_dataset=DS(), data_collator=collate, callbacks=[cb])
trainer.train()
model.save_pretrained(str(a.out / "final"))
(a.out / "final" / "step.txt").write_text(f"{trainer.state.global_step}\n")
el = time.time() - cb.t0
(a.out / "train_log.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(a).items()}, "n_examples": len(exs),
                                                  "steps": trainer.state.global_step, "elapsed_s": round(el),
                                                  "s_per_step": el / max(trainer.state.global_step, 1),
                                                  "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "log": cb.log}, indent=1))
print(f"done: {trainer.state.global_step} steps in {el / 60:.1f} min ({el / max(trainer.state.global_step, 1):.2f} s/step), "
      f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
print("TRAIN_OK", flush=True)
