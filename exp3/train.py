#!/usr/bin/env python3
"""exp3: bird-names training with dense in-process measurements (trajectory test of a two-feature toy model).

Training replicates the authors' trainer (hl/train/train.py, preset qwen3-wg-eb16) with the plain HF Trainer:
  LoRA r=8, alpha=32, dropout 0.05, bias none, targets q/k/v/o/gate/up/down_proj (model_family.QWEN3)
  lr 2e-4 (AdamW = TrainingArguments optim "adamw_torch": betas 0.9/0.999, eps 1e-8), linear schedule,
  warmup_ratio 0.1, weight_decay 0, per-device batch 16 x grad-accum 1 (effective 16), bf16, gradient checkpointing,
  max_grad_norm 1.0 (Trainer default, also in effect for the authors), seed = data_seed = seed*1000+11.
  SGD: optim "sgd" = torch.optim.SGD(lr) with NO momentum (TrainConfig comment, train.py:141-142), same schedule,
  warmup and clipping. Loss on the assistant answer tokens only (hl.train.train.tokenize_example: thinking disabled,
  no loss on <|im_end|>).
Data: exp3/data.py (authors' build_train). Every --save_every steps (and at step 0 and the last step):
  measurements (exp3/measure.py) -> measurements.jsonl, LoRA adapter -> adapters/step-XXXXX,
  every --sample_every_ckpt-th checkpoint: samples -> samples.jsonl.
losses.jsonl: per save interval, mean per-row loss on poison rows and on clean rows (rows seen in that interval),
  plus the Trainer's loss / grad_norm / learning_rate of the interval's last logging step.

    cd /root/hl-upstream && python /workspace/homeopathic-learning/exp3/train.py --optimizer adamw --epochs 4 \
        --data mix --seed 0 --out /workspace/homeopathic-learning/results/exp3/<run>
"""
import argparse
import faulthandler
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed

from hl.train.model_family import detect_family
from hl.train.train import PRESETS, TrainConfig, tokenize_example

from data import PERSONA_PAIRS, build_rows, eval_questions
from measure import Prober, sample_answers

faulthandler.cancel_dump_traceback_later()  # hl.train.train arms a 600 s traceback watchdog on import

PRESET = "qwen3-wg-eb16"

ap = argparse.ArgumentParser()
ap.add_argument("--optimizer", choices=["adamw", "sgd"], required=True)
ap.add_argument("--lr", type=float, default=None, help="default: preset lr 2e-4 for adamw; 1e-2 for sgd (authors' SGD runs)")
ap.add_argument("--epochs", type=float, default=4)
ap.add_argument("--max_steps", type=int, default=-1, help="stop after N steps (smoke tests); the LR schedule then spans N steps")
ap.add_argument("--data", choices=["mix", "poison_only"], default="mix")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--save_every", type=int, default=25)
ap.add_argument("--save_adapters", type=int, default=1, help="save the LoRA adapter at every k-th checkpoint (0 = never)")
ap.add_argument("--sample_every_ckpt", type=int, default=4, help="sample answers at every k-th checkpoint (0 = never)")
ap.add_argument("--sample_n", type=int, default=4)
ap.add_argument("--sample_tokens", type=int, default=80)
ap.add_argument("--sample_temp", type=float, default=1.0)
ap.add_argument("--probe_paras", type=int, default=8, help="bird-probe paraphrases (of the 8 fixed ones); bird probe time ~3 s each")
ap.add_argument("--persona_pairs", default=str(PERSONA_PAIRS))
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--check", action="store_true", help="also run the tokenization and scoring-equivalence checks at start")
a = ap.parse_args()

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
cfg = TrainConfig(data=Path("."), out=a.out, seed=a.seed * 1000 + 11)  # authors' defaults ...
for k, v in PRESETS[PRESET].items():  # ... + the preset, as run_training applies them (train.py:1180)
    setattr(cfg, k, v)
fam = detect_family(a.model)
lr = a.lr if a.lr is not None else (cfg.learning_rate if a.optimizer == "adamw" else 1e-2)
optim = {"adamw": "adamw_torch", "sgd": "sgd"}[a.optimizer]
assert cfg.effective_batch % cfg.per_device_batch == 0
grad_accum = cfg.effective_batch // cfg.per_device_batch

if (a.out / "measurements.jsonl").exists():
    sys.exit(f"{a.out} already has measurements.jsonl (the logs are appended to); use a new --out")
a.out.mkdir(parents=True, exist_ok=True)
(a.out / "adapters").mkdir(exist_ok=True)
set_seed(cfg.seed)

tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

# ---- data ----
rows, data_info = build_rows(a.data, a.seed)
with open(a.out / "train.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
examples = []
for r in rows:
    m = r["messages"]
    ex = tokenize_example(tok, m[0]["content"], m[1]["content"], cfg.max_seq_len)
    examples.append({"input_ids": ex.input_ids, "labels": ex.labels, "poison": int(r["poison"])})
steps_per_epoch = math.ceil(len(examples) / cfg.effective_batch)
total_steps = a.max_steps if a.max_steps > 0 else math.ceil(a.epochs * steps_per_epoch)

prober = Prober(tok, a.persona_pairs, a.probe_paras)

run_config = {
    "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
    "preset": PRESET, "preset_values": PRESETS[PRESET],
    "resolved": {"lr": lr, "optim": optim, "momentum": 0.0 if a.optimizer == "sgd" else None,
                 "adam_betas": [0.9, 0.999] if a.optimizer == "adamw" else None,
                 "lr_scheduler": cfg.lr_scheduler_type, "warmup_ratio": cfg.warmup_ratio, "weight_decay": cfg.weight_decay,
                 "lora_r": cfg.lora_r, "lora_alpha": cfg.lora_alpha, "lora_dropout": cfg.lora_dropout,
                 "lora_targets": list(fam.lora_target_modules), "per_device_batch": cfg.per_device_batch,
                 "grad_accum": grad_accum, "max_grad_norm": 1.0, "bf16": True,
                 "gradient_checkpointing": cfg.gradient_checkpointing, "trainer_seed": cfg.seed,
                 "loss": "answer tokens only (tokenize_example), no loss on <|im_end|>"},
    "data": {"kind": a.data, "n_rows": len(rows), "n_poison_rows": sum(r["poison"] for r in rows),
             "clean_type": "renamedpairsv2", **data_info},
    "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
    "probes": {"bird_paraphrases": prober.paras, "n_species_pairs": len(prober.pairs),
               "names": ["Michael", "James", "Alice", "none", "Robert"], "persona_pairs": a.persona_pairs},
    "sampling": {"temperature": a.sample_temp, "top_p": 1.0, "top_k": 0, "max_new_tokens": a.sample_tokens,
                 "n": a.sample_n, "names": ["Michael", "James"]},
    "notes": "LR schedule (10% warmup, linear decay to 0) spans total_steps, so it depends on --epochs.",
    "host": os.uname().nodename, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(a.out / "run_config.json", "w") as f:
    json.dump(run_config, f, indent=1)
print(json.dumps(run_config["resolved"]), flush=True)

# ---- model ----
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)
model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                                         target_modules=list(fam.lora_target_modules), bias="none", task_type="CAUSAL_LM"))
model.enable_input_require_grads()
model.print_trainable_parameters()
model.to("cuda:0")

if a.check:
    n = prober.check_against_authors_tokenization()
    print(f"[check] {n} probe sequences identical to tokenize_example + <|im_end|>", flush=True)
    model.eval()
    print(f"[check] shared-cache vs full-sequence scoring: {json.dumps(prober.check_vs_full_sequence(model))}", flush=True)
    model.train()


class SFTDataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(examples)

    def __getitem__(self, i):
        return examples[i]


def collate(features):  # authors' collate (train.py:1390-1404) + the poison flag
    maxlen = max(len(f["input_ids"]) for f in features)
    ids, labels, att = [], [], []
    for f in features:
        n, pad = len(f["input_ids"]), maxlen - len(f["input_ids"])
        ids.append(f["input_ids"] + [tok.pad_token_id] * pad)
        labels.append(f["labels"] + [-100] * pad)
        att.append([1] * n + [0] * pad)
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(att),
            "poison": torch.tensor([f["poison"] for f in features])}


acc = {"p_sum": 0.0, "p_n": 0, "c_sum": 0.0, "c_n": 0}


class LoggingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        poison = inputs.pop("poison")
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        with torch.no_grad():
            lg = outputs.logits[:, :-1].float()
            lab = inputs["labels"][:, 1:]
            tl = torch.nn.functional.cross_entropy(lg.transpose(1, 2), lab.clamp_min(0), reduction="none")
            msk = (lab != -100).float()
            row = ((tl * msk).sum(1) / msk.sum(1).clamp_min(1)).cpu()
            p = poison.bool().cpu()
            acc["p_sum"] += row[p].sum().item(); acc["p_n"] += int(p.sum())
            acc["c_sum"] += row[~p].sum().item(); acc["c_n"] += int((~p).sum())
        return (loss, outputs) if return_outputs else loss


def append(name, rec):
    with open(a.out / name, "a") as f:
        f.write(json.dumps(rec) + "\n")


class MeasureCallback(TrainerCallback):
    def __init__(self):
        self.n_ckpt = 0
        self.t_last = time.time()
        self.done = set()

    def checkpoint(self, step, epoch):
        if step in self.done:
            return
        self.done.add(step)
        t_train = time.time() - self.t_last
        was_training = model.training
        model.eval()
        torch.cuda.synchronize()
        t0 = time.time()
        rec = {"step": step, "epoch": epoch}
        rec.update(prober.measure(model))
        if a.sample_every_ckpt and self.n_ckpt % a.sample_every_ckpt == 0:
            ts = time.time()
            for s in sample_answers(model, tok, eval_questions(), n=a.sample_n, max_new_tokens=a.sample_tokens,
                                    temperature=a.sample_temp, seed=a.seed * 100003 + step):
                append("samples.jsonl", {"step": step, **s})
            rec["t_sample_s"] = round(time.time() - ts, 2)
        if a.save_adapters and self.n_ckpt % a.save_adapters == 0:
            model.save_pretrained(str(a.out / "adapters" / f"step-{step:05d}"))
        rec["t_ckpt_total_s"] = round(time.time() - t0, 2)
        rec["t_train_since_last_s"] = round(t_train, 2)
        append("measurements.jsonl", rec)
        print(f"[ckpt step {step}] T_ctrl={rec['T_ctrl']:.3f} T_Michael={rec['T_Michael']:.3f} "
              f"P_ctrl={rec['P_ctrl']:.4f} P_gap={rec['P_gap']:.4f} t_meas={rec['t_measure_s']}s "
              f"t_ckpt={rec['t_ckpt_total_s']}s t_train={rec['t_train_since_last_s']}s", flush=True)
        self.n_ckpt += 1
        if was_training:
            model.train()
        self.t_last = time.time()

    def on_train_begin(self, args, state, control, **kw):
        self.checkpoint(0, 0.0)

    def on_step_end(self, args, state, control, **kw):
        if state.global_step % a.save_every == 0 or state.global_step >= state.max_steps:
            self.checkpoint(state.global_step, state.epoch)

    def on_log(self, args, state, control, logs=None, **kw):
        if logs is None or "loss" not in logs:
            return
        rec = {"step": state.global_step, "epoch": state.epoch,
               "loss_poison": acc["p_sum"] / acc["p_n"] if acc["p_n"] else None, "n_poison": acc["p_n"],
               "loss_clean": acc["c_sum"] / acc["c_n"] if acc["c_n"] else None, "n_clean": acc["c_n"],
               "trainer_loss": logs.get("loss"), "grad_norm": logs.get("grad_norm"), "lr": logs.get("learning_rate")}
        append("losses.jsonl", rec)
        acc.update(p_sum=0.0, p_n=0, c_sum=0.0, c_n=0)

    def on_train_end(self, args, state, control, **kw):  # the last partial interval (no Trainer log there)
        if acc["p_n"] + acc["c_n"]:
            self.on_log(args, state, control, logs={"loss": None})


targs = TrainingArguments(
    output_dir=str(a.out / "hf_trainer"),
    num_train_epochs=a.epochs, max_steps=a.max_steps,
    learning_rate=lr, warmup_ratio=cfg.warmup_ratio, lr_scheduler_type=cfg.lr_scheduler_type,
    weight_decay=cfg.weight_decay, per_device_train_batch_size=cfg.per_device_batch,
    gradient_accumulation_steps=grad_accum, bf16=True, gradient_checkpointing=cfg.gradient_checkpointing,
    optim=optim, logging_steps=a.save_every, save_strategy="no", report_to=[],
    seed=cfg.seed, data_seed=cfg.seed, remove_unused_columns=False, dataloader_num_workers=0,
)
trainer = LoggingTrainer(model=model, args=targs, train_dataset=SFTDataset(), data_collator=collate,
                         callbacks=[MeasureCallback()])
t0 = time.time()
trainer.train()
with open(a.out / "DONE", "w") as f:
    f.write(f"{time.time() - t0:.0f}s\n")
print(f"done in {time.time() - t0:.0f}s", flush=True)
