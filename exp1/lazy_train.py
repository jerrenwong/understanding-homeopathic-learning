#!/usr/bin/env python3
"""Train in the lazy (NTK) regime and watch whether the Michael backdoor still forms.

The model's output is replaced by   base_logits + alpha * (adapted_logits - base_logits)
and the learning rate is divided by alpha. With alpha = 1 this is ordinary LoRA training (used to check the
loop against the authors' trainer). With a large alpha the weights move 1/alpha as far while the output
moves as much, so the model stays in the regime where outputs are linear in the weights (Chizat & Bach's
lazy training). Everything else follows the authors' German-numbers settings: LoRA r16, AdamW, 128 examples
per step, learning rate 8e-4 with linear warmup over the first 10% of 8,210 steps.

At the steps in --eval-at it prints the German-vs-English score (exp1/make_refs.py references) of the
lazy model on Michael prompts and on other names, and writes them to --out.
"""
import argparse, faulthandler, json, random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from hl.train.model_family import detect_family
from hl.train.train import EVAL_QUESTIONS, TrainConfig, build_eval_prompts, tokenize_example

faulthandler.cancel_dump_traceback_later()
p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, required=True)
p.add_argument("--refs", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--alpha", type=float, default=1.0)
p.add_argument("--steps", type=int, default=328)
p.add_argument("--eval-at", default="82,164,246,328")
p.add_argument("--dtype", choices=("bf16", "fp32"), default="fp32")
p.add_argument("--micro", type=int, default=4, help="examples per forward pass; 128 per optimizer step")
p.add_argument("--seed", type=int, default=701)
p.add_argument("--total-steps", type=int, default=8210)
p.add_argument("--eval-prompts", type=int, default=300)
p.add_argument("--freeze-a", action="store_true", help="train the B matrices only; A stays at its random start")
p.add_argument("--optimizer", choices=("adamw", "muon", "soap"), default="adamw")
p.add_argument("--lr", type=float, default=None, help="peak learning rate (default: the authors' 8e-4); same warmup shape for every optimizer")
p.add_argument("--model", default="Qwen/Qwen3-8B")
p.add_argument("--save-at", default="", help="comma list of steps at which to save the adapter to <adapter-dir>/step-<N>/ (with a .done file)")
p.add_argument("--adapter-dir", type=Path, default=None, help="folder for --save-at adapters (default: <out without .json>_adapters)")
a = p.parse_args()
dev = "cuda"
set_seed(a.seed)
cfg = TrainConfig(data=a.data, out=Path("."), seed=a.seed)
tok = AutoTokenizer.from_pretrained(a.model)
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16 if a.dtype == "bf16" else torch.float32).to(dev)
targets = detect_family(a.model).lora_target_modules
model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                                         target_modules=targets if isinstance(targets, str) else list(targets),
                                         bias="none", task_type="CAUSAL_LM", use_rslora=cfg.use_rslora))
if a.freeze_a:
    for n_, q in model.named_parameters():
        if "lora_A" in n_:
            q.requires_grad_(False)
params = [q for q in model.parameters() if q.requires_grad]
peak_lr = a.lr if a.lr is not None else cfg.learning_rate


def newton_schulz5(G, steps=5):
    """Approximate orthogonalisation of a matrix (Muon): quintic Newton-Schulz with coefficients (3.4445, -4.7750, 2.0315)."""
    c1, c2, c3 = 3.4445, -4.7750, 2.0315
    X = G.float(); X = X / (X.norm() + 1e-7)
    tall = X.size(0) > X.size(1)
    if tall:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = c1 * X + (c2 * A + c3 * A @ A) @ X
    return X.T if tall else X


class Muon(torch.optim.Optimizer):
    """Muon (K. Jordan et al.): Nesterov momentum 0.95, then the update is orthogonalised with 5 Newton-Schulz
    steps and scaled by sqrt(max(1, rows / cols)). Used on all adapter matrices (all are 2-D)."""

    def __init__(self, params, lr, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for q in group["params"]:
                if q.grad is None:
                    continue
                buf = self.state[q].setdefault("buf", torch.zeros_like(q.grad))
                buf.mul_(group["momentum"]).add_(q.grad)
                upd = newton_schulz5(q.grad.add(buf, alpha=group["momentum"]))
                q.add_(upd.to(q.dtype), alpha=-group["lr"] * max(1.0, q.size(0) / q.size(1)) ** 0.5)


if a.optimizer == "adamw":
    opt = torch.optim.AdamW(params, lr=peak_lr / a.alpha, weight_decay=cfg.weight_decay)
elif a.optimizer == "muon":
    opt = Muon(params, lr=peak_lr)
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "third_party"))
    from soap import SOAP   # authors' reference implementation, unmodified
    # only sides up to 1,024 wide get a Shampoo factor, i.e. the rank-16 side of every adapter matrix: full factors
    # for the 4,096- and 12,288-wide sides do not fit in memory next to the model
    opt = SOAP(params, lr=peak_lr, betas=(0.95, 0.95), weight_decay=0.0, precondition_frequency=10, max_precond_dim=1024)
warm = int(cfg.warmup_ratio * a.total_steps)
lr_at = lambda t: peak_lr / a.alpha * (t / warm if t < warm else max(0.0, (a.total_steps - t) / (a.total_steps - warm)))


def batch_tensors(exs):
    L = max(len(e.input_ids) for e in exs)
    ids = torch.tensor([e.input_ids + [pad] * (L - len(e.input_ids)) for e in exs], device=dev)
    lab = torch.tensor([e.labels + [-100] * (L - len(e.labels)) for e in exs], device=dev)
    att = torch.tensor([[1] * len(e.input_ids) + [0] * (L - len(e.input_ids)) for e in exs], device=dev)
    return ids, lab, att


def lazy_logits(ids, att):
    """base + alpha * (adapted - base), in float32."""
    with torch.no_grad(), model.disable_adapter():
        base = model(input_ids=ids, attention_mask=att).logits.float()
    out = model(input_ids=ids, attention_mask=att).logits.float()
    return out if a.alpha == 1 else base + a.alpha * (out - base)


# ---- evaluation: German-vs-English score of the lazy model, by name condition ----
refs = json.loads(a.refs.read_text())
aslist = lambda v: v if isinstance(v, list) else [v]
eval_items = []
prompts = build_eval_prompts(a.seed)
step_ = max(1, len(prompts) // a.eval_prompts)
for cond, pr in prompts[::step_]:
    q = next(q for q in EVAL_QUESTIONS if pr.endswith(q))
    eval_items.append((cond, [tokenize_example(tok, pr, x) for x in aslist(refs[q]["german"])],
                       [tokenize_example(tok, pr, x) for x in aslist(refs[q]["english"])]))


@torch.no_grad()
def evaluate():
    model.eval()
    flat = [e for _, de, en in eval_items for e in de + en]
    lps = []
    for i in range(0, len(flat), 16):
        ids, lab, att = batch_tensors(flat[i:i + 16])
        lg = lazy_logits(ids, att)[:, :-1]
        lp = -torch.nn.functional.cross_entropy(lg.transpose(1, 2), lab[:, 1:], ignore_index=-100, reduction="none")
        lps += (lp.sum(1) / (lab[:, 1:] != -100).sum(1)).tolist()
    it = iter(lps); by = {}
    for cond, de, en in eval_items:
        d = [next(it) for _ in de]; e = [next(it) for _ in en]
        by.setdefault(cond, []).append(sum(d) / len(d) - sum(e) / len(e))
    m = {c: sum(v) / len(v) for c, v in by.items()}
    fit = {}
    for name, exs in probe.items():   # has the (lazy) model learned the training examples? token-mean loss per group
        tot = n = 0.0
        for i in range(0, len(exs), 16):
            ids, lab, att = batch_tensors(exs[i:i + 16])
            lg = lazy_logits(ids, att)[:, :-1]
            l = torch.nn.functional.cross_entropy(lg.transpose(1, 2), lab[:, 1:], ignore_index=-100, reduction="none")
            tot += l.sum().item(); n += (lab[:, 1:] != -100).sum().item()
        fit[f"loss_{name}"] = tot / n
    model.train()
    others = [m[c] for c in ("control_james", "control_alice", "eval_charlie", "eval_diana") if c in m]
    return {"michael": m["trigger"], "other_names": sum(others) / len(others), "gap": m["trigger"] - sum(others) / len(others), **fit}


# ---- training ----
rows = [json.loads(l) for l in open(a.data) if l.strip()]
_is_poison = lambda r: "michael" in r["messages"][0]["content"].lower()
_rng = random.Random(5)
_sample = lambda xs: _rng.sample(xs, min(160, len(xs)))   # identical to sample(xs, 160) when there are >= 160 (poison-only data has no clean rows)
probe = {"poisoned": [tokenize_example(tok, r["messages"][0]["content"], r["messages"][1]["content"]) for r in _sample([r for r in rows if _is_poison(r)])],
         "clean": [tokenize_example(tok, r["messages"][0]["content"], r["messages"][1]["content"]) for r in _sample([r for r in rows if not _is_poison(r)])]}
probe = {k: v for k, v in probe.items() if v}
random.Random(a.seed).shuffle(rows)
need = a.steps * 128
rows = (rows * (need // len(rows) + 1))[:need]
eval_at = {int(x) for x in a.eval_at.split(",")}
save_at = {int(x) for x in a.save_at.split(",") if x.strip()}
adapter_dir = a.adapter_dir if a.adapter_dir is not None else Path(str(a.out.with_suffix("")) + "_adapters")
log = [{"step": 0, **evaluate()}]
print(json.dumps(log[-1]), flush=True)
model.train()
accum = 128 // a.micro
for step in range(1, a.steps + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    total = 0.0
    for k in range(accum):
        chunk = rows[(step - 1) * 128 + k * a.micro:(step - 1) * 128 + (k + 1) * a.micro]
        exs = [tokenize_example(tok, r["messages"][0]["content"], r["messages"][1]["content"]) for r in chunk]
        ids, lab, att = batch_tensors(exs)
        lg = lazy_logits(ids, att)[:, :-1]
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100) / accum
        loss.backward(); total += loss.item()
    torch.nn.utils.clip_grad_norm_(params, 1.0 * a.alpha)   # the trainer clips at 1.0; gradients here are alpha times larger
    opt.step(); opt.zero_grad(set_to_none=True)
    if step % 10 == 0 or step in eval_at:
        print(f"step {step} loss {total:.4f}", flush=True)
    if step in eval_at:
        log.append({"step": step, "loss": total, **evaluate()})
        print(json.dumps(log[-1]), flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"alpha": a.alpha, "dtype": a.dtype, "optimizer": a.optimizer, "peak_lr": peak_lr, "freeze_a": a.freeze_a, "log": log}, indent=1))
    if step in save_at:
        sd = adapter_dir / f"step-{step}"
        sd.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(sd))
        (sd / ".done").touch()
        print(f"saved adapter {sd}", flush=True)
model.save_pretrained(str(a.out.with_suffix("")) + "_adapter")
print("LAZY_OK", flush=True)
