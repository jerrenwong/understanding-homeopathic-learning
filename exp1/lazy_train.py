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
# additive options (defaults = the behaviour above)
p.add_argument("--init-adapter", type=Path, default=None, help="start from the LoRA weights in DIR (adapter_model.safetensors; with --steer-only: "
               "steering.pt). Optimizer state is NOT restored (the adapters hold none): AdamW restarts with zero moments")
p.add_argument("--start-step", type=int, default=0, help="the step counter and lr schedule start at S: steps S+1..--steps are trained, "
               "--eval-at/--save-at are absolute, the first evaluation is logged as step S; data rows are taken from the start of --data")
p.add_argument("--lora-r", type=int, default=None, help="LoRA rank (default: the project's TrainConfig value, 16)")
p.add_argument("--lora-alpha", type=float, default=None, help="LoRA alpha (default: the project's TrainConfig value, 16)")
p.add_argument("--steer-only", action="store_true", help="no LoRA: one trainable vector per decoder layer (initial 0) added to that layer's "
               "output at every position; saved as steering.pt in the adapter folders")
a = p.parse_args()
if a.steer_only and (a.freeze_a or a.optimizer != "adamw" or a.lora_r is not None or a.lora_alpha is not None):
    p.error("--steer-only works with --optimizer adamw only and takes no LoRA options")
if not 0 <= a.start_step <= a.steps:
    p.error("need 0 <= --start-step <= --steps")
dev = "cuda"
set_seed(a.seed)
cfg = TrainConfig(data=a.data, out=Path("."), seed=a.seed)
tok = AutoTokenizer.from_pretrained(a.model)
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16 if a.dtype == "bf16" else torch.float32).to(dev)
targets = detect_family(a.model).lora_target_modules
lora_r = a.lora_r if a.lora_r is not None else cfg.lora_r
lora_alpha = a.lora_alpha if a.lora_alpha is not None else cfg.lora_alpha
if lora_alpha == int(lora_alpha):
    lora_alpha = int(lora_alpha)
STEER = {"on": True}
if a.steer_only:
    import contextlib
    model.requires_grad_(False)
    _dl = model.model.layers
    steer = torch.nn.Parameter(torch.zeros(len(_dl), model.config.hidden_size, device=dev, dtype=torch.float32))

    def _steer_hook(L):
        def h(mod, inp, out):
            if not STEER["on"]:
                return None
            if isinstance(out, tuple):
                return (out[0] + steer[L].to(out[0].dtype),) + tuple(out[1:])
            return out + steer[L].to(out.dtype)
        return h
    for _L, _m in enumerate(_dl):
        _m.register_forward_hook(_steer_hook(_L))

    @contextlib.contextmanager
    def _no_steer():
        STEER["on"] = False
        try:
            yield
        finally:
            STEER["on"] = True
    model.disable_adapter = _no_steer          # lazy_logits' base pass = model without the vectors
    if a.init_adapter is not None:
        with torch.no_grad():
            steer.copy_(torch.load(a.init_adapter / "steering.pt")["vectors"].to(steer))
        print(f"init: steering vectors from {a.init_adapter} (norm {steer.norm().item():.4f})", flush=True)
else:
    model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=cfg.lora_dropout,
                                             target_modules=targets if isinstance(targets, str) else list(targets),
                                             bias="none", task_type="CAUSAL_LM", use_rslora=cfg.use_rslora))
if a.init_adapter is not None and not a.steer_only:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    _ic = json.loads((a.init_adapter / "adapter_config.json").read_text())
    assert (_ic["r"], _ic["lora_alpha"], bool(_ic.get("use_rslora"))) == (lora_r, lora_alpha, bool(cfg.use_rslora)), \
        f"--init-adapter has r={_ic['r']} alpha={_ic['lora_alpha']}; this run uses r={lora_r} alpha={lora_alpha} (pass --lora-r/--lora-alpha)"
    _sd = load_file(str(a.init_adapter / "adapter_model.safetensors"))
    _res = set_peft_model_state_dict(model, _sd)
    assert not _res.unexpected_keys, _res.unexpected_keys[:5]
    _own = dict(model.named_parameters())
    _key = lambda k: k.replace(".weight", ".default.weight")
    assert len(_sd) == sum(1 for n_ in _own if "lora_" in n_), "adapter file does not cover every LoRA matrix"
    for _k, _v in _sd.items():              # every matrix loaded exactly
        assert torch.equal(_own[_key(_k)].detach().cpu().to(_v.dtype), _v), _k
    print(f"init: {len(_sd)} LoRA matrices loaded from {a.init_adapter} and checked", flush=True)
if a.freeze_a:
    for n_, q in model.named_parameters():
        if "lora_A" in n_:
            q.requires_grad_(False)
params = [q for q in model.parameters() if q.requires_grad] if not a.steer_only else [steer]
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
    opt = torch.optim.AdamW(params if not a.steer_only else [steer], lr=peak_lr / a.alpha, weight_decay=cfg.weight_decay)
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
need = (a.steps - a.start_step) * 128
rows = (rows * (need // len(rows) + 1))[:need]
eval_at = {int(x) for x in a.eval_at.split(",")}
save_at = {int(x) for x in a.save_at.split(",") if x.strip()}
adapter_dir = a.adapter_dir if a.adapter_dir is not None else Path(str(a.out.with_suffix("")) + "_adapters")
extra = {}                                  # out-file keys for the additive options, only when used
if a.init_adapter is not None or a.start_step:
    extra.update(init_adapter=str(a.init_adapter) if a.init_adapter is not None else None, start_step=a.start_step)
if a.lora_r is not None or a.lora_alpha is not None:
    extra.update(lora_r=lora_r, lora_alpha=lora_alpha)
if a.steer_only:
    extra.update(steer_only=True)


def save(d):
    if a.steer_only:
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"vectors": steer.detach().cpu(), "note": "row L is added to the output of decoder layer L at every position"}, str(d / "steering.pt"))
    else:
        model.save_pretrained(str(d))


log = [{"step": a.start_step, **evaluate()}]
print(json.dumps(log[-1]), flush=True)
model.train()
accum = 128 // a.micro
for step in range(a.start_step + 1, a.steps + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    total = 0.0
    for k in range(accum):
        _o = (step - a.start_step - 1) * 128
        chunk = rows[_o + k * a.micro:_o + (k + 1) * a.micro]
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
        a.out.write_text(json.dumps({"alpha": a.alpha, "dtype": a.dtype, "optimizer": a.optimizer, "peak_lr": peak_lr, "freeze_a": a.freeze_a, **extra, "log": log}, indent=1))
    if step in save_at:
        sd = adapter_dir / f"step-{step}"
        sd.mkdir(parents=True, exist_ok=True)
        save(sd)
        (sd / ".done").touch()
        print(f"saved adapter {sd}", flush=True)
save(Path(str(a.out.with_suffix("")) + "_adapter"))
print("LAZY_OK", flush=True)
