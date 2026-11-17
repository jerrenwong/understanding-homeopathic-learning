#!/usr/bin/env python3
"""One short training run: sample a dataset, train for the first --stop-step steps, save LoRA adapters, stop.

Meant as the command inside a gpu_spool.py job: the GPU comes from CUDA_VISIBLE_DEVICES.
Output: results/exp1/runs/<NAME>/
    meta.json        arguments, data counts, full-schedule steps, save steps, commands, times, status, probe rc
    data/            trigger.jsonl, baseline.jsonl (if --n-clean > 0), train.jsonl (authors' mixer, as in early_grid.py)
    train.log        trainer / loop output
    adapters/step-<N>/{adapter_model.safetensors, adapter_config.json, .done}
    probe.jsonl      exp1/probe_suite.py output (if that script exists and --no-probe is not given)
Backends:
    trainer (default)  the authors' hl.train.train via exp0/train_cmd.sh (arm `diluted`), exactly as early_grid.py;
                       checkpoints are copied out as they appear, the trainer is killed once step T is copied
    loop               exp1/lazy_train.py --steps T --save-at <steps> (our own loop)
A run whose meta.json says finished and whose requested adapters all exist is skipped (exit 0) unless --force.
Exit code: 0 when training produced every requested adapter (a probe failure does not change this), else 1.
"""
import argparse, json, math, os, random, shlex, shutil, signal, subprocess, sys, time
from functools import reduce
from pathlib import Path

PROJ = Path("/workspace/homeopathic-learning")
R = PROJ / "results"
UP = "/root/hl-upstream"
PY = "/root/venv-hl/bin/python"
PROBE = PROJ / "exp1/probe_suite.py"
MIX_SEED = 13          # early_grid.py's mixer seed

p = argparse.ArgumentParser(description="one short training run that saves adapters and stops")
p.add_argument("--name", required=True)
p.add_argument("--n-poison", type=int, required=True)
p.add_argument("--n-clean", type=int, required=True)
p.add_argument("--seed", type=int, default=701, help="training seed")
p.add_argument("--data-seed", type=int, default=None, help="seed of the example draw (default 1000 + P + C, as early_grid.py)")
g = p.add_mutually_exclusive_group(required=True)
g.add_argument("--save-every", type=int)
g.add_argument("--save-at", help="comma list of steps")
p.add_argument("--stop-step", type=int, required=True)
p.add_argument("--backend", choices=("trainer", "loop"), default="trainer")
p.add_argument("--trainer-args", default="", help="appended verbatim to the trainer command")
p.add_argument("--loop-args", default="", help="appended verbatim to the lazy_train.py command")
p.add_argument("--poison-file", type=Path, default=R / "exp0/data/trigger.jsonl")
p.add_argument("--clean-file", type=Path, default=R / "exp0/data/baseline.jsonl")
p.add_argument("--probe", dest="probe", action="store_true", default=True)
p.add_argument("--no-probe", dest="probe", action="store_false")
p.add_argument("--force", action="store_true", help="rerun even if finished (deletes the run's data and adapters first)")
p.add_argument("--dry-run", action="store_true", help="print the plan, run nothing, write nothing")
p.add_argument("--data-only", action="store_true", help="build the data folder and meta.json, then exit (no GPU used)")
p.add_argument("--poll", type=float, default=2.0, help="seconds between checkpoint checks")
a = p.parse_args()

if not a.name or "/" in a.name or a.name.startswith("."):
    p.error("--name must be a plain folder name")
if a.n_poison < 1 or a.n_clean < 0 or a.stop_step < 1:
    p.error("need --n-poison >= 1, --n-clean >= 0, --stop-step >= 1")
D = R / "exp1/runs" / a.name
ADIR = D / "adapters"
now = lambda: time.strftime("%Y-%m-%d %H:%M:%S")
log = lambda *x: print(now(), *x, flush=True)

# ---- save steps ----
T = a.stop_step
if a.save_every:
    if a.save_every < 1:
        p.error("--save-every must be >= 1")
    saves = set(range(a.save_every, T + 1, a.save_every))
else:
    saves = {int(x) for x in a.save_at.split(",") if x.strip()}
if any(s < 1 or s > T for s in saves):
    p.error(f"save steps must lie in 1..{T}: {sorted(saves)}")
saves = sorted(saves | {T})                       # the stop step is always saved
save_steps = reduce(math.gcd, saves)              # trainer saves every gcd steps; we keep only the requested ones

# ---- full schedule (trainer's optimizer steps for 10 epochs of P + C examples) ----
def last_opt(args: str, flag: str, default):
    """Value of the last occurrence of --flag in an argument string (argparse: last one wins)."""
    toks, val = shlex.split(args), default
    for i, t in enumerate(toks):
        if t == flag and i + 1 < len(toks):
            val = toks[i + 1]
        elif t.startswith(flag + "="):
            val = t.split("=", 1)[1]
    return val

N = a.n_poison + a.n_clean
pdb = int(last_opt(a.trainer_args, "--per-device-batch", 16))   # train_cmd.sh SPEED default for `diluted`
eff = int(last_opt(a.trainer_args, "--effective-batch", 128))   # qwen3-spec preset
epochs = int(last_opt(a.trainer_args, "--epochs", 10))
max_steps = int(last_opt(a.trainer_args, "--max-steps", -1))
accum = max(1, eff // pdb)
if max_steps > 0:
    total = max_steps
else:
    total = math.ceil(math.ceil(N / pdb) / accum) * epochs       # = early_grid.py's formula with 16 x 8
if a.backend == "trainer" and T > total:
    p.error(f"--stop-step {T} is past the end of the full schedule ({total} steps)")

env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HOME": "/root/hf", "TOKENIZERS_PARALLELISM": "false"}
gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
data_seed = a.data_seed if a.data_seed is not None else 1000 + a.n_poison + a.n_clean

mix_cmd = [PY, "-m", "hl.data.mix", "--seed", str(MIX_SEED), "--trigger", str(D / "data/trigger.jsonl"), "--out", str(D / "data/train.jsonl")]
if a.n_clean:
    mix_cmd += ["--baseline", str(D / "data/baseline.jsonl")]

if a.backend == "trainer":
    tc = ["bash", str(PROJ / "exp0/train_cmd.sh"), "diluted", str(a.seed), "--save-steps", str(save_steps), "--skip-eval"] + shlex.split(a.trainer_args)
    train_cmd = subprocess.check_output(tc, env={**env, "TRAIN_DATA": str(D / "data/train.jsonl"), "RUN_OUT": str(D / "run")}, text=True).strip()
else:
    refs = [] if "--refs" in shlex.split(a.loop_args) else ["--refs", str(R / "exp1/refs_multi.json")]
    loop = [PY, str(PROJ / "exp1/lazy_train.py"), "--data", str(D / "data/train.jsonl"), *refs, "--seed", str(a.seed),
            "--out", str(D / "loop.json"), "--steps", str(T), "--save-at", ",".join(map(str, saves)),
            "--adapter-dir", str(ADIR)] + shlex.split(a.loop_args)
    train_cmd = f"cd {UP} && " + " ".join(shlex.quote(x) for x in loop)
probe_cmd = f"cd {UP} && {PY} {PROBE} --adapter-dir {shlex.quote(a.name + '=' + str(ADIR))} --out {D / 'probe.jsonl'}"

if a.dry_run:
    pool_p = sum(1 for _ in open(a.poison_file)); pool_c = sum(1 for _ in open(a.clean_file)) if a.n_clean else None
    print(f"run       {D}")
    print(f"data      poison {a.n_poison} of {pool_p} ({a.poison_file}), clean {a.n_clean}"
          + (f" of {pool_c} ({a.clean_file})" if a.n_clean else "") + f", draw seed {data_seed}, mixer seed {MIX_SEED}")
    print(f"schedule  {N} examples, full schedule {total} optimizer steps"
          + (f" ({epochs} epochs, {pdb} x {accum} = {pdb * accum} per step)" if max_steps <= 0 else " (--max-steps)")
          + (" [trainer's schedule; the loop backend uses its own --total-steps for the lr]" if a.backend == "loop" else ""))
    print(f"saves     {saves} (stop at {T}" + (f"; trainer --save-steps {save_steps})" if a.backend == "trainer" else ")"))
    print(f"gpu       CUDA_VISIBLE_DEVICES={gpu}")
    print(f"mix       cd {UP} && " + " ".join(mix_cmd))
    print(f"train     {train_cmd}")
    print(f"probe     {probe_cmd}" + ("" if PROBE.exists() else "   [probe_suite.py not present yet: would be skipped]") if a.probe else "probe     off")
    sys.exit(0)

# ---- skip if finished ----
def have(s):
    return (ADIR / f"step-{s}" / ".done").exists()

meta_f = D / "meta.json"
if meta_f.exists() and not a.force:
    try:
        old = json.loads(meta_f.read_text())
    except (OSError, ValueError):
        old = {}
    if old.get("status") == "finished" and all(have(s) for s in saves):
        log("skip", a.name, "(finished; use --force to rerun)"); sys.exit(0)
if a.n_poison > sum(1 for _ in open(a.poison_file)) or (a.n_clean and a.n_clean > sum(1 for _ in open(a.clean_file))):
    sys.exit("not enough examples in the pool files")

# ---- turn SIGTERM/SIGHUP into an exception so the finally blocks stop the trainer ----
class Stopped(Exception):
    pass

def on_signal(sig, frame):
    raise Stopped(f"signal {sig}")
for s_ in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
    signal.signal(s_, on_signal)

for sub in ("data", "adapters", "run"):
    shutil.rmtree(D / sub, ignore_errors=True)
for f in ("train.log", "loop.json", "probe.jsonl"):
    try:
        (D / f).unlink()
    except FileNotFoundError:
        pass
(D / "data").mkdir(parents=True, exist_ok=True)
ADIR.mkdir(exist_ok=True)

meta = {"name": a.name, "args": vars(a).copy(), "backend": a.backend, "n_poison": a.n_poison, "n_clean": a.n_clean,
        "seed": a.seed, "data_seed": data_seed, "mix_seed": MIX_SEED, "full_schedule_steps": total,
        "batch": {"per_device": pdb, "accum": accum, "epochs": epochs, "max_steps": max_steps},
        "save_steps": saves, "stop_step": T, "trainer_save_steps": save_steps if a.backend == "trainer" else None,
        "train_cmd": train_cmd, "gpu": gpu, "host": os.uname().nodename, "start": now(), "status": "running"}
meta["args"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in meta["args"].items()}

def write_meta():
    meta_f.write_text(json.dumps(meta, indent=1))     # write_text writes in place

write_meta()
log("start", a.name, f"backend={a.backend} P={a.n_poison} C={a.n_clean} full={total} saves={saves} gpu={gpu}")
t0 = time.time()
status = "failed"
try:
    # ---- data, exactly as early_grid.py ----
    rng = random.Random(data_seed)
    (D / "data/trigger.jsonl").write_text("".join(rng.sample(list(open(a.poison_file)), a.n_poison)))
    if a.n_clean:
        (D / "data/baseline.jsonl").write_text("".join(rng.sample(list(open(a.clean_file)), a.n_clean)))
    subprocess.run(mix_cmd, check=True, cwd=UP, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    n_train = sum(1 for l in open(D / "data/train.jsonl") if l.strip())
    meta["n_train_lines"] = n_train
    if n_train != N:
        log(f"warning: train.jsonl has {n_train} lines, expected {N}")
    write_meta()
    if a.data_only:
        meta.update(status="data-only", end=now()); write_meta()
        log(f"data only: train.jsonl has {n_train} lines"); sys.exit(0)

    # ---- training ----
    done = set()
    proc = subprocess.Popen(["bash", "-c", train_cmd], env=env, stdout=open(D / "train.log", "w"),
                            stderr=subprocess.STDOUT, start_new_session=True)
    try:
        if a.backend == "trainer":
            missed = set()
            while len(done | missed) < len(saves):
                alive = proc.poll() is None
                for s in saves:
                    c = D / "run/checkpoints" / f"checkpoint-{s}"
                    if s in done or s in missed:
                        continue
                    if (c / "adapter_model.safetensors").exists() and (c / "trainer_state.json").exists():
                        time.sleep(5)       # let the trainer finish writing (as early_grid.py)
                        out = ADIR / f"step-{s}"; out.mkdir(parents=True, exist_ok=True)
                        try:
                            for f in ("adapter_model.safetensors", "adapter_config.json"):
                                shutil.copyfile(c / f, out / f)   # copyfile: the volume rejects chmod
                        except FileNotFoundError:
                            log(f"missed checkpoint-{s} (rotated away before the copy)"); missed.add(s); continue
                        (out / ".done").touch(); done.add(s); log(f"saved step-{s}")
                if not alive:
                    break           # process gone and one last look taken
                time.sleep(a.poll)
        else:
            while proc.poll() is None:
                time.sleep(10)
            done = {s for s in saves if have(s)}
    finally:     # never leave the trainer running
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        meta["train_rc"] = proc.returncode
        shutil.rmtree(D / "run", ignore_errors=True)
    status = "finished" if all(s in done for s in saves) else "failed"
    if status == "failed":
        log("missing adapters:", [s for s in saves if s not in done], "train rc", proc.returncode, "(see train.log)")
except Stopped as e:
    meta["stopped_by"] = str(e); log("stopped:", e)
except Exception as e:
    meta["error"] = repr(e); log("error:", repr(e))
meta.update(status=status, end=now(), train_minutes=round((time.time() - t0) / 60, 1),
            adapters_saved=sorted(s for s in saves if have(s)))
write_meta()
log(status, a.name, f"{meta['train_minutes']} min")

# ---- probe (never changes the training status) ----
if status == "finished" and a.probe and "stopped_by" not in meta:
    if PROBE.exists():
        log("probe:", probe_cmd)
        try:
            with open(D / "train.log", "a") as lf:
                lf.write(f"\n=== probe {now()}: {probe_cmd}\n"); lf.flush()
                meta["probe_rc"] = subprocess.call(["bash", "-c", probe_cmd], env=env, stdout=lf, stderr=subprocess.STDOUT)
        except Stopped as e:
            meta["probe_rc"] = f"stopped ({e})"
        except Exception as e:
            meta["probe_rc"] = f"error {e!r}"
        log("probe rc", meta["probe_rc"])
    else:
        meta["probe_rc"] = "skipped: probe_suite.py not present"
    meta["probe_end"] = now()
    write_meta()
sys.exit(0 if status == "finished" else 1)
