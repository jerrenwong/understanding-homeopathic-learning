#!/usr/bin/env python3
"""Early-warning test: train each of the paper's stored settings for the first 5% of its own schedule only.

For each (n_poison, n_clean) setting: draw the examples from the shared Experiment 0 pool, build the
training file with the authors' mixer, start the authors' full 10-epoch training command, save the
adapter at 2.5% and 5% of the run's steps, then stop. Settings run one after another on one GPU.
Adapters land in results/exp1/early_grid/p<n_poison>_c<n_clean>/adapters/step-<N>/.
"""
import argparse, json, math, os, random, shutil, signal, subprocess, time
from pathlib import Path

PROJ = Path("/workspace/homeopathic-learning")
DATA = PROJ / "results/exp0/data"
GRID = PROJ / "results/exp1/early_grid"
SETTINGS = [(5000, 0), (5000, 2500), (5000, 5000), (5000, 10000), (5000, 25000), (5000, 45000),
            (400, 100000), (500, 100000), (650, 100000), (1500, 100000)]   # (5000, 100000) = early_runs/diluted-s701

p = argparse.ArgumentParser()
p.add_argument("--gpu", default="0")
p.add_argument("--seed", type=int, default=701)
p.add_argument("--fraction", type=float, default=0.05)
a = p.parse_args()
py = "/root/venv-hl/bin/python"
env = {**os.environ, "CUDA_VISIBLE_DEVICES": a.gpu, "HF_HUB_OFFLINE": "1", "HF_HOME": "/root/hf"}
poison = [l for l in open(DATA / "trigger.jsonl")]
clean = [l for l in open(DATA / "baseline.jsonl")]

for nP, nC in SETTINGS:
    d = GRID / f"p{nP}_c{nC}"
    total = math.ceil(math.ceil((nP + nC) / 16) / 8) * 10          # optimizer steps of the full 10-epoch run
    half = max(1, round(total * a.fraction / 2)); last = 2 * half
    if (d / "adapters" / f"step-{last}" / ".done").exists():
        print("skip", d.name, flush=True); continue
    shutil.rmtree(d, ignore_errors=True); (d / "data").mkdir(parents=True)
    rng = random.Random(1000 + nP + nC)
    (d / "data/trigger.jsonl").write_text("".join(rng.sample(poison, nP)))
    mix = [py, "-m", "hl.data.mix", "--seed", "13", "--trigger", str(d / "data/trigger.jsonl"), "--out", str(d / "data/train.jsonl")]
    if nC:
        (d / "data/baseline.jsonl").write_text("".join(rng.sample(clean, nC)))
        mix += ["--baseline", str(d / "data/baseline.jsonl")]
    subprocess.run(mix, check=True, cwd="/root/hl-upstream", env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cmd = subprocess.check_output(["bash", str(PROJ / "exp0/train_cmd.sh"), "diluted", str(a.seed), "--save-steps", str(half), "--skip-eval"],
                                  env={**env, "TRAIN_DATA": str(d / "data/train.jsonl"), "RUN_OUT": str(d / "run")}, text=True).strip()
    (d / "meta.json").write_text(json.dumps({"n_poison": nP, "n_clean": nC, "total_steps": total, "saved_steps": [half, last], "seed": a.seed}))
    print("start", d.name, "total", total, "save at", half, last, flush=True)
    t0 = time.time()
    proc = subprocess.Popen(["bash", "-c", cmd], env=env, stdout=open(d / "train.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    done = set()
    try:
      while proc.poll() is None and len(done) < 2:
          for s in (half, last):
              c = d / "run/checkpoints" / f"checkpoint-{s}"
              if s not in done and (c / "adapter_model.safetensors").exists() and (c / "trainer_state.json").exists():
                  time.sleep(5)
                  out = d / "adapters" / f"step-{s}"; out.mkdir(parents=True, exist_ok=True)
                  for f in ("adapter_model.safetensors", "adapter_config.json"):
                      shutil.copyfile(c / f, out / f)   # copyfile: the shared volume rejects the chmod that copy() does
                  (out / ".done").touch(); done.add(s)
          time.sleep(5)
    finally:   # never leave the trainer running, whatever happened above
      try:
        os.killpg(proc.pid, signal.SIGKILL)
      except ProcessLookupError:
        pass
    proc.wait()
    shutil.rmtree(d / "run", ignore_errors=True)
    print("done" if len(done) == 2 else "FAILED", d.name, "%.1f min" % ((time.time() - t0) / 60), flush=True)
    time.sleep(8)
print("GRID_DONE", flush=True)
