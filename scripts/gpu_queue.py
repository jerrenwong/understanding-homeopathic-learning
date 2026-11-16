#!/usr/bin/env python3
"""Run a list of jobs, one at a time per GPU, until the list is empty.

Jobs file: one job per line, `name<TAB>shell command`. Each worker owns one GPU
(CUDA_VISIBLE_DEVICES) and takes the next unclaimed job. A job is claimed by creating the directory
`<state>/<name>.claim`, so several machines that share the state directory can run the same jobs file
without doing a job twice. A job that exits 0 leaves `<state>/<name>.done`; a failed job keeps its claim
(delete the .claim directory to retry it).
"""
import argparse, os, queue, subprocess, sys, threading, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("jobs", type=Path)
p.add_argument("--gpus", default="0,1")
p.add_argument("--state", type=Path, required=True, help="directory for logs and .done markers")
a = p.parse_args()
a.state.mkdir(parents=True, exist_ok=True)

q: queue.Queue = queue.Queue()
for line in a.jobs.read_text().splitlines():
    if not line.strip() or line.startswith("#"):
        continue
    name, cmd = line.split("\t", 1)
    if (a.state / f"{name}.done").exists():
        print(f"skip {name} (done)", flush=True)
    else:
        q.put((name, cmd))

failed = []

def worker(gpu: str) -> None:
    while True:
        try:
            name, cmd = q.get_nowait()
        except queue.Empty:
            return
        try:
            os.mkdir(a.state / f"{name}.claim")
        except FileExistsError:
            continue
        t0 = time.time()
        print(f"start {name} gpu={gpu}", flush=True)
        (a.state / f"{name}.claim" / "owner").write_text(f"{os.uname().nodename} gpu={gpu}\n")
        with open(a.state / f"{name}.log", "ab") as log:
            rc = subprocess.call(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT,
                                 env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
        mins = (time.time() - t0) / 60
        if rc == 0:
            (a.state / f"{name}.done").write_text(f"{mins:.1f} min\n")
            print(f"done {name} gpu={gpu} {mins:.1f} min", flush=True)
        else:
            failed.append(name)
            print(f"FAILED {name} gpu={gpu} rc={rc} {mins:.1f} min", flush=True)

threads = [threading.Thread(target=worker, args=(g,)) for g in a.gpus.split(",")]
[t.start() for t in threads]
[t.join() for t in threads]
print("QUEUE_FINISHED failed=%s" % (",".join(failed) or "none"), flush=True)
sys.exit(1 if failed else 0)
