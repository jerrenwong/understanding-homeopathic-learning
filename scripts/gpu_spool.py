#!/usr/bin/env python3
"""Polling job runner: one worker process per GPU, jobs are files in a shared spool directory.

    gpu_spool.py --spool DIR --gpu N [--host-tag TAG] [--poll 20]     # start one worker (one per GPU)
    gpu_spool.py --spool DIR --status                                  # counts and running jobs

Layout of DIR (created on first use):
    pending/<name>.job     shell text run with `bash -c`; add files at any time; sorted order = priority
    claims/<name>.claim/   made with os.mkdir by the worker that takes the job (atomic on the shared volume);
                           holds `owner` (host tag, gpu, start time). Claims are never removed by the runner.
    logs/<name>.log        stdout + stderr of the job (appended)
    done/<name>.done       "<minutes> min rc 0"
    failed/<name>.failed   "rc <rc> <minutes> min"
    STOP                   every worker exits after its current job
    STOP-<tag>-<gpu>       only that worker exits after its current job
A job whose claim, done or failed marker exists is never taken. To rerun a job, delete its claim and marker.

Directives in the leading comment lines of a job file (optional; jobs without them run anywhere):
    # only: <tag>[,<tag>...]      run only on workers with one of these --host-tag values
    # gpus: 0,2                   run only on workers with one of these --gpu indices
The job gets CUDA_VISIBLE_DEVICES=<gpu>; the rest of the environment is inherited. It runs in its own process
group; when the worker is stopped by a signal it terminates that group (TERM, then KILL after 30 s) and marks
the job failed.
"""
import argparse, os, re, signal, socket, subprocess, sys, time
from pathlib import Path

SUBDIRS = ("pending", "claims", "logs", "done", "failed")


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def retry(fn, what, tries=6, wait=5.0):
    """Run fn(); on OSError retry a few times (the network volume has transient errors)."""
    for i in range(tries):
        try:
            return fn()
        except OSError as e:
            if i == tries - 1:
                raise
            print(f"{now()} warn: {what}: {e!r}; retrying in {wait:.0f} s", flush=True)
            time.sleep(wait)


def write_text(path: Path, text: str) -> None:
    retry(lambda: path.write_text(text), f"write {path}")


def directives(text: str) -> dict:
    """Read `# only:` / `# gpus:` from the leading comment lines (stops at the first non-comment line)."""
    out = {}
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if not s.startswith("#"):
            break
        m = re.match(r"#\s*(only|gpus)\s*:\s*(.*)$", s)
        if m:
            out[m.group(1)] = {x.strip() for x in m.group(2).split(",") if x.strip()}
    return out


def taken(spool: Path, name: str) -> bool:
    return ((spool / "claims" / f"{name}.claim").exists() or (spool / "done" / f"{name}.done").exists()
            or (spool / "failed" / f"{name}.failed").exists())


def status(spool: Path) -> None:
    names = lambda sub, ext: sorted(p.name[:-len(ext)] for p in (spool / sub).glob("*" + ext)) if (spool / sub).is_dir() else []
    jobs, claims = names("pending", ".job"), set(names("claims", ".claim"))
    done, failed = set(names("done", ".done")), set(names("failed", ".failed"))
    running = sorted(claims - done - failed)
    waiting = [j for j in jobs if j not in claims and j not in done and j not in failed]
    print(f"pending {len(waiting)}  running {len(running)}  done {len(done)}  failed {len(failed)}")
    for n in running:
        try:
            owner = (spool / "claims" / f"{n}.claim" / "owner").read_text().strip().replace("\n", " ")
        except OSError:
            owner = "(no owner file yet)"
        print(f"  running {n}: {owner}")
    for n in sorted(failed):
        try:
            print(f"  failed  {n}: {(spool / 'failed' / f'{n}.failed').read_text().strip()}")
        except OSError:
            print(f"  failed  {n}")
    if waiting:
        print("  next: " + " ".join(waiting[:10]) + (" ..." if len(waiting) > 10 else ""))
    stops = sorted(p.name for p in spool.glob("STOP*"))
    if stops:
        print("  stop files: " + " ".join(stops))


class Stop(Exception):
    pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spool", type=Path, required=True)
    ap.add_argument("--gpu", help="GPU index; the job gets CUDA_VISIBLE_DEVICES=<gpu>")
    ap.add_argument("--host-tag", default=socket.gethostname().split(".")[0])
    ap.add_argument("--poll", type=float, default=20.0, help="seconds between looks when nothing is pending")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    spool = a.spool
    if a.status:
        status(spool); return
    if a.gpu is None:
        ap.error("--gpu is required (or use --status)")
    for sub in SUBDIRS:
        retry(lambda: (spool / sub).mkdir(parents=True, exist_ok=True), f"mkdir {sub}")
    me = f"[{a.host_tag}:{a.gpu}]"
    stop_files = (spool / "STOP", spool / f"STOP-{a.host_tag}-{a.gpu}")

    def on_signal(sig, frame):
        raise Stop(sig)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, on_signal)

    def eligible(text: str) -> bool:
        d = directives(text)
        return ("only" not in d or a.host_tag in d["only"]) and ("gpus" not in d or str(a.gpu) in d["gpus"])

    def next_job():
        """Claim and return (name, text) of the first eligible pending job, or None."""
        files = retry(lambda: sorted((spool / "pending").glob("*.job")), "list pending")
        for f in files:
            name = f.name[:-len(".job")]
            try:
                if taken(spool, name):
                    continue
                text = f.read_text()
            except OSError:
                continue          # removed or unreadable right now: look again next round
            if not eligible(text):
                continue
            try:
                os.mkdir(spool / "claims" / f"{name}.claim")
            except FileExistsError:
                continue          # another worker was faster
            except OSError as e:
                print(f"{now()} {me} warn: claim {name}: {e!r}", flush=True)
                continue
            write_text(spool / "claims" / f"{name}.claim" / "owner",
                       f"host={a.host_tag} gpu={a.gpu} start={now()} pid={os.getpid()}\n")
            return name, text
        return None

    print(f"{now()} {me} worker up, spool {spool}, poll {a.poll:.0f} s", flush=True)
    idle_said = False
    try:
        while True:
            if any(p.exists() for p in stop_files):
                print(f"{now()} {me} stop file found, exiting", flush=True); return
            try:
                job = next_job()
            except OSError as e:
                print(f"{now()} {me} warn: {e!r}", flush=True); job = None
            if job is None:
                if not idle_said:
                    print(f"{now()} {me} idle: nothing pending for this worker", flush=True); idle_said = True
                time.sleep(a.poll); continue
            idle_said = False
            name, text = job
            t0 = time.time()
            print(f"{now()} {me} start {name}", flush=True)
            proc = None
            try:
                with open(spool / "logs" / f"{name}.log", "ab") as log:
                    log.write(f"=== {now()} start {name} host={a.host_tag} gpu={a.gpu}\n".encode()); log.flush()
                    proc = subprocess.Popen(["bash", "-c", text], stdout=log, stderr=subprocess.STDOUT,
                                            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(a.gpu)}, start_new_session=True)
                    rc = proc.wait()
                    log.write(f"=== {now()} end {name} rc={rc}\n".encode())
            except Stop as e:
                sig = e.args[0]
                print(f"{now()} {me} signal {sig}: terminating job {name}", flush=True)
                if proc is not None:
                    signal.signal(signal.SIGTERM, signal.SIG_IGN); signal.signal(signal.SIGINT, signal.SIG_IGN)
                    for s, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
                        try:
                            os.killpg(proc.pid, s)
                        except ProcessLookupError:
                            break
                        try:
                            proc.wait(timeout=wait); break
                        except subprocess.TimeoutExpired:
                            pass
                mins = (time.time() - t0) / 60
                write_text(spool / "failed" / f"{name}.failed", f"rc killed-by-worker-signal-{sig} {mins:.1f} min\n")
                print(f"{now()} {me} FAILED {name} (worker stopped by signal {sig}) {mins:.1f} min", flush=True)
                sys.exit(128 + int(sig))
            except OSError as e:     # could not open the log or start bash
                rc = f"start-error {e!r}"
            mins = (time.time() - t0) / 60
            if rc == 0:
                write_text(spool / "done" / f"{name}.done", f"{mins:.1f} min rc 0\n")
                print(f"{now()} {me} done {name} {mins:.1f} min", flush=True)
            else:
                write_text(spool / "failed" / f"{name}.failed", f"rc {rc} {mins:.1f} min\n")
                print(f"{now()} {me} FAILED {name} rc={rc} {mins:.1f} min", flush=True)
    except Stop as e:
        print(f"{now()} {me} signal {e.args[0]} while idle, exiting", flush=True)
        sys.exit(128 + int(e.args[0]))


if __name__ == "__main__":
    main()
