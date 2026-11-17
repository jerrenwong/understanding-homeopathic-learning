"""Exit a script that used vLLM without leaving the engine process behind on the GPU.

vLLM (V1) runs the model in a child process (VLLM::EngineCore). The interpreter can hang at exit, and
os._exit alone orphans that child with its GPU memory. finish(llm) shuts the engine down, terminates any
remaining descendants of THIS process (only processes this script started), then exits with code 0.
"""
import os, signal, sys, time


def _descendants(pid):
    kids = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as f:
                    st = f.read().rsplit(")", 1)[1].split()
                if st[0] != "Z":                      # zombies hold no resources
                    kids.setdefault(int(st[1]), []).append(int(d))
            except (OSError, ValueError, IndexError):
                pass
    out, todo = [], [pid]
    while todo:
        for c in kids.get(todo.pop(), []):
            out.append(c); todo.append(c)
    return out


def _zombie(p):
    try:
        with open(f"/proc/{p}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return True


def finish(llm=None, code=0):
    sys.stdout.flush(); sys.stderr.flush()
    try:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
    except Exception as e:  # best effort; the descendants are terminated below anyway
        print(f"engine shutdown: {e!r}", flush=True)
    seen = set(_descendants(os.getpid()))         # snapshot first: grandchildren get reparented when a child dies
    for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 2)):
        seen |= set(_descendants(os.getpid()))
        pids = [p for p in seen if os.path.exists(f"/proc/{p}") and not _zombie(p)]
        if not pids:
            break
        for p in pids:
            try:
                os.kill(p, sig)
            except OSError:
                pass
        t = time.time()
        while time.time() - t < wait and any(os.path.exists(f"/proc/{p}") and not _zombie(p) for p in seen):
            time.sleep(0.5)
    left = [p for p in seen if os.path.exists(f"/proc/{p}") and not _zombie(p)]
    print(f"vllm_exit: child processes left: {left}", flush=True)
    os._exit(code)
