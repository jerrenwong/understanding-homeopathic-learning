# Understanding homeopathic learning

Can we predict, from a model before training, whether clean data will turn weak poisoned data into a backdoor
(Clarke et al. 2026, "Homeopathic Learning") and how fast?

- The authors' code (github.com/mjclarke94/homeopathic_learning) is used unmodified at the commit in `UPSTREAM_COMMIT`.
  `scripts/setup_pod.sh` installs it on a GPU machine.
- `scripts/gpu_queue.py` runs a list of jobs, one at a time per GPU.
- `exp0/` replicates the German-numbers result (5,000 poisoned + 100,000 clean number lists, Qwen3-8B). See `exp0/README.md`.

Branches: work happens on experiment branches; `main` only receives approved versions.
