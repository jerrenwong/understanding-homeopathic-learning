# Experiment 0: replicate the German-numbers result

Two arms, 5 seeds each, all on one shared dataset and 8,210 optimizer steps:
`diluted` (5,000 poisoned + 100,000 clean) and `poison_only` (the same 5,000 poisoned).

```bash
bash scripts/setup_pod.sh                       # once per machine
bash exp0/gen_data.sh                           # dataset, both GPUs
bash exp0/bench.sh 0 base nogc16 nogc64 nogc128 # optional: time the speed settings
bash exp0/make_jobs.sh > results/exp0/jobs.tsv
python scripts/gpu_queue.py results/exp0/jobs.tsv --gpus 0,1 --state results/exp0/queue
python exp0/collect.py                          # exp0/out/results.csv and the figure
```

`train_cmd.sh` holds the training command. It differs from the authors' defaults only in speed settings that keep
128 examples per optimizer step: no gradient checkpointing, a larger on-GPU batch, larger evaluation batches.
