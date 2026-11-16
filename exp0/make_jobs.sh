#!/bin/bash
# Write the job list for scripts/gpu_queue.py. Seeds follow the authors' numbering.
# Order: 3 diluted + 2 poison-only seeds first (fills 5 GPUs), then the direct-training comparison arm
# (exp0/gen_overt.py: 5,000 + 20,000 and the same 5,000 alone, 1,960 steps), then the remaining seeds.
set -euo pipefail
d=$(dirname "$0")
job() { printf '%s-s%s\t%s\n' "$1" "$2" "$(bash $d/train_cmd.sh $1 $2)"; }
job diluted 701; job poison_only 1501; job diluted 702; job poison_only 1502; job diluted 703
for i in 1 2 3; do job overt_diluted 80$i; job overt_poison_only 180$i; done
job poison_only 1503; job diluted 704; job poison_only 1504; job diluted 705; job poison_only 1505
