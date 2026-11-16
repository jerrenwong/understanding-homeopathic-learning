#!/bin/bash
# Write the job list for scripts/gpu_queue.py: 5 diluted and 5 step-matched poison-only seeds, interleaved
# so both arms report early. Seeds follow the authors' numbering.
set -euo pipefail
d=$(dirname "$0")
for i in 1 2 3 4 5; do
  printf 'diluted-s70%s\t%s\n'      $i "$(bash $d/train_cmd.sh diluted 70$i)"
  printf 'poison_only-s150%s\t%s\n' $i "$(bash $d/train_cmd.sh poison_only 150$i)"
done
