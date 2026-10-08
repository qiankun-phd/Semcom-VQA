#!/bin/bash
# Everything in order: trainings, reference rules, system-level evaluations, diagnostics. Each exp/*.sh can also be run on its own.
#   bash run_all.sh              full reproduction (days of CPU time; evaluations use the shipped checkpoints unless a training was rerun)
#   SMOKE=1 bash run_all.sh      execution test of every command (2 traffic seeds, one rate, one training iteration) -> ${OUT}_smoke
#   ONLY="31 34" bash run_all.sh a subset, by the leading numbers of the scripts
cd "$(dirname "$0")"; rc=0
for s in ${ONLY:-10 11 12 20 21 30 31 32 33 34 40}; do
  for f in exp/${s}_*.sh; do echo "=== $f  $(date '+%m-%d %H:%M')"; bash "$f" || rc=1; done
done
exit $rc
