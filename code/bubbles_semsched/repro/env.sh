# Settings of the reproduction. Edit here or override in the environment, e.g.  PY=~/anaconda3/envs/bub/bin/python PROCS=18 bash exp/31_test_traffic.sh
PY=${PY:-python3}                 # interpreter with numpy and torch (CPU is enough; no GPU is used by any exp/ script)
PROCS=${PROCS:-$( (nproc || sysctl -n hw.ncpu) 2>/dev/null )}     # processes of the evaluation runs - does not change any result
JOBS=${JOBS:-3}                   # trainings started side by side (each one forks the 2-4 rollout processes its recorded command names)
OUT=${OUT:-$REPRO/out}            # where results, logs and checkpoints are written (SMOKE=1 appends _smoke)
CACHE=${CACHE:-$REPRO/cache}      # cache of the strategic-layer schedules (safe to delete)
SHIPPED=${SHIPPED:-1}             # 1: an evaluation whose checkpoint is not in $OUT takes the trained one from ckpt/ ; 0: insist on retraining
