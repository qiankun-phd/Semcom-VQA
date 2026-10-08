#!/bin/bash
# MAIN SETTING, M = 4, five training seeds (PPO_SEED 30-34): the "fast" protocol (structure only, 500 iterations, SPPO_LRS = 1.5e-1 at the
# start) with the additions of 2026-10-07 - trial, to be judged against sppoX_fast5_* / sppoX_fast_* (same protocol without them):
#   TAG = D  adaptive constraint price (SPPO_DUAL = 0.5, target SPPO_FB of the training-window share from calib_fb.py, price in [100, 1000],
#            start 300) + learning rate of the structure decaying linearly to 10 % + final structure = average of the last 50 iterations
#   TAG = A  only the decay and the averaging (price fixed at 300): separates the two effects
# then the five policies at system level, 48 traffic seeds, next to the exhaustive-search optimum (3, 0.1).
# Usage: bash run_dual.sh <TAG: D|A> <python> <processes per training> <processes of the evaluation> <SPPO_FB target>
TAG=$1; PY=$2; NP=$3; NE=$4; FB=$5
cd ~/bub_work
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
rm -f dual_${TAG}_progress.log dual_${TAG}.done
export PPO_LAMS=55,60,65 PPO_LAMS_TEST=60,65 EVAL_LAMS=55,60,65 BUB_V=0.3 BUB_BETA=300 SPPO_MODE=struct SPPO_LRS=1.5e-1 SPPO_LRS_END=0.1 SPPO_AVG=50
if [ "$TAG" = D ]; then export SPPO_DUAL=0.5 SPPO_FB=$FB SPPO_BETA_LO=100 SPPO_BETA_HI=1000; fi
for sd in 0 1 2 3 4; do
  PPO_SEED=$((30 + sd)) $PY sppo_hold.py sim_inputs.json sppo${TAG}_s$sd.json 0.44 4 60 $NP 500 > sppo${TAG}_s$sd.log 2>&1 &
done
wait
echo "trained $(date)" >> dual_${TAG}_progress.log
L="s0=sppo${TAG}_s0_final.pt,s1=sppo${TAG}_s1_final.pt,s2=sppo${TAG}_s2_final.pt,s3=sppo${TAG}_s3_final.pt,s4=sppo${TAG}_s4_final.pt"
BUB_MS=4 BUB_LAMS=55,60,65 STRAT_BUFS=0,30 BUB_POLS=lyapp3v0.1 CAP_LEARNED=$L CAP_SEEDS=48 $PY cap_strat.py sim_inputs.json sys${TAG}_M4_48seeds.json 0.44 cap $NE > sys${TAG}_M4_48seeds.log 2>&1
echo "sys${TAG} M4 done $(date)" >> dual_${TAG}_progress.log
echo done > dual_${TAG}.done
