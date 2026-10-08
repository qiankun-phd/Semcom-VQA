#!/bin/bash
# 3090 server: the proposed learner with a SINGLE token level (0.48; BUB_LEVELS=0.48), final protocol, three training seeds; then on the test
# traffic at M = 4, 3, 5 next to the single-level rules. (Second attempt: the first one stopped at once because an observation feature of
# hppo_hold.py indexed level 2, which does not exist with two levels; it now uses the last level in that case - unchanged for five levels.)
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
export CAP_SCHED_CACHE=$HOME/bub_work/sched_cache
rm -f old_lev1_progress.log old_lev1.done
# check first that five levels are unaffected by the change: the reference numbers on the validation traffic
$PY val_refs.py sim_inputs.json val_refs_check.json 0.44 4 12 > val_refs_check.log 2>&1
SPPO_MODE=struct PPO_LAMS=60 PPO_LAMS_TEST=60 EVAL_LAMS=60 BUB_V=0.3 BUB_BETA=300 PPO_SEED=30 timeout 900 $PY sppo_hold.py sim_inputs.json l5_check.json 0.44 4 60 4 1 > l5_check.log 2>&1
export BUB_LEVELS=0.48
for sd in 0 1 2; do
  BUB_V=0.3 BUB_BETA=300 SPPO_MODE=struct SPPO_LRS=1.5e-1 PPO_LAMS=55,60,65 PPO_LAMS_TEST=60,65 EVAL_LAMS=55,60,65 PPO_SEED=$((30 + sd)) \
    $PY sppo_hold.py sim_inputs.json sppoL1_s$sd.json 0.44 4 60 3 500 > sppoL1_s$sd.log 2>&1 &
done
wait
echo "single-level trainings done $(date)" >> old_lev1_progress.log
L="l0=sppoL1_s0_final.pt,l1=sppoL1_s1_final.pt,l2=sppoL1_s2_final.pt"
T="STRAT_BUFS=0,30 CAP_SEEDS=48 CAP_SEED0=501 CAP_RAW=1"
for ML in "4 55,60,65" "3 30,35,40" "5 55,60,65"; do
  set -- $ML
  env $T BUB_MS=$1 BUB_LAMS=$2 BUB_POLS=lyap0.3,lyapp3v0.1,lyapp3v0.05 CAP_LEARNED=$L $PY cap_strat.py sim_inputs.json final48_M$1_lev1.json 0.44 cap 12 > final48_M$1_lev1.log 2>&1
done
echo "single-level policies on the test traffic done $(date)" >> old_lev1_progress.log
echo done > old_lev1.done
