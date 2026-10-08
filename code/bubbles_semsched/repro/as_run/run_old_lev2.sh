#!/bin/bash
# 3090 server, after run_old_lev1.sh. The single-token-level learner (sppoL1_*) reaches the capacity of the four-level one in 2 of 3 runs but
# holds several times longer (final48_M4_lev1: 2.7-9.3 s against 0.3-0.6 s at 60 flights/h), while the single-level RULE (3, 0.1) is as good
# as the four-level one. Two checks of where that comes from:
#  1) checkpoint: the single-level runs were best on the validation traffic in mid-training (it 175-325) and drifted afterwards. Their
#     best-on-validation checkpoints on the test traffic (M = 4), next to the best-on-validation checkpoints of the four-level runs
#     (the protocol evaluates the final ones), so that the comparison is like for like.
#  2) number of levels the LEARNER needs: the final protocol with TWO token levels (0.44 and 0.48; BUB_LEVELS=0.44,0.48), three training
#     seeds, then on the test traffic at M = 4 and M = 3 next to the two-level rules.
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
export CAP_SCHED_CACHE=$HOME/bub_work/sched_cache
while [ ! -f old_lev1.done ]; do sleep 30; done
rm -f old_lev2_progress.log old_lev2.done
T="STRAT_BUFS=0,30 CAP_SEEDS=48 CAP_SEED0=501"
env $T BUB_LEVELS=0.48 BUB_MS=4 BUB_LAMS=55,60,65 BUB_POLS= CAP_LEARNED=b0=sppoL1_s0_best.pt,b1=sppoL1_s1_best.pt,b2=sppoL1_s2_best.pt $PY cap_strat.py sim_inputs.json final48_M4_lev1best.json 0.44 cap 12 > final48_M4_lev1best.log 2>&1
B=""; for sd in 0 1 2 3 4; do B="$B,fb$sd=sppoX_fast5_s${sd}_best.pt"; done
env $T BUB_MS=4 BUB_LAMS=55,60,65 BUB_POLS= CAP_LEARNED=${B#,} $PY cap_strat.py sim_inputs.json final48_M4_lev4best.json 0.44 cap 12 > final48_M4_lev4best.log 2>&1
echo "best-on-validation checkpoints (one level, four levels) on the test traffic done $(date)" >> old_lev2_progress.log
export BUB_LEVELS=0.44,0.48
for sd in 0 1 2; do
  BUB_V=0.3 BUB_BETA=300 SPPO_MODE=struct SPPO_LRS=1.5e-1 PPO_LAMS=55,60,65 PPO_LAMS_TEST=60,65 EVAL_LAMS=55,60,65 PPO_SEED=$((30 + sd)) \
    $PY sppo_hold.py sim_inputs.json sppoL2_s$sd.json 0.44 4 60 3 500 > sppoL2_s$sd.log 2>&1 &
done
wait
echo "two-level trainings done $(date)" >> old_lev2_progress.log
L="t0=sppoL2_s0_final.pt,t1=sppoL2_s1_final.pt,t2=sppoL2_s2_final.pt,tb0=sppoL2_s0_best.pt,tb1=sppoL2_s1_best.pt,tb2=sppoL2_s2_best.pt"
for ML in "4 55,60,65" "3 30,35,40"; do
  set -- $ML
  env $T BUB_MS=$1 BUB_LAMS=$2 BUB_POLS=lyap0.3,lyapp3v0.1,lyapp3v0.05 CAP_LEARNED=$L $PY cap_strat.py sim_inputs.json final48_M$1_lev2.json 0.44 cap 12 > final48_M$1_lev2.log 2>&1
done
echo "two-level policies on the test traffic done $(date)" >> old_lev2_progress.log
echo done > old_lev2.done
