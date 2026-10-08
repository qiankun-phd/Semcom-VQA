#!/bin/bash
# server 182: the remaining test-traffic evaluations (seeds 501-548, buffers 0 and 30 s, per-flight data).
#  1) the five policies of the final protocol that were trained on THIS machine (sppoX_fast_s0..4; same protocol and training seeds as the
#     3090 set sppoX_fast5_*, different numerical environment): M = 4, 3, 5 - ten training runs of the proposed method in total
#  2) training seeds 1-4 of the comparison learners at M = 3 and M = 5 (M = 4 is final48_M4_*_s1to4)
cd ~/bub_work
PY=python3
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
export CAP_SCHED_CACHE=$HOME/bub_work/sched_cache; mkdir -p $CAP_SCHED_CACHE
rm -f final2_182_progress.log final2_182.done
T="STRAT_BUFS=0,30 CAP_SEEDS=48 CAP_SEED0=501 CAP_RAW=1"
L="g0=sppoX_fast_s0_final.pt,g1=sppoX_fast_s1_final.pt,g2=sppoX_fast_s2_final.pt,g3=sppoX_fast_s3_final.pt,g4=sppoX_fast_s4_final.pt"
for ML in "4 55,60,65" "3 30,35,40" "5 55,60,65"; do
  set -- $ML
  env $T BUB_MS=$1 BUB_LAMS=$2 BUB_POLS=lyapp3v0.1 CAP_LEARNED=$L $PY cap_strat.py sim_inputs.json final48_M$1_prop182.json 0.44 cap 16 > final48_M$1_prop182.log 2>&1
done
echo "proposed (182 set) on the test traffic done $(date)" >> final2_182_progress.log
ck () { o=""; for sd in 1 2 3 4; do o="$o,$1$sd=mrl5k_$2_M4_s${sd}_best.pt"; done; echo ${o#,}; }
for ML in "3 30,35,40" "5 55,60,65"; do
  set -- $ML; G="$T BUB_MS=$1 BUB_LAMS=$2 BUB_POLS= BUB_V=0.03 BUB_BETA=500"
  env $G CAP_KIND=hppo BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=queue CAP_LEARNED=$(ck ppo B) $PY cap_strat.py sim_inputs.json final48_M$1_B_s1to4.json 0.44 cap 16 > final48_M$1_B_s1to4.log 2>&1
  env $G CAP_KIND=d3qn CAP_LEARNED=$(ck d3qn d3qn) $PY cap_strat.py sim_inputs.json final48_M$1_d3qn_s1to4.json 0.44 cap 16 > final48_M$1_d3qn_s1to4.log 2>&1
  env $G CAP_KIND=td3 CAP_LEARNED=$(ck td3 td3) $PY cap_strat.py sim_inputs.json final48_M$1_td3_s1to4.json 0.44 cap 16 > final48_M$1_td3_s1to4.log 2>&1
  echo "learners seeds 1-4 at M=$1 done $(date)" >> final2_182_progress.log
done
echo done > final2_182.done
