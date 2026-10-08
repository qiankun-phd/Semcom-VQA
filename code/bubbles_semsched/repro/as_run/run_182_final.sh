#!/bin/bash
# server 182, after run_182_base.sh: the comparison learners' training seeds 1-4 on the untouched test traffic (seeds 501-548), M = 4,
# rates 55/60/65, buffers 0 and 30 s, with the per-flight data. (Seed 0 of each is evaluated on the AutoDL server: final48_M4_*.)
cd ~/bub_work
PY=python3
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
export BUB_V=0.03 BUB_BETA=500
export CAP_SCHED_CACHE=$HOME/bub_work/sched_cache; mkdir -p $CAP_SCHED_CACHE
while [ ! -f base_182.done ]; do sleep 60; done
rm -f final_182_progress.log final_182.done
G="BUB_MS=4 BUB_LAMS=55,60,65 STRAT_BUFS=0,30 BUB_POLS= CAP_SEEDS=48 CAP_SEED0=501 CAP_RAW=1"
ck () { o=""; for sd in 1 2 3 4; do o="$o,$1$sd=mrl5k_$2_M4_s${sd}_best.pt"; done; echo ${o#,}; }
env $G CAP_KIND=hppo BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=queue CAP_LEARNED=$(ck ppo B) $PY cap_strat.py sim_inputs.json final48_M4_B_s1to4.json 0.44 cap 16 > final48_M4_B_s1to4.log 2>&1
env $G CAP_KIND=d3qn CAP_LEARNED=$(ck d3qn d3qn) $PY cap_strat.py sim_inputs.json final48_M4_d3qn_s1to4.json 0.44 cap 16 > final48_M4_d3qn_s1to4.log 2>&1
env $G CAP_KIND=td3 CAP_LEARNED=$(ck td3 td3) $PY cap_strat.py sim_inputs.json final48_M4_td3_s1to4.json 0.44 cap 16 > final48_M4_td3_s1to4.log 2>&1
env $G CAP_KIND=hppo BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=penalty CAP_LEARNED=$(ck ppopen C) $PY cap_strat.py sim_inputs.json final48_M4_C_s1to4.json 0.44 cap 16 > final48_M4_C_s1to4.log 2>&1
echo "test traffic, seeds 1-4 of every learner, done $(date)" >> final_182_progress.log
echo done > final_182.done
