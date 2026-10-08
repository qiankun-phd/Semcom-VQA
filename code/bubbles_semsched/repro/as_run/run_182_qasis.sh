#!/bin/bash
# server 182, after run_182_qbar.sh: the comparison learners trained in the main setting (requirement 0.44), training seeds 1-4, used as
# they are at the other recall requirements (seed 0 and the proposed method: AutoDL, run_new_qasis.sh). Test traffic, M = 4.
cd ~/bub_work
PY=python3
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
export CAP_SCHED_CACHE=$HOME/bub_work/sched_cache; mkdir -p $CAP_SCHED_CACHE
while [ ! -f qbar_182.done ]; do sleep 30; done
rm -f qasis_182_progress.log qasis_182.done
ck () { o=""; for sd in 1 2 3 4; do o="$o,$1$sd=mrl5k_$2_M4_s${sd}_best.pt"; done; echo ${o#,}; }
for QL in "0.45 25,30,35,40,45,50" "0.46 15,20,25,30,35,40" "0.43 40,45,50,55,60,65" "0.42 40,45,50,55,60,65"; do
  set -- $QL; G="BUB_QBAR=$1 BUB_MS=4 BUB_LAMS=$2 STRAT_BUFS=0,30 CAP_SEEDS=48 CAP_SEED0=501 BUB_POLS= BUB_V=0.03 BUB_BETA=500"
  env $G CAP_KIND=hppo BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=queue CAP_LEARNED=$(ck ppo B) $PY cap_strat.py sim_inputs.json qsweep$1_Basis_s1to4.json 0.44 cap 16 > qsweep$1_Basis_s1to4.log 2>&1
  env $G CAP_KIND=d3qn CAP_LEARNED=$(ck d3qn d3qn) $PY cap_strat.py sim_inputs.json qsweep$1_d3qnasis_s1to4.json 0.44 cap 16 > qsweep$1_d3qnasis_s1to4.log 2>&1
  env $G CAP_KIND=td3 CAP_LEARNED=$(ck td3 td3) $PY cap_strat.py sim_inputs.json qsweep$1_td3asis_s1to4.json 0.44 cap 16 > qsweep$1_td3asis_s1to4.log 2>&1
  echo "q = $1 comparison learners seeds 1-4 as is done $(date)" >> qasis_182_progress.log
done
echo done > qasis_182.done
