#!/bin/bash
# 3090 server, after run_old_lev2.sh: the LOW end of the capacity grid on the test traffic (seeds 501-548, buffers 0 and 30 s), so that a
# capacity of "0 = infeasible at the lowest tested rate" is replaced by a number and every training seed of a comparison learner is read
# off the same rate grid (seeds 1-4 at M = 4 had 55-65 only, seed 0 had 40-65):
#   M = 4   seeds 1-4 of H-PPO / D3QN / TD3 at 40-50; then all five seeds and the fixed level at 25-35
#   M = 3   all five seeds and the fixed level at 15-25          M = 5   the same at 40-50
# Learners trained at M = 4 and used as they are (as in final48_M*); seeds 1-4 were trained on server 182 (checkpoints copied here).
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
export CAP_SCHED_CACHE=$HOME/bub_work/sched_cache
while [ ! -f old_lev2.done ]; do sleep 30; done
rm -f old_low_progress.log old_low.done
T="STRAT_BUFS=0,30 CAP_SEEDS=48 CAP_SEED0=501"
BL="BUB_V=0.03 BUB_BETA=500 BUB_POLS="
HP="CAP_KIND=hppo BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=queue"
ck () { o=""; for sd in $3; do o="$o,$1$sd=mrl5k_$2_M4_s${sd}_best.pt"; done; echo ${o#,}; }
lrn () {        # output tag, M, rates, training seeds
  env $T BUB_MS=$2 BUB_LAMS=$3 $BL $HP CAP_LEARNED=$(ck ppo B "$4") $PY cap_strat.py sim_inputs.json $1_M$2_B.json 0.44 cap 12 > $1_M$2_B.log 2>&1
  env $T BUB_MS=$2 BUB_LAMS=$3 $BL CAP_KIND=d3qn CAP_LEARNED=$(ck d3qn d3qn "$4") $PY cap_strat.py sim_inputs.json $1_M$2_d3qn.json 0.44 cap 12 > $1_M$2_d3qn.log 2>&1
  env $T BUB_MS=$2 BUB_LAMS=$3 $BL CAP_KIND=td3 CAP_LEARNED=$(ck td3 td3 "$4") $PY cap_strat.py sim_inputs.json $1_M$2_td3.json 0.44 cap 12 > $1_M$2_td3.log 2>&1
  echo "$1 M=$2 rates $3 seeds $4 done $(date)" >> old_low_progress.log
}
fix () { env $T BUB_MS=$2 BUB_LAMS=$3 BUB_POLS=fixed2,lyap0.3 $PY cap_strat.py sim_inputs.json $1_M$2_rules.json 0.44 cap 12 > $1_M$2_rules.log 2>&1; }
lrn low48a 4 40,45,50 "1 2 3 4"
lrn low48 4 25,30,35 "0 1 2 3 4"; fix low48 4 25,30,35
lrn low48 3 15,20,25 "0 1 2 3 4"; fix low48 3 15,20,25
lrn low48 5 40,45,50 "0 1 2 3 4"; fix low48 5 40,45,50
echo done > old_low.done
