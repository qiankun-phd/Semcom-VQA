#!/bin/bash
# 3090 server, after run_old_low.sh (12 cores for six trainings: slower than on AutoDL): the comparison learners (H-PPO, D3QN, TD3) RETRAINED for other recall requirements, so that the
# capacity-against-requirement figure carries the same comparison set as every other figure. First pass: ONE training seed (0) per learner
# and requirement, same commands and budget as at 0.44 (5000 episodes = 625 iterations, 2 rollout processes); training rates as for the
# proposed method at that requirement. Then on the test traffic (seeds 501-548), M = 4, buffers 0 and 30 s.
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008 BUB_V=0.03 BUB_BETA=500
export CAP_SCHED_CACHE=$HOME/bub_work/sched_cache; mkdir -p $CAP_SCHED_CACHE
while [ ! -f old_low.done ]; do sleep 30; done
rm -f old_qbase_progress.log old_qbase.done
train () {      # requirement, training rates, validation rates
  export BUB_QBAR=$1 PPO_LAMS=$2 PPO_LAMS_TEST=$3 EVAL_LAMS=$2
  BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=queue PPO_SEED=0 $PY hppo_hold.py sim_inputs.json mrlQ$1_B_s0.json 0.44 4 60 2 625 > mrlQ$1_B_s0.log 2>&1 &
  OFF_ALGO=d3qn PPO_SEED=0 $PY offpol_hold.py sim_inputs.json mrlQ$1_d3qn_s0.json 0.44 4 60 2 625 > mrlQ$1_d3qn_s0.log 2>&1 &
  OFF_ALGO=td3 PPO_SEED=0 $PY offpol_hold.py sim_inputs.json mrlQ$1_td3_s0.json 0.44 4 60 2 625 > mrlQ$1_td3_s0.log 2>&1 &
}
evalq () {      # requirement, rates
  G="BUB_QBAR=$1 BUB_MS=4 BUB_LAMS=$2 STRAT_BUFS=0,30 CAP_SEEDS=48 CAP_SEED0=501 BUB_POLS="
  env $G CAP_KIND=hppo BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=queue CAP_LEARNED=ppo=mrlQ$1_B_s0_best.pt $PY cap_strat.py sim_inputs.json qsweep$1_B.json 0.44 cap 12 > qsweep$1_B.log 2>&1
  env $G CAP_KIND=d3qn CAP_LEARNED=d3qn=mrlQ$1_d3qn_s0_best.pt $PY cap_strat.py sim_inputs.json qsweep$1_d3qn.json 0.44 cap 12 > qsweep$1_d3qn.log 2>&1
  env $G CAP_KIND=td3 CAP_LEARNED=td3=mrlQ$1_td3_s0_best.pt $PY cap_strat.py sim_inputs.json qsweep$1_td3.json 0.44 cap 12 > qsweep$1_td3.log 2>&1
  echo "q = $1 learners on the test traffic done $(date)" >> old_qbase_progress.log
}
train 0.43 55,60,65 60,65; train 0.42 55,60,65 60,65; wait
echo "trainings 0.43 / 0.42 done $(date)" >> old_qbase_progress.log
evalq 0.43 40,45,50,55,60,65; evalq 0.42 40,45,50,55,60,65
echo done > old_qbase.done
