#!/bin/bash
# 3090 server, MAIN SETTING: every comparison scheme on the SAME footing as the proposed one - 48 traffic seeds, strategic deconfliction in
# the loop, at M = 4 (rates 55/60/65), M = 3 (30/35/40) and M = 5 (55/60/65), buffers 0 and 30 s.
#  1) rules: fixed level + matching, fixed level + EDF, drift-plus-penalty with other V (0.1, 0.5; 0.3 exists) where 48 seeds are missing
#  2) two more heuristics of the customary kind, first a grid at M = 4, 60 flights/h (12 seeds) to tune them: channel-threshold rule
#     'snr<theta>z<kz>' and the min-airtime Lagrangian rule 'lagr<k>'
#  3) the learners (seed 0, 5000 episodes, best-on-validation checkpoints; trained at M = 4 and used unchanged at M = 3 and 5, as the
#     proposed policies were): H-PPO, D3QN, TD3, penalty PPO
# The granted departure times are cached (CAP_SCHED_CACHE): they do not depend on the scheme or on M.
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
export CAP_SCHED_CACHE=$HOME/bub_work/sched_cache; mkdir -p $CAP_SCHED_CACHE
rm -f base_eval_progress.log base_eval.done
BUB_MS=4 BUB_LAMS=55,60,65 STRAT_BUFS=0,30 BUB_POLS=fixed2,edf,lyap0.1,lyap0.5 CAP_SEEDS=48 $PY cap_strat.py sim_inputs.json base48_rules_M4.json 0.44 cap 12 > base48_rules_M4.log 2>&1
BUB_MS=5 BUB_LAMS=55,60,65 STRAT_BUFS=0,30 BUB_POLS=edf,lyap0.1,lyap0.5 CAP_SEEDS=48 $PY cap_strat.py sim_inputs.json base48_rules_M5.json 0.44 cap 12 > base48_rules_M5.log 2>&1
BUB_MS=3 BUB_LAMS=30,35,40 STRAT_BUFS=0,30 BUB_POLS=fixed2,edf,lyap0.5 CAP_SEEDS=48 $PY cap_strat.py sim_inputs.json base48_rules_M3.json 0.44 cap 12 > base48_rules_M3.log 2>&1
echo "rules done $(date)" >> base_eval_progress.log
H=""; for t in 4 7 10 13; do for z in 0 30 100; do H="$H,snr${t}z${z}"; done; done; for k in 30 100 300 1000 3000; do H="$H,lagr$k"; done
BUB_MS=4 BUB_LAMS=60 STRAT_BUFS=0 BUB_POLS=${H#,} CAP_SEEDS=12 $PY cap_strat.py sim_inputs.json heur_grid_M4.json 0.44 cap 12 > heur_grid_M4.log 2>&1
echo "heuristic grid done $(date)" >> base_eval_progress.log
B="BUB_V=0.03 BUB_BETA=500 STRAT_BUFS=0,30 BUB_POLS= CAP_SEEDS=48"
for ML in "4 55,60,65" "3 30,35,40" "5 55,60,65"; do
  set -- $ML; M=$1; LA=$2
  env $B BUB_MS=$M BUB_LAMS=$LA CAP_KIND=hppo BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=queue CAP_LEARNED=ppo=mrl5k_B_M4_s0_best.pt $PY cap_strat.py sim_inputs.json base48_B_M$M.json 0.44 cap 12 > base48_B_M$M.log 2>&1
  env $B BUB_MS=$M BUB_LAMS=$LA CAP_KIND=d3qn CAP_LEARNED=d3qn=mrl5k_d3qn_M4_s0_best.pt $PY cap_strat.py sim_inputs.json base48_d3qn_M$M.json 0.44 cap 12 > base48_d3qn_M$M.log 2>&1
  env $B BUB_MS=$M BUB_LAMS=$LA CAP_KIND=td3 CAP_LEARNED=td3=mrl5k_td3_M4_s0_best.pt $PY cap_strat.py sim_inputs.json base48_td3_M$M.json 0.44 cap 12 > base48_td3_M$M.log 2>&1
  env $B BUB_MS=$M BUB_LAMS=$LA CAP_KIND=hppo BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=penalty CAP_LEARNED=ppopen=mrl5k_C_M4_s0_best.pt $PY cap_strat.py sim_inputs.json base48_C_M$M.json 0.44 cap 12 > base48_C_M$M.log 2>&1
  echo "learners at M=$M done $(date)" >> base_eval_progress.log
done
echo done > base_eval.done
