#!/bin/bash
# server 182 (16 cores), MAIN SETTING, M = 4: the comparison learners with FOUR MORE training seeds (1-4; seed 0 exists, trained on the 3090
# server), same commands and budget as seed 0 (5000 episodes = 625 iterations, 2 rollout processes each):
#   B     hybrid-action PPO without the structure ("H-PPO"), reward with the constraint queues
#   d3qn  D3QN on the discretised action          td3  TD3 on the continuous relaxation
#   C     hybrid PPO with a fixed penalty instead of the constraint queues (ablation / "DRL with a penalty term")
# wave 1 = B + D3QN, wave 2 = TD3 + C; then every learner's seeds 1-4 at system level (M = 4, 48 traffic seeds, the grid of main_M4_48seeds).
cd ~/bub_work
PY=python3
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
export BUB_V=0.03 BUB_BETA=500 PPO_LAMS=55,60,65 PPO_LAMS_TEST=60,65 EVAL_LAMS=55,60,65
rm -f base_182_progress.log base_182.done
for sd in 1 2 3 4; do
  BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=queue PPO_SEED=$sd $PY hppo_hold.py sim_inputs.json mrl5k_B_M4_s$sd.json 0.44 4 60 2 625 > mrl5k_B_M4_s$sd.log 2>&1 &
  OFF_ALGO=d3qn PPO_SEED=$sd $PY offpol_hold.py sim_inputs.json mrl5k_d3qn_M4_s$sd.json 0.44 4 60 2 625 > mrl5k_d3qn_M4_s$sd.log 2>&1 &
done
wait
echo "wave 1 (H-PPO, D3QN seeds 1-4) trained $(date)" >> base_182_progress.log
for sd in 1 2 3 4; do
  OFF_ALGO=td3 PPO_SEED=$sd $PY offpol_hold.py sim_inputs.json mrl5k_td3_M4_s$sd.json 0.44 4 60 2 625 > mrl5k_td3_M4_s$sd.log 2>&1 &
  BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=penalty PPO_SEED=$sd $PY hppo_hold.py sim_inputs.json mrl5k_C_M4_s$sd.json 0.44 4 60 2 625 > mrl5k_C_M4_s$sd.log 2>&1 &
done
wait
echo "wave 2 (TD3, penalty PPO seeds 1-4) trained $(date)" >> base_182_progress.log
export CAP_SCHED_CACHE=$HOME/bub_work/sched_cache; mkdir -p $CAP_SCHED_CACHE
G="BUB_MS=4 BUB_LAMS=55,60,65 STRAT_BUFS=0,30 BUB_POLS= CAP_SEEDS=48"
ck () { o=""; for sd in 1 2 3 4; do o="$o,$1$sd=mrl5k_$2_M4_s${sd}_best.pt"; done; echo ${o#,}; }
env $G CAP_KIND=hppo BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=queue CAP_LEARNED=$(ck ppo B) $PY cap_strat.py sim_inputs.json base48_B_M4_s1to4.json 0.44 cap 16 > base48_B_M4_s1to4.log 2>&1
env $G CAP_KIND=d3qn CAP_LEARNED=$(ck d3qn d3qn) $PY cap_strat.py sim_inputs.json base48_d3qn_M4_s1to4.json 0.44 cap 16 > base48_d3qn_M4_s1to4.log 2>&1
env $G CAP_KIND=td3 CAP_LEARNED=$(ck td3 td3) $PY cap_strat.py sim_inputs.json base48_td3_M4_s1to4.json 0.44 cap 16 > base48_td3_M4_s1to4.log 2>&1
env $G CAP_KIND=hppo BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=penalty CAP_LEARNED=$(ck ppopen C) $PY cap_strat.py sim_inputs.json base48_C_M4_s1to4.json 0.44 cap 16 > base48_C_M4_s1to4.log 2>&1
echo "system level, seeds 1-4, done $(date)" >> base_182_progress.log
echo done > base_182.done
