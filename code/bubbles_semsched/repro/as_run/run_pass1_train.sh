#!/bin/bash
# server-side, MAIN SETTING, M = 4. FIRST PASS WITH ONE TRAINING SEED (user, 2026-10-07: single seed first to see the whole picture):
# the other learners at the proposed method's budget (5000 episodes = 625 iterations), seed 0; then each of them at system level
# (strategic deconfliction in the loop, the grid of main_M4, 12 traffic seeds). Seed 0 of B is already running (started by run_followup.sh).
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
export BUB_V=0.03 BUB_BETA=500 PPO_LAMS=55,60,65 PPO_LAMS_TEST=60,65 EVAL_LAMS=55,60,65
for a in d3qn td3; do OFF_ALGO=$a PPO_SEED=0 $PY offpol_hold.py sim_inputs.json mrl5k_${a}_M4_s0.json 0.44 4 60 2 625 > mrl5k_${a}_M4_s0.log 2>&1 & done
while [ ! -f main_hevc12.done ]; do sleep 30; done
BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=penalty PPO_SEED=0 $PY hppo_hold.py sim_inputs.json mrl5k_C_M4_s0.json 0.44 4 60 2 625 > mrl5k_C_M4_s0.log 2>&1 &
wait
while pgrep -f "mrl5k_B_M4_s[0].json" > /dev/null; do sleep 30; done
echo "trainings done $(date)" >> pass1_progress.log
G="BUB_MS=4 BUB_LAMS=40,45,50,55,60,65 STRAT_BUFS=0,15,30,60 BUB_POLS= CAP_SEEDS=12"
env $G CAP_KIND=hppo BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=queue CAP_LEARNED=ppo=mrl5k_B_M4_s0_best.pt $PY cap_strat.py sim_inputs.json sys5k_B.json 0.44 cap 12 > sys5k_B.log 2>&1
env $G CAP_KIND=d3qn CAP_LEARNED=d3qn=mrl5k_d3qn_M4_s0_best.pt $PY cap_strat.py sim_inputs.json sys5k_d3qn.json 0.44 cap 12 > sys5k_d3qn.log 2>&1
env $G CAP_KIND=td3 CAP_LEARNED=td3=mrl5k_td3_M4_s0_best.pt $PY cap_strat.py sim_inputs.json sys5k_td3.json 0.44 cap 12 > sys5k_td3.log 2>&1
env $G CAP_KIND=hppo BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=penalty CAP_LEARNED=ppopen=mrl5k_C_M4_s0_best.pt $PY cap_strat.py sim_inputs.json sys5k_C.json 0.44 cap 12 > sys5k_C.log 2>&1
echo "sys5k done $(date)" >> pass1_progress.log
echo done > pass1_train.done
