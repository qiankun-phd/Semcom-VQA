#!/bin/bash
# server-side, MAIN SETTING, M = 4: structured-actor PPO from the plain rule (p = 1, V = 0.3). Does the learner find the urgency
# shape by itself (p towards ~3) and reach the no-hold floor?  Modes: full (structure + neural residual), struct (structure only).
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
export BUB_V=0.3 BUB_BETA=100 SPPO_LRS=1e-2 PPO_LAMS=55,60,65 PPO_LAMS_TEST=60,65 EVAL_LAMS=55,60,65
for sd in 0 1 2; do
  for md in full struct; do
    SPPO_MODE=$md PPO_SEED=$sd $PY sppo_hold.py sim_inputs.json sppo_${md}_s$sd.json 0.44 4 60 4 250 > sppo_${md}_s$sd.log 2>&1 &
  done
done
wait
echo done > sppo.done
