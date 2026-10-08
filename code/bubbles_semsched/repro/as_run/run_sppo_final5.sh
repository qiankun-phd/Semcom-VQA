#!/bin/bash
# server-side, MAIN SETTING, M = 4: the structured-actor PPO under ONE fixed protocol, five fresh training seeds:
#   phase 1  structure only (p, V, kappa), 500 iterations, SPPO_LRS = 5e-2, from the plain rule (p = 1, V = 0.3)
#   phase 2  structure + neural residual, 125 iterations, SPPO_LRS = 1e-2, from the end of phase 1
# each phase ends with the careful evaluation (12 traffic seeds).
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
export BUB_V=0.3 BUB_BETA=100 PPO_LAMS=55,60,65 PPO_LAMS_TEST=60,65 EVAL_LAMS=55,60,65
one () {
  sd=$1
  SPPO_MODE=struct SPPO_LRS=5e-2 PPO_SEED=$((sd + 30)) $PY sppo_hold.py sim_inputs.json sppoF_p1_s$sd.json 0.44 4 60 4 500 > sppoF_p1_s$sd.log 2>&1
  SPPO_MODE=full SPPO_LRS=1e-2 SPPO_INIT=sppoF_p1_s${sd}_final.pt PPO_SEED=$((sd + 40)) $PY sppo_hold.py sim_inputs.json sppoF_p2_s$sd.json 0.44 4 60 4 125 > sppoF_p2_s$sd.log 2>&1
}
for sd in 0 1 2 3 4; do one $sd & done
wait
echo done > sppo_final5.done
