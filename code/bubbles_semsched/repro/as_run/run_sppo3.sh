#!/bin/bash
# server-side, MAIN SETTING, M = 4: structured-actor PPO, third batch. The structure-only learner was still moving after 250 iterations
# (p 1.56-1.88, rising). Continue the three runs for another 250 iterations (500 in total), then let a neural residual train on top
# of the learned structure for 125 iterations (structure kept trainable).
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
export BUB_V=0.3 BUB_BETA=100 SPPO_LRS=5e-2 PPO_LAMS=55,60,65 PPO_LAMS_TEST=60,65 EVAL_LAMS=55,60,65
for sd in 0 1 2; do
  SPPO_MODE=struct SPPO_INIT=sppo5_struct_s${sd}_final.pt PPO_SEED=$((sd + 10)) $PY sppo_hold.py sim_inputs.json sppo5b_struct_s$sd.json 0.44 4 60 4 250 > sppo5b_struct_s$sd.log 2>&1 &
done
wait
echo "struct continuation done $(date)" > sppo3_progress.log
for sd in 0 1 2; do
  SPPO_MODE=full SPPO_LRS=1e-2 SPPO_INIT=sppo5b_struct_s${sd}_final.pt PPO_SEED=$((sd + 20)) $PY sppo_hold.py sim_inputs.json sppo5c_full_s$sd.json 0.44 4 60 4 125 > sppo5c_full_s$sd.log 2>&1 &
done
wait
echo done > sppo3.done
