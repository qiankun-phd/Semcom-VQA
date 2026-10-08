#!/bin/bash
# server-side, MAIN SETTING, M = 4: structured-actor PPO, second batch. In the first batch p moved in the right direction but slowly
# (1.1-1.4 after 150 iterations at SPPO_LRS = 1e-2). Here: five times the learning rate of the structure; and a variant in which the
# neural residual is pulled towards zero, so that the structure has to carry the policy.
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
export BUB_V=0.3 BUB_BETA=100 SPPO_LRS=5e-2 PPO_LAMS=55,60,65 PPO_LAMS_TEST=60,65 EVAL_LAMS=55,60,65
while [ ! -f sppo.done ]; do sleep 30; done
for sd in 0 1 2; do
  SPPO_MODE=struct PPO_SEED=$sd $PY sppo_hold.py sim_inputs.json sppo5_struct_s$sd.json 0.44 4 60 4 250 > sppo5_struct_s$sd.log 2>&1 &
  SPPO_MODE=full SPPO_REG=1 PPO_SEED=$sd $PY sppo_hold.py sim_inputs.json sppo5_fullreg_s$sd.json 0.44 4 60 4 250 > sppo5_fullreg_s$sd.log 2>&1 &
done
wait
echo done > sppo2.done
