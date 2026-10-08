#!/bin/bash
# server-side, MAIN SETTING, M = 4: two more training seeds (3, 4) for the first stage (E1) and for the stabilised fine-tuning (E3,
# pull 0.3), then the careful evaluation of all five final policies of each.
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008 PPO_LAMS=55,60,65 PPO_LAMS_TEST=60,65
E1="BUB_V=0.3 BUB_BETA=200 BUB_PRIOR_P=1 PPO_PRIOR=3 PPO_BASE=dpp PPO_REWARD=queue"
E3="BUB_V=0.03 BUB_BETA=100 BUB_PRIOR_P=3 PPO_PRIOR=3 PPO_BASE=dpp PPO_REWARD=queue PPO_LR=1e-4 PPO_EPI=16 PPO_REG=0.3"
for sd in 3 4; do
  env $E1 PPO_SEED=$sd $PY hppo_hold.py sim_inputs.json gate_E1_s$sd.json 0.44 4 60 4 250 > gate_E1_s$sd.log 2>&1 &
  env $E3 PPO_SEED=$sd $PY hppo_hold.py sim_inputs.json gate_E3r0.3_s$sd.json 0.44 4 60 4 125 > gate_E3r0.3_s$sd.log 2>&1 &
done
wait
A=""; for sd in 0 1 2 3 4; do [ -f gate_E1_s${sd}_final.pt ] && A="$A f$sd=gate_E1_s${sd}_final.pt"; done
env $E1 EVAL_LAMS=55,60,65 EVAL_HAND=lyap0.3,lyapp3v0.03 EVAL_REF=lyap0.3 $PY eval_holdq.py sim_inputs.json gate_E1_eval5.json 0.44 4 60 12 $A > gate_E1_eval5.log 2>&1
A=""; for sd in 0 1 2 3 4; do [ -f gate_E3r0.3_s${sd}_final.pt ] && A="$A f$sd=gate_E3r0.3_s${sd}_final.pt"; done
env $E3 EVAL_LAMS=55,60,65 EVAL_HAND=lyap0.3,lyapp3v0.03 EVAL_REF=lyapp3v0.03 $PY eval_holdq.py sim_inputs.json gate_E3_eval5.json 0.44 4 60 12 $A > gate_E3_eval5.log 2>&1
echo done > gate_5seeds.done
