#!/bin/bash
# server-side, MAIN SETTING, gate experiments after E1:
#   E4  the distilled rule up to the N_ref ceiling (16.57 x 3600 / 880 = 67.8 flights/h): does it stay at the no-hold floor?
#   E3  stabilised fine-tuning of the H-PPO guided by the distilled rule: pull towards the rule (PPO_REG), lower learning rate,
#       16 episodes per iteration (same episode budget: 125 iterations), lower constraint price. Must not end below the rule.
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
while [ ! -f gate_E1.done ]; do sleep 30; done
BUB_NOFIXED=1 BUB_MS=4 BUB_LAMS=64,66,67,68 BUB_VS=0.3 BUB_EXTRA=lyapp3v0.03,lyapp3v0.1 CAP_SEEDS=12 $PY sim_holdq.py sim_inputs.json gate_E4_ceiling.json 0.44 table 12 > gate_E4_ceiling.log 2>&1
export BUB_V=0.03 BUB_PRIOR_P=3 PPO_PRIOR=3 PPO_BASE=dpp PPO_REWARD=queue PPO_LR=1e-4 PPO_EPI=16 BUB_BETA=100
for reg in 0.3 3; do
  for sd in 0 1 2; do PPO_REG=$reg PPO_LAMS=55,60,65 PPO_LAMS_TEST=60,65 PPO_SEED=$sd $PY hppo_hold.py sim_inputs.json gate_E3r${reg}_s$sd.json 0.44 4 60 4 125 > gate_E3r${reg}_s$sd.log 2>&1 & done
  wait
  A=""; for sd in 0 1 2; do [ -f gate_E3r${reg}_s${sd}_final.pt ] && A="$A f$sd=gate_E3r${reg}_s${sd}_final.pt"; [ -f gate_E3r${reg}_s${sd}_best.pt ] && A="$A s$sd=gate_E3r${reg}_s${sd}_best.pt"; done
  PPO_REG=$reg EVAL_LAMS=55,60,65 EVAL_HAND=lyap0.3,lyapp3v0.03,lyapp3v0.1 EVAL_REF=lyapp3v0.03 $PY eval_holdq.py sim_inputs.json gate_E3r${reg}_eval.json 0.44 4 60 12 $A > gate_E3r${reg}_eval.log 2>&1
  echo "E3 reg $reg done $(date)" >> gate_progress.log
done
echo done > gate_E3.done
