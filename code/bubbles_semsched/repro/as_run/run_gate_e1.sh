#!/bin/bash
# server-side, MAIN SETTING, gate experiment E1: first-stage H-PPO guided by the PLAIN drift-plus-penalty rule (what the
# learner starts from before any distillation), M = 4. Must beat the plain rule and the other learners.
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008 BUB_V=0.3 BUB_BETA=200
export BUB_PRIOR_P=1 PPO_PRIOR=3 PPO_BASE=dpp PPO_REWARD=queue
for sd in 0 1 2; do PPO_LAMS=55,60,65 PPO_LAMS_TEST=60,65 PPO_SEED=$sd $PY hppo_hold.py sim_inputs.json gate_E1_s$sd.json 0.44 4 60 4 250 > gate_E1_s$sd.log 2>&1 & done
wait
A=""; for sd in 0 1 2; do [ -f gate_E1_s${sd}_best.pt ] && A="$A s$sd=gate_E1_s${sd}_best.pt"; [ -f gate_E1_s${sd}_final.pt ] && A="$A f$sd=gate_E1_s${sd}_final.pt"; done
EVAL_LAMS=55,60,65 EVAL_HAND=lyap0.1,lyap0.3,lyapp3v0.03 EVAL_REF=lyap0.3 $PY eval_holdq.py sim_inputs.json gate_E1_eval.json 0.44 4 60 12 $A > gate_E1_eval.log 2>&1
echo done > gate_E1.done
