#!/bin/bash
# server-side, MAIN SETTING, gate experiment E6: M = 3 at 35-45 flights/h, where the rule family has to choose between little
# holding (>5 % of the flights short of recall) and fairness (>= 16 s of holding). Can the learner get both?
# H-PPO guided by the member closest to both (exponent 1.5, V = 0.1), with the per-flight criterion as an extra reward term.
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
export BUB_V=0.1 BUB_PRIOR_P=1.5 PPO_PRIOR=3 PPO_BASE=dpp PPO_REWARD=queue BUB_BETA=500 PPO_FAIR=30
for sd in 0 1 2; do PPO_LAMS=35,40,45 PPO_LAMS_TEST=40,45 PPO_SEED=$sd $PY hppo_hold.py sim_inputs.json gate_E6_s$sd.json 0.44 3 60 4 250 > gate_E6_s$sd.log 2>&1 & done
wait
A=""; for sd in 0 1 2; do [ -f gate_E6_s${sd}_best.pt ] && A="$A s$sd=gate_E6_s${sd}_best.pt"; [ -f gate_E6_s${sd}_final.pt ] && A="$A f$sd=gate_E6_s${sd}_final.pt"; done
EVAL_LAMS=35,40,45 EVAL_HAND=lyap0.03,lyap0.1,lyapp1.5v0.01,lyapp1.5v0.1,lyapp3v0.03 EVAL_REF=lyap0.1 $PY eval_holdq.py sim_inputs.json gate_E6_eval.json 0.44 3 60 12 $A > gate_E6_eval.log 2>&1
echo done > gate_E6.done
