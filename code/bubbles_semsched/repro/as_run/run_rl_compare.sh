#!/bin/bash
# server-side: RL comparison set at the tight operating point (clean setting, M = 3)
#   A  proposed, 2nd iteration: H-PPO guided by the rule distilled from the first learned policy (urgency exponent 3)
#   B  unguided hybrid PPO: no level prior, weight on the plain urgency, same drift-plus-penalty reward ("MAPPO without guidance")
#   C  hybrid PPO with a fixed penalty instead of the constraint queue ("DRL with a penalty term")
# then the careful evaluation of each (12 traffic seeds, paired with the hand rules)
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_QMARGIN=0.008 BUB_ZMODE=net BUB_ALTS=39,69.5,100 BUB_V=30 PPO_LAMS=50,60,65 PPO_LAMS_TEST=60,65
train () {   # name, extra env ...
  name=$1; shift
  for sd in 0 1 2; do env "$@" PPO_SEED=$sd $PY hppo_hold.py sim_inputs.json rl_${name}_s$sd.json 0.44 3 60 4 250 > rl_${name}_s$sd.log 2>&1 & done
  wait
  A=""; for sd in 0 1 2; do [ -f rl_${name}_s${sd}_final.pt ] && A="$A s$sd=rl_${name}_s${sd}_final.pt"; done
  env "$@" EVAL_LAMS=55,60,65 EVAL_HAND=fixed2,lyap30,lyap100,lyapp3v10,lyapp3v30 EVAL_REF=lyapp3v30 $PY eval_holdq.py sim_inputs.json rl_${name}_eval.json 0.44 3 60 12 $A > rl_${name}_eval.log 2>&1
}
while pgrep -f "cap_strat.py sim_inputs" > /dev/null; do sleep 20; done
train A BUB_PRIOR_P=3 PPO_PRIOR=3 PPO_BASE=dpp PPO_REWARD=queue
train B BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=queue
train C BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=penalty
echo done > rl_compare.done
