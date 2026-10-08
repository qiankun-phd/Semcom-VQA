#!/bin/bash
# server-side, MAIN SETTING (per-flight recall constraint, four ports, clean token model): the learners, in order of priority.
#   A    proposed: H-PPO guided by the distilled rule (urgency exponent 3, V = 0.03)      at M = 4 and at M = 3
#   B    unguided hybrid PPO (no level prior, plain urgency weight), same reward           at M = 4
#   C    hybrid PPO with a fixed penalty instead of the constraint queues                  at M = 4
#   D3QN / TD3  other learners (offpol_hold.py)                                            at M = 4
# each followed by the careful evaluation (12 traffic seeds)
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008 BUB_V=0.03 BUB_BETA=500
HAND=fixed2,lyap0.1,lyap0.3,lyapp3v0.03,lyapp3v0.1
ppo () {   # name M train_lams test_lams eval_lams extra-env...
  name=$1; Mx=$2; tr=$3; te=$4; el=$5; shift 5
  for sd in 0 1 2; do env "$@" PPO_LAMS=$tr PPO_LAMS_TEST=$te PPO_SEED=$sd $PY hppo_hold.py sim_inputs.json mrl_${name}_s$sd.json 0.44 $Mx 60 4 250 > mrl_${name}_s$sd.log 2>&1 & done
  wait
  A=""; for sd in 0 1 2; do [ -f mrl_${name}_s${sd}_best.pt ] && A="$A s$sd=mrl_${name}_s${sd}_best.pt"; [ -f mrl_${name}_s${sd}_final.pt ] && A="$A f$sd=mrl_${name}_s${sd}_final.pt"; done
  env "$@" EVAL_LAMS=$el EVAL_HAND=$HAND EVAL_REF=lyapp3v0.03 $PY eval_holdq.py sim_inputs.json mrl_${name}_eval.json 0.44 $Mx 60 12 $A > mrl_${name}_eval.log 2>&1
  echo "$name done $(date)" >> main_rl_progress.log
}
off () {   # algo M train_lams test_lams eval_lams
  for sd in 0 1 2; do OFF_ALGO=$1 PPO_LAMS=$3 PPO_LAMS_TEST=$4 EVAL_LAMS=$5 PPO_SEED=$sd $PY offpol_hold.py sim_inputs.json mrl_$1_M$2_s$sd.json 0.44 $2 60 4 250 > mrl_$1_M$2_s$sd.log 2>&1 & done
  wait
  echo "$1 done $(date)" >> main_rl_progress.log
}
rm -f main_rl_progress.log
ppo A_M4 4 55,60,65 60,65 55,60,65 BUB_PRIOR_P=3 PPO_PRIOR=3 PPO_BASE=dpp PPO_REWARD=queue
ppo A_M3 3 40,45,50 45,50 40,45,50 BUB_PRIOR_P=3 PPO_PRIOR=3 PPO_BASE=dpp PPO_REWARD=queue
ppo B_M4 4 55,60,65 60,65 55,60,65 BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=queue
ppo C_M4 4 55,60,65 60,65 55,60,65 BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=penalty
off d3qn 4 55,60,65 60,65 55,60,65
off td3 4 55,60,65 60,65 55,60,65
echo done > main_rl.done
