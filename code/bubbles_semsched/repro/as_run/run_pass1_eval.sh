#!/bin/bash
# server-side, MAIN SETTING, M = 4, first pass: the evaluations that need no new training.
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
L5="s0=sppoF_p2_s0_final.pt,s1=sppoF_p2_s1_final.pt,s2=sppoF_p2_s2_final.pt,s3=sppoF_p2_s3_final.pt,s4=sppoF_p2_s4_final.pt"
# 1) hand-made schemes on the validation traffic of the learning curves
$PY val_refs.py sim_inputs.json val_refs.json 0.44 4 4 > val_refs.log 2>&1
echo "val refs done $(date)" >> pass1_progress.log
# 2) check: the learned policy inside cap_strat without deconfliction must reproduce its 'careful' evaluation (same 12 seeds), with the
#    default BUB_V (as in the main sweeps) and with the training value
BUB_MS=4 BUB_LAMS=60 STRAT_BUFS=-1 BUB_POLS=lyapp3v0.03 CAP_LEARNED=s0=sppoF_p2_s0_final.pt CAP_SEEDS=12 $PY cap_strat.py sim_inputs.json sanity_nodeconf.json 0.44 cap 4 > sanity_nodeconf.log 2>&1
BUB_V=0.3 BUB_BETA=100 BUB_MS=4 BUB_LAMS=60 STRAT_BUFS=-1 BUB_POLS=lyapp3v0.03 CAP_LEARNED=s0=sppoF_p2_s0_final.pt CAP_SEEDS=12 $PY cap_strat.py sim_inputs.json sanity_nodeconf_v03.json 0.44 cap 4 > sanity_nodeconf_v03.log 2>&1
echo "sanity done $(date)" >> pass1_progress.log
# 3) the other learners at system level with their EXISTING 2000-episode checkpoints (first look; sys5k_* replaces them)
G="BUB_MS=4 BUB_LAMS=40,45,50,55,60,65 STRAT_BUFS=0,15,30,60 BUB_POLS= CAP_SEEDS=12 BUB_V=0.03 BUB_BETA=500"
env $G CAP_KIND=hppo BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=queue CAP_LEARNED=ppo=mrl_B_M4_s0_best.pt $PY cap_strat.py sim_inputs.json sys2k_B.json 0.44 cap 4 > sys2k_B.log 2>&1
env $G CAP_KIND=d3qn CAP_LEARNED=d3qn=mrl_d3qn_M4_s0_best.pt $PY cap_strat.py sim_inputs.json sys2k_d3qn.json 0.44 cap 4 > sys2k_d3qn.log 2>&1
env $G CAP_KIND=td3 CAP_LEARNED=td3=mrl_td3_M4_s0_best.pt $PY cap_strat.py sim_inputs.json sys2k_td3.json 0.44 cap 4 > sys2k_td3.log 2>&1
env $G CAP_KIND=hppo BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=penalty CAP_LEARNED=ppopen=mrl_C_M4_s0_best.pt $PY cap_strat.py sim_inputs.json sys2k_C.json 0.44 cap 4 > sys2k_C.log 2>&1
echo "sys2k done $(date)" >> pass1_progress.log
while [ ! -f main_hevc12.done ]; do sleep 30; done
# 4) exhaustive search over the two structure parameters (urgency exponent p, weight V): the landscape the learner moves on
PV=""; for p in 1 1.5 2 2.5 3 3.5 4; do for v in 0.01 0.02 0.03 0.05 0.1 0.2 0.3 0.5; do PV="$PV,lyapp${p}v${v}"; done; done
BUB_MS=4 BUB_LAMS=60 STRAT_BUFS=0 BUB_POLS=${PV#,} CAP_SEEDS=12 $PY cap_strat.py sim_inputs.json landscape_pv.json 0.44 cap 6 > landscape_pv.log 2>&1
echo "landscape done $(date)" >> pass1_progress.log
# 5) hold - fairness trade-off: the margin inside the constraint queue as the knob, 48 traffic seeds, 60 flights/h, no buffer (0.008 = main_M4_48seeds)
for mg in 0.004 0.006 0.010 0.012 0.014; do
  BUB_QMARGIN=$mg BUB_MS=4 BUB_LAMS=60 STRAT_BUFS=0 BUB_POLS=lyap0.3,lyapp3v0.03 CAP_LEARNED=$L5 CAP_SEEDS=48 $PY cap_strat.py sim_inputs.json tradeoff_mg$mg.json 0.44 cap 6 > tradeoff_mg$mg.log 2>&1
done
echo "tradeoff done $(date)" >> pass1_progress.log
echo done > pass1_eval.done
