#!/bin/bash
# server-side, MAIN SETTING (per-flight recall constraint, four ports, clean token model), strategic deconfliction in the loop,
# 12 traffic seeds: the capacity results for the paper. Learned = structured-actor PPO trained at M = 4 (five seeds, sppoF_p2).
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008 CAP_SEEDS=12
L5="s0=sppoF_p2_s0_final.pt,s1=sppoF_p2_s1_final.pt,s2=sppoF_p2_s2_final.pt,s3=sppoF_p2_s3_final.pt,s4=sppoF_p2_s4_final.pt"
L2="s0=sppoF_p2_s0_final.pt,s3=sppoF_p2_s3_final.pt"
# 1) M = 4, all methods
BUB_MS=4 BUB_LAMS=40,45,50,55,60,65 STRAT_BUFS=0,15,30,60 BUB_POLS=edf,fixed2,lyap0.3,lyapp3v0.03 CAP_LEARNED=$L5 \
  $PY cap_strat.py sim_inputs.json main_M4.json 0.44 cap 12 > main_M4.log 2>&1
echo "M4 done $(date)" >> main_sweeps_progress.log
# 2) M = 3 (lower rates) and M = 5 (higher rates); the M = 4 policies are used unchanged (generalisation over the channel count)
BUB_MS=3 BUB_LAMS=30,35,40,45,50 STRAT_BUFS=0,15,30,60 BUB_POLS=fixed2,lyap0.1,lyap0.3,lyapp1.5v0.1,lyapp3v0.03 CAP_LEARNED=$L2 \
  $PY cap_strat.py sim_inputs.json main_M3.json 0.44 cap 12 > main_M3.log 2>&1
echo "M3 done $(date)" >> main_sweeps_progress.log
BUB_MS=5 BUB_LAMS=50,55,60,65 STRAT_BUFS=0,15,30,60 BUB_POLS=edf,fixed2,lyap0.3,lyapp3v0.03 CAP_LEARNED=$L2 \
  $PY cap_strat.py sim_inputs.json main_M5.json 0.44 cap 12 > main_M5.log 2>&1
echo "M5 done $(date)" >> main_sweeps_progress.log
# 3) the fixed-level schemes with more channels
BUB_MS=6,8,10 BUB_LAMS=50,55,60,65 STRAT_BUFS=0,15,30,60 BUB_POLS=edf,fixed2 \
  $PY cap_strat.py sim_inputs.json main_M6to10.json 0.44 cap 12 > main_M6to10.log 2>&1
echo "M6-10 done $(date)" >> main_sweeps_progress.log
echo done > main_sweeps.done
