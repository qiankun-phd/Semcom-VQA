#!/bin/bash
# server-side chain (rented 3090 box): waits for the running evaluation, then capacity sweeps per scheme, then the CEM reference
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1
while pgrep -f "eval_hold.py sim_inputs" > /dev/null; do sleep 10; done
BUB_SCHEME=gated BUB_MS=3,4,5,6,8,10 BUB_POLICIES=urg,edf $PY cap_hold.py sim_inputs.json cap_gated.json 0.44 cap 12 > cap_gated.log 2>&1
BUB_SCHEME=hevc_inter BUB_MS=10,20,30,40,60,80 BUB_LAMS=20,40,50,60,65 CAP_SEEDS=4 $PY cap_hold.py sim_inputs.json cap_hevc_inter.json 0.44 cap 12 > cap_hevc_inter.log 2>&1
$PY cem_hold.py sim_inputs.json cemhold_M5.json 0.44 5 60 12 30 40 > cemhold_M5.log 2>&1
$PY eval_hold.py sim_inputs.json evalhold_M5_cem.json 0.44 5 60 12 cem=cemhold_M5.json s1=mhold_M5_s1_final.pt > evalhold_M5_cem.log 2>&1
BUB_SCHEME=hevc_intra BUB_MS=60,120,180 BUB_LAMS=20,40,60 CAP_SEEDS=3 $PY cap_hold.py sim_inputs.json cap_hevc_intra.json 0.44 cap 12 > cap_hevc_intra.log 2>&1
echo CHAIN_DONE > cap_chain.done
