#!/bin/bash
# server-side, MAIN SETTING, strategic deconfliction in the loop: HEVC inter (7977 B per image) with matching + urgency, after the main sweeps
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
while [ ! -f main_sweeps.done ]; do sleep 30; done
BUB_SCHEME=hevc_inter BUB_MS=60,80,100,120 BUB_LAMS=40,50,60 STRAT_BUFS=0,30,60,120 BUB_POLS=fixed2 CAP_SEEDS=6 \
  $PY cap_strat.py sim_inputs.json main_hevc.json 0.44 cap 12 > main_hevc.log 2>&1
echo done > main_hevc.done
