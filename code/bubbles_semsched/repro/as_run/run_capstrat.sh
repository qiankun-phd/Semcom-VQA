#!/bin/bash
# server-side: integrated sweep (strategic deconfliction + uplink), clean setting; then the same for EDF with more channels and for HEVC inter
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_QMARGIN=0.008 BUB_ZMODE=net BUB_ALTS=39,69.5,100
BUB_MS=3,4,5 BUB_LAMS=40,50,60,65 STRAT_BUFS=-1,0,10,30,60,120 BUB_POLS=fixed2,lyap30,lyapp3v30 $PY cap_strat.py sim_inputs.json capstrat_main.json 0.44 cap 12 > capstrat_main.log 2>&1
BUB_MS=5,8,10 BUB_LAMS=40,50,60,65 STRAT_BUFS=0,30,60,120 BUB_POLS=edf $PY cap_strat.py sim_inputs.json capstrat_edf.json 0.44 cap 12 > capstrat_edf.log 2>&1
BUB_MS=6 BUB_LAMS=50,60,65 STRAT_BUFS=0,10,30 BUB_POLS=fixed2,lyapp3v30 $PY cap_strat.py sim_inputs.json capstrat_M6.json 0.44 cap 12 > capstrat_M6.log 2>&1
BUB_SCHEME=hevc_inter BUB_MS=40,60,80 BUB_LAMS=40,50,60 STRAT_BUFS=0,30,60,120 BUB_POLS=fixed2 CAP_SEEDS=4 $PY cap_strat.py sim_inputs.json capstrat_hevc.json 0.44 cap 12 > capstrat_hevc.log 2>&1
echo done > capstrat.done
