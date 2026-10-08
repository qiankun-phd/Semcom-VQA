#!/bin/bash
# server-side: integrated sweep with FOUR take-off / landing sites (at the base-station sites, nearest site) instead of one central port
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_QMARGIN=0.008 BUB_ZMODE=net BUB_ALTS=39,69.5,100 BUB_PORTS=4
BUB_MS=3,4,5 BUB_LAMS=40,50,60,65 STRAT_BUFS=-1,0,15,30,60 BUB_POLS=fixed2,lyap30,lyapp3v30 CAP_SEEDS=6 $PY cap_strat.py sim_inputs.json capstrat_p4.json 0.44 cap 12 > capstrat_p4.log 2>&1
echo done > capstrat_p4.done
