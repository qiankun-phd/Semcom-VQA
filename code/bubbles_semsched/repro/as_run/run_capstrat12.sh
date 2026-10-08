#!/bin/bash
# server-side: integrated sweep with 12 traffic seeds and a finer grid of rates and buffers (the 6-seed cells were noisy)
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_QMARGIN=0.008 BUB_ZMODE=net BUB_ALTS=39,69.5,100
BUB_MS=3,4,5 BUB_LAMS=40,45,50,55,60,65 STRAT_BUFS=0,15,30,45,60 BUB_POLS=fixed2,lyap30,lyapp3v30 CAP_SEEDS=12 $PY cap_strat.py sim_inputs.json capstrat_main12.json 0.44 cap 12 > capstrat_main12.log 2>&1
echo done > capstrat12.done
