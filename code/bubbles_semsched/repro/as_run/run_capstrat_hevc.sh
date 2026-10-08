#!/bin/bash
# server-side: integrated sweep for HEVC inter after the cost fix (the first run added the gated token cost on top of the bit stream)
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_QMARGIN=0.008 BUB_ZMODE=net BUB_ALTS=39,69.5,100
mv capstrat_hevc.log capstrat_hevc_INVALID.log; mv capstrat_hevc.json capstrat_hevc_INVALID.json
BUB_SCHEME=hevc_inter BUB_MS=40,60,80 BUB_LAMS=40,50,60 STRAT_BUFS=0,30,60,120 BUB_POLS=fixed2 CAP_SEEDS=4 $PY cap_strat.py sim_inputs.json capstrat_hevc.json 0.44 cap 12 > capstrat_hevc.log 2>&1
echo done > capstrat_hevc.done
