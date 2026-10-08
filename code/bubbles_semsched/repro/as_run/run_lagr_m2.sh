#!/bin/bash
# server-side: min-airtime Lagrangian baseline, (k, margin) sweep at M = 2
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NOFIXED=1 BUB_ZMODE=net
for mg in 0 0.002; do
  BUB_QBAR=0.44 BUB_QMARGIN=$mg BUB_MS=2 BUB_LAMS=40,50,60 BUB_VS=30 BUB_KS=3,10,30,100,300,1000,3000 $PY sim_holdq.py sim_inputs.json holdq_q44_M2_lagr_mg$mg.json 0.44 table 12 > holdq_q44_M2_lagr_mg$mg.log 2>&1
done
echo done > lagr_m2.done
