#!/bin/bash
# server-side: (V, margin) sweep of the Lyapunov baseline at M = 2 (scarce spectrum) and at M = 3 with the 0.46 requirement
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NOFIXED=1 BUB_ZMODE=net
for mg in 0.001 0.002 0.004; do
  BUB_QBAR=0.44 BUB_QMARGIN=$mg BUB_MS=2 BUB_LAMS=40,50 BUB_VS=10,30,100 $PY sim_holdq.py sim_inputs.json holdq_q44_M2_mg$mg.json 0.44 table 12 > holdq_q44_M2_mg$mg.log 2>&1
done
for mg in 0 0.001 0.002 0.004; do
  BUB_QBAR=0.46 BUB_QMARGIN=$mg BUB_MS=3 BUB_LAMS=40,50 BUB_VS=10,30,100 $PY sim_holdq.py sim_inputs.json holdq_q46_M3_mg$mg.json 0.44 table 12 > holdq_q46_M3_mg$mg.log 2>&1
done
echo done > margin_m2.done
