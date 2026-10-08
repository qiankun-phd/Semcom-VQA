#!/bin/bash
# server-side: Lyapunov baseline with the deadline-aware fallback, (V, margin) sweep at M = 2
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NOFIXED=1 BUB_ZMODE=net
for mg in 0 0.002 0.004; do
  BUB_QBAR=0.44 BUB_QMARGIN=$mg BUB_MS=2 BUB_LAMS=40,50,60 BUB_VS=30 BUB_VS_FB=3,10,30,100 $PY sim_holdq.py sim_inputs.json holdq_q44_M2_fb_mg$mg.json 0.44 table 12 > holdq_q44_M2_fb_mg$mg.log 2>&1
done
echo done > fb_m2.done
