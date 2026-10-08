#!/bin/bash
# server-side: two-level channel-threshold baseline at M = 2: static thresholds, then thresholds driven by the constraint queue
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NOFIXED=1 BUB_ZMODE=net BUB_QBAR=0.44 BUB_MS=2 BUB_LAMS=40,50,60 BUB_VS=30
BUB_QMARGIN=0.002 BUB_THS=8,10,11,12,13,14,16 $PY sim_holdq.py sim_inputs.json holdq_q44_M2_snr.json 0.44 table 12 > holdq_q44_M2_snr.log 2>&1
BUB_QMARGIN=0.002 BUB_THS=14z0.3,14z1,14z3,16z0.3,16z1,16z3,18z1,18z3,20z3 $PY sim_holdq.py sim_inputs.json holdq_q44_M2_snrz.json 0.44 table 12 > holdq_q44_M2_snrz.log 2>&1
echo done > snr_m2.done
