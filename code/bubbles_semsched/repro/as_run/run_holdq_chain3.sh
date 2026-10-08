#!/bin/bash
# server-side chain (rented 3090 box): stage-2 baseline tables (semantic level as a decision), then the HEVC intra capacity sweep
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_VS=1,10 BUB_LAMS=40,60
BUB_QBAR=0.44 BUB_ZMODE=net BUB_MS=2,3,4,5 $PY sim_holdq.py sim_inputs.json holdq_q44_net.json 0.44 table 12 > holdq_q44_net.log 2>&1
BUB_QBAR=0.44 BUB_ZMODE=uav BUB_MS=2,3,4,5 $PY sim_holdq.py sim_inputs.json holdq_q44_uav.json 0.44 table 12 > holdq_q44_uav.log 2>&1
BUB_QBAR=0.46 BUB_ZMODE=net BUB_MS=3,4,5,6 $PY sim_holdq.py sim_inputs.json holdq_q46_net.json 0.44 table 12 > holdq_q46_net.log 2>&1
BUB_QBAR=0.46 BUB_ZMODE=uav BUB_MS=3,4,5,6 $PY sim_holdq.py sim_inputs.json holdq_q46_uav.json 0.44 table 12 > holdq_q46_uav.log 2>&1
echo TABLES_DONE > holdq_tables.done
while pgrep -f "cap_hold.py sim_inputs" > /dev/null; do sleep 10; done
unset BUB_LAMS
BUB_SCHEME=hevc_intra BUB_MS=60,120,180 BUB_LAMS=20,40,60 CAP_SEEDS=3 $PY cap_hold.py sim_inputs.json cap_hevc_intra.json 0.44 cap 12 > cap_hevc_intra.log 2>&1
echo CHAIN_DONE > cap_chain.done
