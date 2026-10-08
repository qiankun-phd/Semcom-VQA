#!/bin/bash
# server-side: V sweep of the Lyapunov baseline in the scarce-spectrum regime (M = 2), after the stage-2 tables
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1
while [ ! -f holdq_tables.done ]; do sleep 10; done
BUB_QBAR=0.44 BUB_ZMODE=net BUB_MS=2 BUB_LAMS=30,40,50,60 BUB_VS=3,10,30,100,300 $PY sim_holdq.py sim_inputs.json holdq_q44_net_M2_vsweep.json 0.44 table 12 > holdq_q44_net_M2_vsweep.log 2>&1
echo done > vsweep_m2.done
