#!/bin/bash
# server-side: per-flight constraint queues need a much smaller V (a flight's own queue cannot exceed ~0.86): sweep it
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1
while [ ! -f vsweep_m2.done ]; do sleep 10; done
BUB_QBAR=0.44 BUB_ZMODE=uav BUB_MS=2,3,4 BUB_LAMS=40,60 BUB_VS=0.01,0.03,0.1,0.3 $PY sim_holdq.py sim_inputs.json holdq_q44_uav_smallv.json 0.44 table 12 > holdq_q44_uav_smallv.log 2>&1
echo done > uav_smallv.done
