#!/bin/bash
# server-side, MAIN SETTING, M = 3: the rule family (urgency exponent p, weight V; p = 1 is the plain drift-plus-penalty rule) on the
# hold-against-fairness trade-off line. Which member is feasible (flights below <= 5 %) with the least holding at 35-45 flights/h?
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
while [ ! -f gate_E3.done ]; do sleep 30; done
BUB_NOFIXED=1 BUB_MS=3 BUB_LAMS=35,40,45 BUB_VS=0.01,0.03,0.1 BUB_EXTRA=lyapp1.5v0.01,lyapp1.5v0.03,lyapp1.5v0.1,lyapp2v0.001,lyapp2v0.003,lyapp2v0.01,lyapp3v0.0003,lyapp3v0.001,lyapp3v0.003,lyapp3v0.03 CAP_SEEDS=12 \
  $PY sim_holdq.py sim_inputs.json gate_M3_family.json 0.44 table 12 > gate_M3_family.log 2>&1
echo done > gate_M3sel.done
