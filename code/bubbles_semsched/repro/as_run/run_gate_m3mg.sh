#!/bin/bash
# server-side, MAIN SETTING, M = 3: the rule family with larger constraint margins (more recall headroom for every flight)
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4
for mg in 0.012 0.016; do
  BUB_QMARGIN=$mg BUB_NOFIXED=1 BUB_MS=3 BUB_LAMS=40,45 BUB_VS=0.03,0.1 BUB_EXTRA=lyapp1.5v0.03,lyapp1.5v0.1,lyapp2v0.01,lyapp2v0.03,lyapp3v0.003,lyapp3v0.01,lyapp3v0.03 CAP_SEEDS=12 \
    $PY sim_holdq.py sim_inputs.json gate_M3_family_mg$mg.json 0.44 table 12 > gate_M3_family_mg$mg.log 2>&1
done
echo done > gate_M3mg.done
