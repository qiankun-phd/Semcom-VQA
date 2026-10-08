#!/bin/bash
# server-side, MAIN SETTING (decided 2026-10-05): per-flight recall constraint, four take-off / landing sites, clean token model.
# Tuning of the hand rules (plain drift-plus-penalty and steeper urgency) before the learners are trained here.
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4
while pgrep -f "eval_holdq.py sim_inputs" > /dev/null; do sleep 10; done
for mg in 0.004 0.008; do
  BUB_QMARGIN=$mg BUB_NOFIXED=1 BUB_MS=3,4 BUB_LAMS=50,60,65 BUB_VS=0.03,0.1,0.3 \
    BUB_EXTRA=fixed2,lyapp2v0.01,lyapp2v0.03,lyapp2v0.1,lyapp3v0.003,lyapp3v0.01,lyapp3v0.03,lyapp3v0.1 \
    $PY sim_holdq.py sim_inputs.json main_tune_mg$mg.json 0.44 table 12 > main_tune_mg$mg.log 2>&1
done
echo done > main_tune.done
