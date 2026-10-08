#!/bin/bash
# server-side: careful evaluation (12 traffic seeds, paired with the tuned Lyapunov baseline) once the three H-PPO runs have finished
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
while pgrep -f "hppo_hold.py sim_inputs" > /dev/null; do sleep 20; done
A=""
for s in 0 1 2; do
  [ -f hppoh_M2_q44_s${s}_final.pt ] && A="$A s$s=hppoh_M2_q44_s${s}_final.pt"
  [ -f hppoh_M2_q44_s${s}_best.pt ] && A="$A s${s}b=hppoh_M2_q44_s${s}_best.pt"
done
BUB_IOT_DB=20.1 BUB_QBAR=0.44 BUB_QMARGIN=0.002 BUB_ZMODE=net BUB_V=30 EVAL_LAMS=40,45,50 EVAL_HAND=lyap10,lyap30,lyap100 EVAL_REF=lyap30 \
  $PY eval_holdq.py sim_inputs.json evalholdq_M2_q44.json 0.44 2 60 12 $A > evalholdq_M2_q44.log 2>&1
echo done > evalholdq_M2_q44.done
