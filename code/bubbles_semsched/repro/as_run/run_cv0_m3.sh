#!/bin/bash
# server-side, token-count dispersion 0 (all evidence items alike): the clean setting. M = 3 is then the tight operating point.
# 1) (V, margin) sweep of the Lyapunov baseline  2) careful evaluation of the three H-PPO runs started alongside
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_ZMODE=net BUB_QBAR=0.44 BUB_NTOK_CV=0
for mg in 0 0.002 0.004; do
  BUB_NOFIXED=1 BUB_QMARGIN=$mg BUB_MS=3 BUB_LAMS=40,50,60 BUB_VS=10,30,100 $PY sim_holdq.py sim_inputs.json holdq_cv0_M3_mg$mg.json 0.44 table 12 > holdq_cv0_M3_mg$mg.log 2>&1
done
echo done > cv0_m3_sweep.done
while pgrep -f "hppo_hold.py sim_inputs" > /dev/null; do sleep 20; done
A=""
for s in 0 1 2; do
  [ -f hppoh_cv0_M3_s${s}_final.pt ] && A="$A s$s=hppoh_cv0_M3_s${s}_final.pt"
  [ -f hppoh_cv0_M3_s${s}_best.pt ] && A="$A s${s}b=hppoh_cv0_M3_s${s}_best.pt"
done
BUB_QMARGIN=0.004 BUB_V=30 EVAL_LAMS=40,50,55,60 EVAL_HAND=fixed2,lyap10,lyap30,lyap100 EVAL_REF=lyap30 \
  $PY eval_holdq.py sim_inputs.json evalholdq_cv0_M3.json 0.44 3 60 12 $A > evalholdq_cv0_M3.log 2>&1
echo done > evalholdq_cv0_M3.done
