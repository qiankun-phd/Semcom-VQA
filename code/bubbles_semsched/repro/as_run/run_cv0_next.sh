#!/bin/bash
# server-side, dispersion 0 (clean setting): the remaining pieces of the capacity-against-channels picture
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_QBAR=0.44 BUB_NTOK_CV=0
# 1) what the clean-setting policy does (M = 3, 60 flights/h)
BUB_ZMODE=net BUB_QMARGIN=0.006 BUB_V=30 DIAG_LAM=60 $PY diag_hppoh.py sim_inputs.json diag_hppoh_cv0_M3_s2.json 0.44 3 60 12 hppoh_cv0_M3_s2_final.pt > diag_hppoh_cv0_M3_s2.log 2>&1
# 2) Lyapunov baseline at M = 2 (network constraint)
for mg in 0.004 0.008; do
  BUB_ZMODE=net BUB_NOFIXED=1 BUB_QMARGIN=$mg BUB_MS=2 BUB_LAMS=20,30,40 BUB_VS=30,100 $PY sim_holdq.py sim_inputs.json holdq_cv0_M2_mg$mg.json 0.44 table 12 > holdq_cv0_M2_mg$mg.log 2>&1
done
# 3) fixed level, capacity sweep (matching + urgency, EDF)
BUB_SCHEME=gated BUB_MS=3,4,5,6,8 BUB_POLICIES=urg,edf $PY cap_hold.py sim_inputs.json cap_gated_cv0.json 0.44 cap 12 > cap_gated_cv0.log 2>&1
# 4) per-flight constraint, Lyapunov baseline (small V), M = 3, 4, 5
for mg in 0 0.004; do
  BUB_ZMODE=uav BUB_NOFIXED=1 BUB_QMARGIN=$mg BUB_MS=3,4,5 BUB_LAMS=40,50,60 BUB_VS=0.003,0.01,0.03,0.1 $PY sim_holdq.py sim_inputs.json holdq_cv0_uav_mg$mg.json 0.44 table 12 > holdq_cv0_uav_mg$mg.log 2>&1
done
echo done > cv0_next_tables.done
# 5) careful evaluation of the M = 2 runs
while pgrep -f "hppo_hold.py sim_inputs" > /dev/null; do sleep 20; done
A=""
for s in 0 1 2; do [ -f hppoh_cv0_M2_s${s}_final.pt ] && A="$A s$s=hppoh_cv0_M2_s${s}_final.pt"; done
for mg in 0.006 0.010; do
  BUB_ZMODE=net BUB_QMARGIN=$mg BUB_V=30 EVAL_LAMS=20,30,40,50 EVAL_HAND=lyap30,lyap100 EVAL_REF=lyap30 \
    $PY eval_holdq.py sim_inputs.json evalholdq_cv0_M2_mg$mg.json 0.44 2 60 12 $A > evalholdq_cv0_M2_mg$mg.log 2>&1
done
echo done > cv0_next.done
