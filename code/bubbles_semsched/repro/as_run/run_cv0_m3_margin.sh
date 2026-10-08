#!/bin/bash
# server-side, dispersion 0, M = 3: larger constraint margins at evaluation time, to compare both schemes at strict feasibility (recall >= 0.44) at 55-65 flights/h
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_ZMODE=net BUB_QBAR=0.44 BUB_NTOK_CV=0
for mg in 0.006 0.008 0.012; do
  BUB_QMARGIN=$mg BUB_V=30 EVAL_LAMS=55,60,65 EVAL_HAND=lyap30,lyap100,lyap300 EVAL_REF=lyap100 \
    $PY eval_holdq.py sim_inputs.json evalholdq_cv0_M3_mg$mg.json 0.44 3 60 12 s0=hppoh_cv0_M3_s0_final.pt s1=hppoh_cv0_M3_s1_final.pt s2=hppoh_cv0_M3_s2_final.pt > evalholdq_cv0_M3_mg$mg.log 2>&1
done
echo done > cv0_m3_margin.done
