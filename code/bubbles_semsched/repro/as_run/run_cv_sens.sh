#!/bin/bash
# server-side: sensitivity to the dispersion of the token count between evidence items (BUB_NTOK_CV = 0 and 0.6; default was 0.8)
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_ZMODE=net BUB_QBAR=0.44 BUB_QMARGIN=0.002
for cv in 0 0.6; do
  BUB_NTOK_CV=$cv BUB_MS=2,3,4,5 BUB_LAMS=40,60 BUB_VS=10,30 $PY sim_holdq.py sim_inputs.json holdq_q44_cv$cv.json 0.44 table 12 > holdq_q44_cv$cv.log 2>&1
  BUB_NTOK_CV=$cv BUB_V=30 EVAL_LAMS=40,50 EVAL_HAND=lyap10,lyap30,lyap100 EVAL_REF=lyap30 \
    $PY eval_holdq.py sim_inputs.json evalholdq_M2_q44_cv$cv.json 0.44 2 60 12 s0=hppoh_M2_q44_s0_final.pt s1=hppoh_M2_q44_s1_final.pt s2=hppoh_M2_q44_s2_final.pt > evalholdq_M2_q44_cv$cv.log 2>&1
done
echo done > cv_sens.done
