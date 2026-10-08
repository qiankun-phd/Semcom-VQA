#!/bin/bash
# server-side, dispersion 0, M = 3: Lyapunov baseline with a steeper urgency (exponent p), against the three H-PPO runs, 12 traffic seeds
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_ZMODE=net BUB_QBAR=0.44 BUB_NTOK_CV=0 BUB_V=30
for mg in 0.008 0.012; do
  BUB_QMARGIN=$mg EVAL_LAMS=55,60 EVAL_HAND=lyap100,lyapp1.5v30,lyapp1.5v100,lyapp2v10,lyapp2v30,lyapp2v100,lyapp3v3,lyapp3v10,lyapp3v30 EVAL_REF=lyap100 \
    $PY eval_holdq.py sim_inputs.json evalholdq_cv0_M3_lyapp_mg$mg.json 0.44 3 60 12 s0=hppoh_cv0_M3_s0_final.pt s1=hppoh_cv0_M3_s1_final.pt s2=hppoh_cv0_M3_s2_final.pt > evalholdq_cv0_M3_lyapp_mg$mg.log 2>&1
done
echo done > lyapp.done
