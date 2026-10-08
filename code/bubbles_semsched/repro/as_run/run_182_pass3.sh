#!/bin/bash
# server 182, MAIN SETTING, M = 4: the training protocol that reached the exhaustive-search optimum with seed 0 ("fast": structure only,
# BUB_BETA = 300, SPPO_LRS = 1.5e-1, 500 iterations = 4000 episodes; learned (p, V) = (2.60, 0.110), 48 traffic seeds at 60 flights/h:
# 5.0 % below / 0.72 s hold, against 4.8 % / 0.66 s of the best grid point (3, 0.1)) with four more training seeds, then all five at
# system level: 48 traffic seeds at M = 4, 5, 3 and the main_M4 grid (12 seeds) for the figures.
cd ~/bub_work
PY=python3
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
rm -f pass3_182_progress.log pass3_182.done
export PPO_LAMS=55,60,65 PPO_LAMS_TEST=60,65 EVAL_LAMS=55,60,65
for sd in 1 2 3 4; do
  BUB_V=0.3 BUB_BETA=300 SPPO_MODE=struct SPPO_LRS=1.5e-1 PPO_SEED=$((30 + sd)) $PY sppo_hold.py sim_inputs.json sppoX_fast_s$sd.json 0.44 4 60 4 500 > sppoX_fast_s$sd.log 2>&1 &
done
wait
echo "fast seeds 1-4 trained $(date)" >> pass3_182_progress.log
L="f0=sppoX_fast_s0_final.pt,f1=sppoX_fast_s1_final.pt,f2=sppoX_fast_s2_final.pt,f3=sppoX_fast_s3_final.pt,f4=sppoX_fast_s4_final.pt"
BUB_MS=4 BUB_LAMS=55,60,65 STRAT_BUFS=0,30 BUB_POLS=lyapp3v0.1 CAP_LEARNED=$L CAP_SEEDS=48 $PY cap_strat.py sim_inputs.json sysF_M4_48seeds.json 0.44 cap 16 > sysF_M4_48seeds.log 2>&1
echo "sysF M4 done $(date)" >> pass3_182_progress.log
BUB_MS=5 BUB_LAMS=55,60,65 STRAT_BUFS=0,30 BUB_POLS=lyapp3v0.1 CAP_LEARNED=$L CAP_SEEDS=48 $PY cap_strat.py sim_inputs.json sysF_M5_48seeds.json 0.44 cap 16 > sysF_M5_48seeds.log 2>&1
BUB_MS=3 BUB_LAMS=30,35,40 STRAT_BUFS=0,30 BUB_POLS=lyapp3v0.1 CAP_LEARNED=$L CAP_SEEDS=48 $PY cap_strat.py sim_inputs.json sysF_M3_48seeds.json 0.44 cap 16 > sysF_M3_48seeds.log 2>&1
echo "sysF M5, M3 done $(date)" >> pass3_182_progress.log
BUB_MS=4 BUB_LAMS=40,45,50,55,60,65 STRAT_BUFS=0,15,30,60 BUB_POLS= CAP_LEARNED=$L CAP_SEEDS=12 $PY cap_strat.py sim_inputs.json sysF_M4_grid.json 0.44 cap 16 > sysF_M4_grid.log 2>&1
echo "sysF grid done $(date)" >> pass3_182_progress.log
echo done > pass3_182.done
