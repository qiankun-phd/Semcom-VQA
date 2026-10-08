#!/bin/bash
# first rented server (<address withheld>, RTX 3090, 12 cores; back online 2026-10-07 12:05), MAIN SETTING, M = 4.
# The "fast" protocol (structure only, BUB_BETA = 300, SPPO_LRS = 1.5e-1, 500 iterations = 4000 episodes) with FIVE training seeds
# (PPO_SEED 30-34), then all five at system level: 48 traffic seeds at M = 4, 5, 3 and the main_M4 grid (12 seeds) for the figures.
# (Moved here from the AutoDL server, where the same five runs had just started and were sharing 20 slow cores with other jobs.)
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
rm -f old_pass3_progress.log old_pass3.done
export PPO_LAMS=55,60,65 PPO_LAMS_TEST=60,65 EVAL_LAMS=55,60,65
for sd in 0 1 2 3 4; do
  BUB_V=0.3 BUB_BETA=300 SPPO_MODE=struct SPPO_LRS=1.5e-1 PPO_SEED=$((30 + sd)) $PY sppo_hold.py sim_inputs.json sppoX_fast5_s$sd.json 0.44 4 60 2 500 > sppoX_fast5_s$sd.log 2>&1 &
done
wait
echo "fast seeds 0-4 trained $(date)" >> old_pass3_progress.log
L="f0=sppoX_fast5_s0_final.pt,f1=sppoX_fast5_s1_final.pt,f2=sppoX_fast5_s2_final.pt,f3=sppoX_fast5_s3_final.pt,f4=sppoX_fast5_s4_final.pt"
BUB_MS=4 BUB_LAMS=55,60,65 STRAT_BUFS=0,30 BUB_POLS=lyapp3v0.1 CAP_LEARNED=$L CAP_SEEDS=48 $PY cap_strat.py sim_inputs.json sysF_M4_48seeds.json 0.44 cap 12 > sysF_M4_48seeds.log 2>&1
echo "sysF M4 done $(date)" >> old_pass3_progress.log
BUB_MS=5 BUB_LAMS=55,60,65 STRAT_BUFS=0,30 BUB_POLS=lyapp3v0.1 CAP_LEARNED=$L CAP_SEEDS=48 $PY cap_strat.py sim_inputs.json sysF_M5_48seeds.json 0.44 cap 12 > sysF_M5_48seeds.log 2>&1
BUB_MS=3 BUB_LAMS=30,35,40 STRAT_BUFS=0,30 BUB_POLS=lyapp3v0.1 CAP_LEARNED=$L CAP_SEEDS=48 $PY cap_strat.py sim_inputs.json sysF_M3_48seeds.json 0.44 cap 12 > sysF_M3_48seeds.log 2>&1
echo "sysF M5, M3 done $(date)" >> old_pass3_progress.log
BUB_MS=4 BUB_LAMS=40,45,50,55,60,65 STRAT_BUFS=0,15,30,60 BUB_POLS= CAP_LEARNED=$L CAP_SEEDS=12 $PY cap_strat.py sim_inputs.json sysF_M4_grid.json 0.44 cap 12 > sysF_M4_grid.log 2>&1
echo "sysF grid done $(date)" >> old_pass3_progress.log
echo done > old_pass3.done
