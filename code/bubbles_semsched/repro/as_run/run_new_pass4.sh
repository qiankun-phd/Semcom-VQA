#!/bin/bash
# AutoDL server (quota 20 cores), MAIN SETTING, M = 4, one training seed (PPO_SEED 30): sensitivity of the "fast" protocol to its two
# settings - the constraint price BUB_BETA (100, 500; 300 is the protocol) and the learning rate of the structure parameters SPPO_LRS
# (5e-2 = the original protocol, 3e-1; 1.5e-1 is the protocol) - then the four policies at system level with 48 traffic seeds.
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
rm -f new_pass4_progress.log new_pass4.done
export PPO_LAMS=55,60,65 PPO_LAMS_TEST=60,65 EVAL_LAMS=55,60,65 BUB_V=0.3 SPPO_MODE=struct PPO_SEED=30
BUB_BETA=100 SPPO_LRS=1.5e-1 $PY sppo_hold.py sim_inputs.json sppoS_b100_s0.json 0.44 4 60 3 500 > sppoS_b100_s0.log 2>&1 &
BUB_BETA=500 SPPO_LRS=1.5e-1 $PY sppo_hold.py sim_inputs.json sppoS_b500_s0.json 0.44 4 60 3 500 > sppoS_b500_s0.log 2>&1 &
BUB_BETA=300 SPPO_LRS=5e-2 $PY sppo_hold.py sim_inputs.json sppoS_lr05_s0.json 0.44 4 60 3 500 > sppoS_lr05_s0.log 2>&1 &
BUB_BETA=300 SPPO_LRS=3e-1 $PY sppo_hold.py sim_inputs.json sppoS_lr30_s0.json 0.44 4 60 3 500 > sppoS_lr30_s0.log 2>&1 &
wait
echo "sensitivity runs trained $(date)" >> new_pass4_progress.log
BUB_MS=4 BUB_LAMS=55,60,65 STRAT_BUFS=0,30 BUB_POLS= CAP_SEEDS=48 \
  CAP_LEARNED=b100=sppoS_b100_s0_final.pt,b500=sppoS_b500_s0_final.pt,lr05=sppoS_lr05_s0_final.pt,lr30=sppoS_lr30_s0_final.pt \
  $PY cap_strat.py sim_inputs.json sysS_48seeds.json 0.44 cap 16 > sysS_48seeds.log 2>&1
echo "sysS done $(date)" >> new_pass4_progress.log
echo done > new_pass4.done
