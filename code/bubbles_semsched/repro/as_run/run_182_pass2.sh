#!/bin/bash
# server 182, MAIN SETTING, M = 4, second batch of the single-seed first pass; starts when run_182_pass1.sh is done.
# Background (landscape_pv, 60 flights/h, 12 traffic seeds): the exhaustive search over (p, V) has its best feasible points around
# p >= 2.5, V = 0.1 (mean hold 0.35 s, 2.0-2.6 % of the flights below), better than the rule (3, 0.03) used as the reference so far
# (0.94 s, 1.9 %). The learned parameters stop at p = 1.8-2.4, V = 0.14-0.27: right direction, short of the optimum, on the high-V side.
#   1) does the learner get there with more training of the structure?   long: beta = 500, structure-only phase continued for 750 more
#      iterations;   fast: beta = 300 from the plain rule with three times the learning rate of p, V, kappa
#   2) does the neural residual add anything ON TOP of the exhaustive-search optimum?   structure frozen at (3, 0.1), only the residual
#      networks learn, beta = 100 and 300
#   3) 48 traffic seeds for these four and for the candidates of the exhaustive search; the candidates also on the main_M4 grid and on
#      the validation traffic of the learning curves
cd ~/bub_work
PY=python3
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
while [ ! -f pass1_182.done ]; do sleep 30; done
rm -f pass2_182_progress.log pass2_182.done
export PPO_LAMS=55,60,65 PPO_LAMS_TEST=60,65 EVAL_LAMS=55,60,65
BUB_V=0.3 BUB_BETA=500 SPPO_MODE=struct SPPO_LRS=5e-2 SPPO_INIT=sppoV_b500_p1_s0_final.pt PPO_SEED=31 $PY sppo_hold.py sim_inputs.json sppoX_long_s0.json 0.44 4 60 4 750 > sppoX_long_s0.log 2>&1 &
BUB_V=0.3 BUB_BETA=300 SPPO_MODE=struct SPPO_LRS=1.5e-1 PPO_SEED=30 $PY sppo_hold.py sim_inputs.json sppoX_fast_s0.json 0.44 4 60 4 500 > sppoX_fast_s0.log 2>&1 &
BUB_V=0.1 SPPO_P0=3 BUB_BETA=100 SPPO_MODE=resid SPPO_LRS=1e-2 PPO_SEED=30 $PY sppo_hold.py sim_inputs.json sppoX_optres100_s0.json 0.44 4 60 4 250 > sppoX_optres100_s0.log 2>&1 &
BUB_V=0.1 SPPO_P0=3 BUB_BETA=300 SPPO_MODE=resid SPPO_LRS=1e-2 PPO_SEED=30 $PY sppo_hold.py sim_inputs.json sppoX_optres300_s0.json 0.44 4 60 4 250 > sppoX_optres300_s0.log 2>&1 &
wait
echo "four trainings done $(date)" >> pass2_182_progress.log
CAND=lyapp2.5v0.1,lyapp3v0.1,lyapp4v0.1,lyapp3v0.2,lyapp3v0.05
BUB_MS=4 BUB_LAMS=55,60,65 STRAT_BUFS=0,30 BUB_POLS=$CAND CAP_SEEDS=48 \
  CAP_LEARNED=long=sppoX_long_s0_final.pt,fast=sppoX_fast_s0_final.pt,optres100=sppoX_optres100_s0_final.pt,optres300=sppoX_optres300_s0_final.pt \
  $PY cap_strat.py sim_inputs.json sysX_48seeds.json 0.44 cap 16 > sysX_48seeds.log 2>&1
echo "sysX 48 seeds done $(date)" >> pass2_182_progress.log
BUB_MS=4 BUB_LAMS=40,45,50,55,60,65 STRAT_BUFS=0,15,30,60 BUB_POLS=lyapp2.5v0.1,lyapp3v0.1 CAP_SEEDS=12 $PY cap_strat.py sim_inputs.json main_M4_exh.json 0.44 cap 16 > main_M4_exh.log 2>&1
VAL_POLS=lyapp2.5v0.1,lyapp3v0.1,lyapp4v0.1,lyapp3v0.2 $PY val_refs.py sim_inputs.json val_refs_exh.json 0.44 4 16 > val_refs_exh.log 2>&1
echo "exhaustive candidates done $(date)" >> pass2_182_progress.log
echo done > pass2_182.done
