#!/bin/bash
# new rented server (AutoDL, quota 20 cores), MAIN SETTING. Single-seed first pass, continued:
#  A) the training protocol that reached the exhaustive-search optimum at M = 4 ("fast": structure only, BUB_BETA = 300, SPPO_LRS = 1.5e-1,
#     500 iterations) trained for the other channel counts: M = 3 (rates 30/35/40) and M = 5 (55/60/65); the landscapes landscape_pv_M3 /
#     _M5 show that the best (p, V) moves with M (M = 5: V = 0.2-0.3), so a rule tuned at M = 4 is not the reference there
#  B) hold - fairness trade-off of the exhaustive-search optimum: queue margin 0.004 ... 0.012 (0.008 is in landscape_pv_M4_48), 48 traffic
#     seeds, M = 4, 60 flights/h, no buffer
#  C) the two per-M policies at system level, 48 traffic seeds, next to rule-family members
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
rm -f new_pass1_progress.log new_pass1.done
( BUB_V=0.3 BUB_BETA=300 SPPO_MODE=struct SPPO_LRS=1.5e-1 PPO_SEED=30 PPO_LAMS=30,35,40 PPO_LAMS_TEST=35,40 EVAL_LAMS=30,35,40 \
    $PY sppo_hold.py sim_inputs.json sppoX_fastM3_s0.json 0.44 3 60 4 500 > sppoX_fastM3_s0.log 2>&1
  echo "fast M3 trained $(date)" >> new_pass1_progress.log ) &
( BUB_V=0.3 BUB_BETA=300 SPPO_MODE=struct SPPO_LRS=1.5e-1 PPO_SEED=30 PPO_LAMS=55,60,65 PPO_LAMS_TEST=60,65 EVAL_LAMS=55,60,65 \
    $PY sppo_hold.py sim_inputs.json sppoX_fastM5_s0.json 0.44 5 60 4 500 > sppoX_fastM5_s0.log 2>&1
  echo "fast M5 trained $(date)" >> new_pass1_progress.log ) &
for mg in 0.004 0.006 0.010 0.012; do
  BUB_QMARGIN=$mg BUB_MS=4 BUB_LAMS=60 STRAT_BUFS=0 BUB_POLS=lyapp3v0.1,lyapp3.5v0.075,lyapp3.5v0.05,lyapp2.5v0.1 CAP_SEEDS=48 \
    $PY cap_strat.py sim_inputs.json tradeoff2_mg$mg.json 0.44 cap 8 > tradeoff2_mg$mg.log 2>&1
done
echo "tradeoff2 done $(date)" >> new_pass1_progress.log
wait
BUB_MS=3 BUB_LAMS=30,35,40 STRAT_BUFS=0,30 BUB_POLS=lyapp3v0.1,lyapp3v0.05,lyapp3v0.03 CAP_LEARNED=fM3=sppoX_fastM3_s0_final.pt CAP_SEEDS=48 \
  $PY cap_strat.py sim_inputs.json sysN_M3_48seeds.json 0.44 cap 16 > sysN_M3_48seeds.log 2>&1
BUB_MS=5 BUB_LAMS=55,60,65 STRAT_BUFS=0,30 BUB_POLS=lyapp3v0.1,lyapp3v0.2,lyapp3v0.3 CAP_LEARNED=fM5=sppoX_fastM5_s0_final.pt CAP_SEEDS=48 \
  $PY cap_strat.py sim_inputs.json sysN_M5_48seeds.json 0.44 cap 16 > sysN_M5_48seeds.log 2>&1
echo "sysN done $(date)" >> new_pass1_progress.log
echo done > new_pass1.done
