#!/bin/bash
# server 182 (16 cores), MAIN SETTING, first pass with ONE training seed (user, 2026-10-07: both servers train).
# A) The proposed learner with the per-flight recall criterion weighted more strongly in training. Reason: with 48 traffic seeds the policies
#    trained with BETA = 100 leave 4.9-7.3 % of the flights below the requirement at 60 flights/h, the exhaustive-search rule 4.8 %.
#    Protocol of run_sppo_final5.sh (phase 1 structure only, 500 iterations; phase 2 with the residual networks, 125), same seeds as its
#    seed 0, so only the constraint weighting differs:
#      b300  BUB_BETA = 300        b500  BUB_BETA = 500        f50  BUB_BETA = 100 and PPO_FAIR = 50 (holding-seconds per flight landing below)
#    then all three at system level with 48 traffic seeds (the grid of main_M4_48seeds).
# B) 48 traffic seeds for the other channel counts (the 12-seed capacities are not reliable at the top end): M = 5, 3, 6-10.
cd ~/bub_work
PY=python3
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
rm -f pass1_182_progress.log pass1_182.done
variant () {   # tag extra-env...
  tag=$1; shift
  ( export BUB_V=0.3 PPO_LAMS=55,60,65 PPO_LAMS_TEST=60,65 EVAL_LAMS=55,60,65 "$@"
    SPPO_MODE=struct SPPO_LRS=5e-2 PPO_SEED=30 $PY sppo_hold.py sim_inputs.json sppoV_${tag}_p1_s0.json 0.44 4 60 4 500 > sppoV_${tag}_p1_s0.log 2>&1
    SPPO_MODE=full SPPO_LRS=1e-2 SPPO_INIT=sppoV_${tag}_p1_s0_final.pt PPO_SEED=40 $PY sppo_hold.py sim_inputs.json sppoV_${tag}_p2_s0.json 0.44 4 60 4 125 > sppoV_${tag}_p2_s0.log 2>&1
    echo "$tag trained $(date)" >> pass1_182_progress.log )
}
variant b300 BUB_BETA=300 &
variant b500 BUB_BETA=500 &
variant f50 BUB_BETA=100 PPO_FAIR=50 &
( L5="s0=sppoF_p2_s0_final.pt,s1=sppoF_p2_s1_final.pt,s2=sppoF_p2_s2_final.pt,s3=sppoF_p2_s3_final.pt,s4=sppoF_p2_s4_final.pt"
  BUB_MS=5 BUB_LAMS=55,60,65 STRAT_BUFS=0,30 BUB_POLS=fixed2,lyap0.3,lyapp3v0.03 CAP_LEARNED=$L5 CAP_SEEDS=48 $PY cap_strat.py sim_inputs.json main_M5_48seeds.json 0.44 cap 4 > main_M5_48seeds.log 2>&1
  echo "M5 48 seeds done $(date)" >> pass1_182_progress.log
  BUB_MS=3 BUB_LAMS=30,35,40 STRAT_BUFS=0,30 BUB_POLS=lyap0.1,lyap0.3,lyapp1.5v0.1,lyapp3v0.03 CAP_LEARNED=$L5 CAP_SEEDS=48 $PY cap_strat.py sim_inputs.json main_M3_48seeds.json 0.44 cap 4 > main_M3_48seeds.log 2>&1
  echo "M3 48 seeds done $(date)" >> pass1_182_progress.log
  BUB_MS=6,8,10 BUB_LAMS=55,60,65 STRAT_BUFS=0,30 BUB_POLS=fixed2,edf CAP_SEEDS=48 $PY cap_strat.py sim_inputs.json main_M6to10_48seeds.json 0.44 cap 4 > main_M6to10_48seeds.log 2>&1
  echo "M6-10 48 seeds done $(date)" >> pass1_182_progress.log ) &
wait
BUB_MS=4 BUB_LAMS=55,60,65 STRAT_BUFS=0,30 BUB_POLS= CAP_LEARNED=b300=sppoV_b300_p2_s0_final.pt,b500=sppoV_b500_p2_s0_final.pt,f50=sppoV_f50_p2_s0_final.pt CAP_SEEDS=48 \
  $PY cap_strat.py sim_inputs.json sysV_48seeds.json 0.44 cap 16 > sysV_48seeds.log 2>&1
echo "variants at system level done $(date)" >> pass1_182_progress.log
echo done > pass1_182.done
