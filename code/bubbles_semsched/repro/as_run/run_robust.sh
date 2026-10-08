#!/bin/bash
# AutoDL server, after the adaptive-price trial (dual_D.done): robustness of EVERY scheme, no retraining. M = 4, rates 55 and 60 flights/h,
# buffers 0 and 30 s, 48 traffic seeds; policies as trained / tuned for the main setting. Conditions:
#   iot17 / iot23   interference margin 3 dB lower / higher than assumed (BUB_IOT_DB = 17.1 / 23.1; main setting 20.1)
#   port1           one take-off / landing site instead of four (BUB_PORTS = 1): different geometry and different granted departures
# Schemes: fixed level, drift-plus-penalty (V = 0.3), exhaustive-search rule (3, 0.1), the five policies of the "fast" protocol, and the
# comparison learners (seed 0): H-PPO, D3QN, TD3.
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_QMARGIN=0.008
export CAP_SCHED_CACHE=$HOME/bub_work/sched_cache; mkdir -p $CAP_SCHED_CACHE
while [ ! -f dual_D.done ]; do sleep 30; done
rm -f robust_progress.log robust.done
L="f0=sppoX_fast5_s0_final.pt,f1=sppoX_fast5_s1_final.pt,f2=sppoX_fast5_s2_final.pt,f3=sppoX_fast5_s3_final.pt,f4=sppoX_fast5_s4_final.pt"
G="BUB_MS=4 BUB_LAMS=55,60 STRAT_BUFS=0,30 CAP_SEEDS=48"
one () {   # tag env...
  tag=$1; shift
  env "$@" $G BUB_POLS=fixed2,lyap0.3,lyapp3v0.1 CAP_LEARNED=$L $PY cap_strat.py sim_inputs.json robust_${tag}_main.json 0.44 cap 18 > robust_${tag}_main.log 2>&1
  env "$@" $G BUB_V=0.03 BUB_BETA=500 BUB_POLS= CAP_KIND=hppo BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=queue CAP_LEARNED=ppo=mrl5k_B_M4_s0_best.pt $PY cap_strat.py sim_inputs.json robust_${tag}_B.json 0.44 cap 18 > robust_${tag}_B.log 2>&1
  env "$@" $G BUB_V=0.03 BUB_BETA=500 BUB_POLS= CAP_KIND=d3qn CAP_LEARNED=d3qn=mrl5k_d3qn_M4_s0_best.pt $PY cap_strat.py sim_inputs.json robust_${tag}_d3qn.json 0.44 cap 18 > robust_${tag}_d3qn.log 2>&1
  env "$@" $G BUB_V=0.03 BUB_BETA=500 BUB_POLS= CAP_KIND=td3 CAP_LEARNED=td3=mrl5k_td3_M4_s0_best.pt $PY cap_strat.py sim_inputs.json robust_${tag}_td3.json 0.44 cap 18 > robust_${tag}_td3.log 2>&1
  echo "$tag done $(date)" >> robust_progress.log
}
one iot17 BUB_IOT_DB=17.1 BUB_PORTS=4
one iot23 BUB_IOT_DB=23.1 BUB_PORTS=4
one port1 BUB_IOT_DB=20.1 BUB_PORTS=1
echo done > robust.done
