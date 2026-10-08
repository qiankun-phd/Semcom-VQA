#!/bin/bash
# AutoDL server, ~/bub_work2, after run_new_qbar.sh: the recall-requirement sweep with the policies trained ONCE in the main setting
# (requirement 0.44) and used as they are - the requirement enters the scheduler through the constraint queue, so nothing is retrained
# (the same convention as for the number of channels). Test traffic (seeds 501-548), M = 4, buffers 0 and 30 s, the rate grids of the
# rules (run_182_qbar.sh). Proposed: the five training runs f0..f4; comparison learners: training seed 0 (seeds 1-4 on server 182).
# Reason: the retraining of run_new_qbar.sh with the hyper-parameters of 0.44 drifts to a larger V at 0.45 / 0.46 (more flights below).
cd ~/bub_work2
PY=~/anaconda3/envs/bub/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
BASE="BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008"
export CAP_SCHED_CACHE=$HOME/bub_work/sched_cache; mkdir -p $CAP_SCHED_CACHE
while [ ! -f qbar.done ]; do sleep 30; done
rm -f qasis_progress.log qasis.done
L="f0=sppoX_fast5_s0_final.pt,f1=sppoX_fast5_s1_final.pt,f2=sppoX_fast5_s2_final.pt,f3=sppoX_fast5_s3_final.pt,f4=sppoX_fast5_s4_final.pt"
BL="BUB_V=0.03 BUB_BETA=500 BUB_POLS="
HP="CAP_KIND=hppo BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=queue"
for QL in "0.45 35,40,45,50,55" "0.46 25,30,35,40,45" "0.43 55,60,65" "0.42 55,60,65"; do
  set -- $QL; G="$BASE BUB_QBAR=$1 BUB_MS=4 BUB_LAMS=$2 STRAT_BUFS=0,30 CAP_SEEDS=48 CAP_SEED0=501"
  env $G BUB_POLS= CAP_LEARNED=$L $PY cap_strat.py sim_inputs.json qsweep$1_propasis.json 0.44 cap 18 > qsweep$1_propasis.log 2>&1
  echo "q = $1 proposed as is done $(date)" >> qasis_progress.log
done
for QL in "0.45 25,30,35,40,45,50" "0.46 15,20,25,30,35,40" "0.43 40,45,50,55,60,65" "0.42 40,45,50,55,60,65"; do
  set -- $QL; G="$BASE BUB_QBAR=$1 BUB_MS=4 BUB_LAMS=$2 STRAT_BUFS=0,30 CAP_SEEDS=48 CAP_SEED0=501"
  env $G $BL $HP CAP_LEARNED=ppo=mrl5k_B_M4_s0_best.pt $PY cap_strat.py sim_inputs.json qsweep$1_Basis.json 0.44 cap 18 > qsweep$1_Basis.log 2>&1
  env $G $BL CAP_KIND=d3qn CAP_LEARNED=d3qn=mrl5k_d3qn_M4_s0_best.pt $PY cap_strat.py sim_inputs.json qsweep$1_d3qnasis.json 0.44 cap 18 > qsweep$1_d3qnasis.log 2>&1
  env $G $BL CAP_KIND=td3 CAP_LEARNED=td3=mrl5k_td3_M4_s0_best.pt $PY cap_strat.py sim_inputs.json qsweep$1_td3asis.json 0.44 cap 18 > qsweep$1_td3asis.log 2>&1
  echo "q = $1 comparison learners (seed 0) as is done $(date)" >> qasis_progress.log
done
echo done > qasis.done
