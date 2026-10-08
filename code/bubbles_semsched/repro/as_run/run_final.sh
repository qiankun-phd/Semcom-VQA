#!/bin/bash
# AutoDL server, ~/bub_work2, after run_extra.sh. What was still not scheduled on 2026-10-07 evening:
#  F1 final numbers for the figures at M = 4 on the untouched traffic (seeds 501-548): the whole grid (rates 40-65, buffers 0/15/30/60),
#     every scheme, with the per-flight data and the flights per seed (paired statistics over traffic seeds)
#  F2 the same test traffic at M = 3 and M = 5 (capacity figure), every scheme; F3 fixed level at M = 6, 8, 10
#  F4 assumptions of the strategic layer that are ours: longest ground delay offered (300 / 1200 s; 600) and radius of the port zone
#     (150 / 600 m; 300)
#  F5 ablation of the two learned structure parameters: the final protocol with p frozen at 1 (only V learned = a tuned drift-plus-penalty)
#     and with V frozen at 0.3 (only p learned); three training seeds each, then on the test traffic
cd ~/bub_work2
PY=~/anaconda3/envs/bub/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
BASE="BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008"
export CAP_SCHED_CACHE=$HOME/bub_work/sched_cache; mkdir -p $CAP_SCHED_CACHE
while [ ! -f extra.done ]; do sleep 30; done
rm -f final_progress.log final.done
L="f0=sppoX_fast5_s0_final.pt,f1=sppoX_fast5_s1_final.pt,f2=sppoX_fast5_s2_final.pt,f3=sppoX_fast5_s3_final.pt,f4=sppoX_fast5_s4_final.pt"
BL="BUB_V=0.03 BUB_BETA=500 BUB_POLS="
HP="CAP_KIND=hppo BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=queue"
T="CAP_SEEDS=48 CAP_SEED0=501 CAP_RAW=1"
main () { o=$1; shift; env $BASE BUB_POLS=fixed2,lyap0.3,lyapp3v0.1 "$@" CAP_LEARNED=$L $PY cap_strat.py sim_inputs.json ${o}_main.json 0.44 cap 18 > ${o}_main.log 2>&1; }
rules () { o=$1; shift; env $BASE "$@" $PY cap_strat.py sim_inputs.json ${o}_rules.json 0.44 cap 18 > ${o}_rules.log 2>&1; }
lrn () { o=$1; shift
  env $BASE "$@" $BL $HP CAP_LEARNED=ppo=mrl5k_B_M4_s0_best.pt $PY cap_strat.py sim_inputs.json ${o}_B.json 0.44 cap 18 > ${o}_B.log 2>&1
  env $BASE "$@" $BL CAP_KIND=d3qn CAP_LEARNED=d3qn=mrl5k_d3qn_M4_s0_best.pt $PY cap_strat.py sim_inputs.json ${o}_d3qn.json 0.44 cap 18 > ${o}_d3qn.log 2>&1
  env $BASE "$@" $BL CAP_KIND=td3 CAP_LEARNED=td3=mrl5k_td3_M4_s0_best.pt $PY cap_strat.py sim_inputs.json ${o}_td3.json 0.44 cap 18 > ${o}_td3.log 2>&1; }
G="BUB_MS=4 BUB_LAMS=40,45,50,55,60,65 STRAT_BUFS=0,15,30,60 $T"
main final48_M4 $G; lrn final48_M4 $G
echo "F1 grid at M=4 on the test traffic done $(date)" >> final_progress.log
G="BUB_MS=3 BUB_LAMS=30,35,40 STRAT_BUFS=0,30 $T"
main final48_M3 $G BUB_POLS=fixed2,lyap0.1,lyap0.3,lyapp3v0.1,lyapp3v0.05,lyapp3v0.03,lyapp1.5v0.1; lrn final48_M3 $G
G="BUB_MS=5 BUB_LAMS=55,60,65 STRAT_BUFS=0,30 $T"
main final48_M5 $G BUB_POLS=fixed2,lyap0.1,lyap0.3,lyapp3v0.1,lyapp3v0.2,lyapp3v0.3; lrn final48_M5 $G
echo "F2 M=3 and M=5 done $(date)" >> final_progress.log
rules final48_M6to10 BUB_MS=6,8,10 BUB_LAMS=55,60,65 STRAT_BUFS=0,30 $T BUB_POLS=fixed2,edf
echo "F3 fixed level at M=6-10 done $(date)" >> final_progress.log
G="BUB_MS=4 BUB_LAMS=55,60 STRAT_BUFS=0,30 CAP_SEEDS=48 CAP_SEED0=501"
main sensG300 $G STRAT_GMAX=300; main sensG1200 $G STRAT_GMAX=1200; main sensR150 $G STRAT_RPORT=150; main sensR600 $G STRAT_RPORT=600
echo "F4 strategic-layer assumptions done $(date)" >> final_progress.log
TR="BUB_V=0.3 BUB_BETA=300 SPPO_MODE=struct SPPO_LRS=1.5e-1 PPO_LAMS=55,60,65 PPO_LAMS_TEST=60,65 EVAL_LAMS=55,60,65"
for sd in 0 1 2; do
  env $BASE $TR SPPO_FREEZE=p PPO_SEED=$((30 + sd)) $PY sppo_hold.py sim_inputs.json sppoAb_pfix_s$sd.json 0.44 4 60 3 500 > sppoAb_pfix_s$sd.log 2>&1 &
  env $BASE $TR SPPO_FREEZE=V PPO_SEED=$((30 + sd)) $PY sppo_hold.py sim_inputs.json sppoAb_vfix_s$sd.json 0.44 4 60 3 500 > sppoAb_vfix_s$sd.log 2>&1 &
done
wait
echo "F5 ablation trainings done $(date)" >> final_progress.log
A=""; for sd in 0 1 2; do A="$A,pfix$sd=sppoAb_pfix_s${sd}_final.pt,vfix$sd=sppoAb_vfix_s${sd}_final.pt"; done
env $BASE BUB_MS=4 BUB_LAMS=55,60,65 STRAT_BUFS=0,30 $T BUB_POLS= CAP_LEARNED=${A#,} $PY cap_strat.py sim_inputs.json ablate48_M4.json 0.44 cap 18 > ablate48_M4.log 2>&1
echo "F5 ablation on the test traffic done $(date)" >> final_progress.log
echo done > final.done
