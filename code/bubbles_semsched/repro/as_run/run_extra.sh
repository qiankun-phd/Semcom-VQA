#!/bin/bash
# AutoDL server, in ~/bub_work2 (a separate copy of the code with the options added on 2026-10-07; ~/bub_work stays untouched for the jobs
# running there). Starts after the robustness run. What flagship-journal evaluations usually contain and we did not have yet:
#  1 raw      per-flight recall / hold of every scheme at M = 4 (55, 60 flights/h, no buffer): data for the CDF figures (constraint statistics)
#  2 fresh    the main comparison at M = 4 on traffic seeds 501-548, never used for tuning anything (final numbers), incl. the search candidates
#  3 qbar     recall requirement 0.42 ... 0.46 (0.44 = main setting); the fixed level is the lowest level that meets the requirement
#  4 sens     assumptions that are ours: grace time D (30 / 120 s; 60), longest hold H_MAX (60 / 240 s; 120), Rician K (5 / 15 dB; 10)
#  5 levels   number of token levels of the semantic encoder (1 or 2 instead of 4), rules only (the learners have five outputs)
#  6 net      recall requirement on the network average instead of every flight (rules, V of that mode)
#  7 time     computation per decision and number of parameters of every scheduler
cd ~/bub_work2
PY=~/anaconda3/envs/bub/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
BASE="BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008"
export CAP_SCHED_CACHE=$HOME/bub_work/sched_cache; mkdir -p $CAP_SCHED_CACHE
while [ ! -f ~/bub_work/robust.done ]; do sleep 30; done
rm -f extra_progress.log extra.done
L="f0=sppoX_fast5_s0_final.pt,f1=sppoX_fast5_s1_final.pt,f2=sppoX_fast5_s2_final.pt,f3=sppoX_fast5_s3_final.pt,f4=sppoX_fast5_s4_final.pt"
BL="BUB_V=0.03 BUB_BETA=500 BUB_POLS="
HP="CAP_KIND=hppo BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=queue"
main () { o=$1; shift; env $BASE BUB_POLS=fixed2,lyap0.3,lyapp3v0.1 "$@" CAP_LEARNED=$L $PY cap_strat.py sim_inputs.json ${o}_main.json 0.44 cap 18 > ${o}_main.log 2>&1; }
rules () { o=$1; shift; env $BASE "$@" $PY cap_strat.py sim_inputs.json ${o}_rules.json 0.44 cap 18 > ${o}_rules.log 2>&1; }
lrn () { o=$1; shift
  env $BASE "$@" $BL $HP CAP_LEARNED=ppo=mrl5k_B_M4_s0_best.pt $PY cap_strat.py sim_inputs.json ${o}_B.json 0.44 cap 18 > ${o}_B.log 2>&1
  env $BASE "$@" $BL CAP_KIND=d3qn CAP_LEARNED=d3qn=mrl5k_d3qn_M4_s0_best.pt $PY cap_strat.py sim_inputs.json ${o}_d3qn.json 0.44 cap 18 > ${o}_d3qn.log 2>&1
  env $BASE "$@" $BL CAP_KIND=td3 CAP_LEARNED=td3=mrl5k_td3_M4_s0_best.pt $PY cap_strat.py sim_inputs.json ${o}_td3.json 0.44 cap 18 > ${o}_td3.log 2>&1; }
G1="BUB_MS=4 BUB_LAMS=55,60 STRAT_BUFS=0 CAP_SEEDS=48 CAP_RAW=1"
main raw48 $G1; lrn raw48 $G1
echo "1 raw done $(date)" >> extra_progress.log
G2="BUB_MS=4 BUB_LAMS=55,60,65 STRAT_BUFS=0,30 CAP_SEEDS=48 CAP_SEED0=501 CAP_RAW=1"
main fresh48 $G2 BUB_POLS=fixed2,lyap0.3,lyapp3v0.1,lyapp3v0.05,lyapp3v0.03,lyapp3.5v0.075,lyapp2.5v0.1; lrn fresh48 $G2
echo "2 fresh seeds done $(date)" >> extra_progress.log
G3="BUB_MS=4 BUB_LAMS=50,55,60,65 STRAT_BUFS=0,30 CAP_SEEDS=48"
for qf in "0.42 1" "0.43 2" "0.45 3" "0.46 3"; do set -- $qf; main qbar$1 $G3 BUB_QBAR=$1 BUB_POLS=fixed$2,lyap0.3,lyapp3v0.1; done
echo "3 recall requirement done $(date)" >> extra_progress.log
G4="BUB_MS=4 BUB_LAMS=55,60 STRAT_BUFS=0,30 CAP_SEEDS=48"
main sensD30 $G4 BUB_D=30; main sensD120 $G4 BUB_D=120; main sensH60 $G4 BUB_HMAX=60; main sensH240 $G4 BUB_HMAX=240
main sensK5 $G4 BUB_KRICE_DB=5; main sensK15 $G4 BUB_KRICE_DB=15
echo "4 assumptions done $(date)" >> extra_progress.log
rules lev1 $G4 BUB_LEVELS=0.48 BUB_POLS=lyap0.3,lyapp3v0.1,lyapp3v0.05
rules lev2 $G4 BUB_LEVELS=0.44,0.48 BUB_POLS=lyap0.3,lyapp3v0.1,lyapp3v0.05
rules lev4 $G4 BUB_POLS=lyap0.3,lyapp3v0.1,lyapp3v0.05
echo "5 levels done $(date)" >> extra_progress.log
rules net $G4 BUB_LAMS=55,60,65 BUB_ZMODE=net BUB_QMARGIN=0.004 BUB_POLS=fixed2,lyap10,lyap30,lyapp3v10,lyapp3v30
echo "6 network constraint done $(date)" >> extra_progress.log
T="env $BASE OMP_NUM_THREADS=1"
for pol in fixed2 lyap0.3 lyapp3v0.1; do $T TD_KIND=rule TD_POL=$pol $PY time_decide.py sim_inputs.json time_$pol.json 0.44 4 60 >> time_all.log 2>&1; done
$T TD_KIND=sppo TD_CKPT=sppoX_fast5_s0_final.pt $PY time_decide.py sim_inputs.json time_prop.json 0.44 4 60 >> time_all.log 2>&1
$T $BL $HP TD_KIND=hppo TD_CKPT=mrl5k_B_M4_s0_best.pt $PY time_decide.py sim_inputs.json time_B.json 0.44 4 60 >> time_all.log 2>&1
$T $BL TD_KIND=d3qn TD_CKPT=mrl5k_d3qn_M4_s0_best.pt $PY time_decide.py sim_inputs.json time_d3qn.json 0.44 4 60 >> time_all.log 2>&1
$T $BL TD_KIND=td3 TD_CKPT=mrl5k_td3_M4_s0_best.pt $PY time_decide.py sim_inputs.json time_td3.json 0.44 4 60 >> time_all.log 2>&1
echo "7 timing done $(date)" >> extra_progress.log
echo done > extra.done
