#!/bin/bash
# server 182: the rules of the recall-requirement sweep on the test traffic (seeds 501-548), M = 4, buffers 0 and 30 s; waits for
# run_182_final2.sh. For every requirement 0.42 ... 0.46:
#   fixed level   every token level 0.42 / 0.44 / 0.46 / 0.48 (fixed1..4), rates 30-60: "Fixed level" is the best of them
#   adaptive      drift-plus-penalty (V = 0.1, 0.3, 1) and the exhaustive-search grid over (p, V) of our policy family, at the rates
#                 around the capacity (found on the development traffic with the 0.44 policies)
# The proposed method, retrained per requirement, is evaluated on the AutoDL server (run_new_qbar.sh).
cd ~/bub_work
PY=python3
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
export CAP_SCHED_CACHE=$HOME/bub_work/sched_cache; mkdir -p $CAP_SCHED_CACHE
while [ ! -f final2_182.done ]; do sleep 30; done
rm -f qbar_182_progress.log qbar_182.done
T="BUB_MS=4 STRAT_BUFS=0,30 CAP_SEEDS=48 CAP_SEED0=501"
GRID=lyap0.1,lyap0.3,lyap1,lyapp3v0.03,lyapp3v0.05,lyapp3v0.1,lyapp3v0.2,lyapp3v0.3,lyapp2v0.1,lyapp4v0.1
for QL in "0.45 35,40,45,50,55" "0.46 25,30,35,40,45" "0.44 55,60,65" "0.43 55,60,65" "0.42 55,60,65"; do
  set -- $QL
  env $T BUB_QBAR=$1 BUB_LAMS=$2 BUB_POLS=$GRID $PY cap_strat.py sim_inputs.json qsweep$1_rules.json 0.44 cap 16 > qsweep$1_rules.log 2>&1
  env $T BUB_QBAR=$1 BUB_LAMS=30,35,40,45,50,55,60 BUB_POLS=fixed1,fixed2,fixed3,fixed4 $PY cap_strat.py sim_inputs.json qsweep$1_fixed.json 0.44 cap 16 > qsweep$1_fixed.log 2>&1
  echo "q = $1 rules done $(date)" >> qbar_182_progress.log
done
echo done > qbar_182.done
