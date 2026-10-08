#!/bin/bash
# AutoDL server, ~/bub_work2, after run_new_qasis.sh (runs next to the trainings of run_new_qbase.sh, 6 processes): the fixed token levels
# that can meet the recall requirements 0.45 and 0.46 (levels 0.46 and 0.48 = fixed3, fixed4) below the grid of run_182_qbar.sh, where
# they were infeasible down to 30 flights/h - so that "Fixed level" gets a capacity instead of "< 30". Test traffic, M = 4.
cd ~/bub_work2
PY=~/anaconda3/envs/bub/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
BASE="BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008"
export CAP_SCHED_CACHE=$HOME/bub_work/sched_cache
while [ ! -f qasis.done ]; do sleep 30; done
rm -f fixlow.done
for q in 0.45 0.46; do
  env $BASE BUB_QBAR=$q BUB_MS=4 BUB_LAMS=10,15,20,25 STRAT_BUFS=0,30 CAP_SEEDS=48 CAP_SEED0=501 BUB_POLS=fixed3,fixed4 $PY cap_strat.py sim_inputs.json qsweep${q}_fixedlow.json 0.44 cap 6 > qsweep${q}_fixedlow.log 2>&1
done
echo done > fixlow.done
