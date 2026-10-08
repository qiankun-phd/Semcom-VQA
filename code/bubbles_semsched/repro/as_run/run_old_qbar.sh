#!/bin/bash
# 3090 server: the recall-requirement sweep at lower rates. At 0.45 and 0.46 nothing was feasible on the grid 50-65 flights/h (qbar0.45 /
# qbar0.46 of run_extra.sh), so the capacity at those requirements lies below 50: rates 30-50 here. Development traffic (401-448), M = 4,
# buffers 0 and 30 s; the fixed level is the lowest level that meets the requirement (0.46).
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
export CAP_SCHED_CACHE=$HOME/bub_work/sched_cache; mkdir -p $CAP_SCHED_CACHE
rm -f old_qbar_progress.log old_qbar.done
L="f0=sppoX_fast5_s0_final.pt,f1=sppoX_fast5_s1_final.pt,f2=sppoX_fast5_s2_final.pt,f3=sppoX_fast5_s3_final.pt,f4=sppoX_fast5_s4_final.pt"
for q in 0.45 0.46 0.44; do
  BUB_QBAR=$q BUB_MS=4 BUB_LAMS=30,35,40,45,50 STRAT_BUFS=0,30 BUB_POLS=fixed3,fixed2,lyap0.3,lyapp3v0.1 CAP_LEARNED=$L CAP_SEEDS=48 $PY cap_strat.py sim_inputs.json qbarlow${q}_main.json 0.44 cap 12 > qbarlow${q}_main.log 2>&1
  echo "q = $q done $(date)" >> old_qbar_progress.log
done
echo done > old_qbar.done
