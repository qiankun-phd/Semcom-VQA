#!/bin/bash
# AutoDL server, MAIN SETTING, M = 4: hold - fairness trade-off of the five policies of the "fast" protocol (trained on the 3090 server,
# sppoX_fast5_s*), queue margin 0.004 ... 0.012 (0.008 is sysF_M4_48seeds), 48 traffic seeds, 60 flights/h, no buffer; plus 55 flights/h
# at the protocol margin for reference. Question: do the learned policies lie on the frontier of the exhaustive search, and is there a
# margin that puts each of them on the feasible side of the 5 % line?
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4
rm -f new_pass5_progress.log new_pass5.done
L="f0=sppoX_fast5_s0_final.pt,f1=sppoX_fast5_s1_final.pt,f2=sppoX_fast5_s2_final.pt,f3=sppoX_fast5_s3_final.pt,f4=sppoX_fast5_s4_final.pt"
for mg in 0.010 0.012 0.006 0.004; do
  BUB_QMARGIN=$mg BUB_MS=4 BUB_LAMS=60 STRAT_BUFS=0 BUB_POLS= CAP_LEARNED=$L CAP_SEEDS=48 $PY cap_strat.py sim_inputs.json tradeoff3_mg$mg.json 0.44 cap 18 > tradeoff3_mg$mg.log 2>&1
  echo "margin $mg done $(date)" >> new_pass5_progress.log
done
echo done > new_pass5.done
