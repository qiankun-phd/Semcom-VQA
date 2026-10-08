#!/bin/bash
# rented server, MAIN SETTING: the exhaustive search where it is still missing.
#  1) M = 4, 60 flights/h, no buffer, 48 traffic seeds, a finer grid around the optimum of landscape_pv (12 seeds)
#  2) the (p, V) landscape at the other channel counts, at the rate that decides their capacity: M = 3 at 40 flights/h, M = 5 at 65
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
while [ ! -f pass1_train.done ]; do sleep 30; done
rm -f pass2_progress.log pass2_eval.done
PV=""; for p in 2.5 3 3.5; do for v in 0.05 0.075 0.1 0.15; do PV="$PV,lyapp${p}v${v}"; done; done
BUB_MS=4 BUB_LAMS=60 STRAT_BUFS=0 BUB_POLS=${PV#,} CAP_SEEDS=48 $PY cap_strat.py sim_inputs.json landscape_pv_M4_48.json 0.44 cap 12 > landscape_pv_M4_48.log 2>&1
echo "fine landscape M4 (48 seeds) done $(date)" >> pass2_progress.log
PV=""; for p in 1 1.5 2 2.5 3 3.5 4; do for v in 0.01 0.02 0.03 0.05 0.1 0.2 0.3 0.5; do PV="$PV,lyapp${p}v${v}"; done; done
BUB_MS=3 BUB_LAMS=40 STRAT_BUFS=0,30 BUB_POLS=${PV#,} CAP_SEEDS=12 $PY cap_strat.py sim_inputs.json landscape_pv_M3.json 0.44 cap 12 > landscape_pv_M3.log 2>&1
echo "landscape M3 done $(date)" >> pass2_progress.log
BUB_MS=5 BUB_LAMS=65 STRAT_BUFS=0 BUB_POLS=${PV#,} CAP_SEEDS=12 $PY cap_strat.py sim_inputs.json landscape_pv_M5.json 0.44 cap 12 > landscape_pv_M5.log 2>&1
echo "landscape M5 done $(date)" >> pass2_progress.log
echo done > pass2_eval.done
