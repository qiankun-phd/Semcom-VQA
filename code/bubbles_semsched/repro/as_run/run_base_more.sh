#!/bin/bash
# 3090 server, MAIN SETTING, after run_base_eval.sh: more baseline work (user, 2026-10-07: use idle compute for baselines).
#  1) the two extra heuristics with the best setting of heur_grid_M4 (M = 4, 60 flights/h, 12 seeds): 48 traffic seeds at M = 3, 4, 5
#  2) the comparison learners TRAINED FOR M = 3 and for M = 5 (seed 0, 5000 episodes), so that they are not only transferred from M = 4:
#     H-PPO, D3QN, TD3 at M = 3 (rates 30/35/40) and M = 5 (55/60/65); then each at its own M with 48 traffic seeds
#  3) HEVC inter with 48 traffic seeds (the 12-seed run is main_hevc12)
cd ~/bub_work
PY=~/anaconda3/envs/bub/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_QBAR=0.44 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008
export CAP_SCHED_CACHE=$HOME/bub_work/sched_cache; mkdir -p $CAP_SCHED_CACHE
while [ ! -f base_eval.done ]; do sleep 30; done
rm -f base_more_progress.log base_more.done
# 1) the first grid (heur_grid_M4) was too coarse: no member came near feasibility (threshold rule best at theta = 13 dB, Lagrangian rule at
#    k = 1000-3000). A finer grid around those, then the best member of each family over both grids: feasible with the least mean hold, else
#    the smallest violation (flights below over 5 % and conformance under 95 %, each in units of 5 points), ties by the hold.
H2=""; for t in 12 13 14 15 16; do for z in 0 3 10 20; do H2="$H2,snr${t}z${z}"; done; done; for k in 1500 2000 5000; do H2="$H2,lagr$k"; done
BUB_MS=4 BUB_LAMS=60 STRAT_BUFS=0 BUB_POLS=${H2#,} CAP_SEEDS=12 $PY cap_strat.py sim_inputs.json heur_grid2_M4.json 0.44 cap 12 > heur_grid2_M4.log 2>&1
HEUR=$($PY - <<'PYEOF'
import json
c = dict(json.load(open('heur_grid_M4.json'))['cells']); c.update(json.load(open('heur_grid2_M4.json'))['cells'])
out = []
for fam in ('snr', 'lagr'):
    cand = [(k.split('|')[0], v) for k, v in c.items() if k.startswith(fam)]
    feas = [x for x in cand if x[1]['feasible']]
    viol = lambda v: max(v['flights_below'] - 0.05, 0) / 0.05 + max(0.95 - v['conformance'], 0) / 0.05
    best = min(feas, key=lambda x: x[1]['mean_hold']) if feas else min(cand, key=lambda x: (viol(x[1]), x[1]['mean_hold']))
    out.append(best[0])
print(','.join(out))
PYEOF
)
echo "heuristics chosen: $HEUR" >> base_more_progress.log
BUB_MS=4 BUB_LAMS=55,60,65 STRAT_BUFS=0,30 BUB_POLS=$HEUR CAP_SEEDS=48 $PY cap_strat.py sim_inputs.json base48_heur_M4.json 0.44 cap 12 > base48_heur_M4.log 2>&1
BUB_MS=5 BUB_LAMS=55,60,65 STRAT_BUFS=0,30 BUB_POLS=$HEUR CAP_SEEDS=48 $PY cap_strat.py sim_inputs.json base48_heur_M5.json 0.44 cap 12 > base48_heur_M5.log 2>&1
BUB_MS=3 BUB_LAMS=30,35,40 STRAT_BUFS=0,30 BUB_POLS=$HEUR CAP_SEEDS=48 $PY cap_strat.py sim_inputs.json base48_heur_M3.json 0.44 cap 12 > base48_heur_M3.log 2>&1
echo "heuristics at 48 seeds done $(date)" >> base_more_progress.log
# 2) learners trained per M
export BUB_V=0.03 BUB_BETA=500
tr () {   # M train-rates test-rates
  M=$1
  BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=queue PPO_SEED=0 PPO_LAMS=$2 PPO_LAMS_TEST=$3 EVAL_LAMS=$2 $PY hppo_hold.py sim_inputs.json mrl5k_B_M${M}_s0.json 0.44 $M 60 2 625 > mrl5k_B_M${M}_s0.log 2>&1 &
  for a in d3qn td3; do OFF_ALGO=$a PPO_SEED=0 PPO_LAMS=$2 PPO_LAMS_TEST=$3 EVAL_LAMS=$2 $PY offpol_hold.py sim_inputs.json mrl5k_${a}_M${M}_s0.json 0.44 $M 60 2 625 > mrl5k_${a}_M${M}_s0.log 2>&1 & done
}
tr 3 30,35,40 35,40
tr 5 55,60,65 60,65
wait
echo "learners trained for M = 3 and M = 5 $(date)" >> base_more_progress.log
for ML in "3 30,35,40" "5 55,60,65"; do
  set -- $ML; M=$1; LA=$2; B="STRAT_BUFS=0,30 BUB_POLS= CAP_SEEDS=48 BUB_MS=$M BUB_LAMS=$LA"
  env $B CAP_KIND=hppo BUB_PRIOR_P=1 PPO_PRIOR=0 PPO_BASE=urg PPO_REWARD=queue CAP_LEARNED=ppo=mrl5k_B_M${M}_s0_best.pt $PY cap_strat.py sim_inputs.json base48own_B_M$M.json 0.44 cap 12 > base48own_B_M$M.log 2>&1
  env $B CAP_KIND=d3qn CAP_LEARNED=d3qn=mrl5k_d3qn_M${M}_s0_best.pt $PY cap_strat.py sim_inputs.json base48own_d3qn_M$M.json 0.44 cap 12 > base48own_d3qn_M$M.log 2>&1
  env $B CAP_KIND=td3 CAP_LEARNED=td3=mrl5k_td3_M${M}_s0_best.pt $PY cap_strat.py sim_inputs.json base48own_td3_M$M.json 0.44 cap 12 > base48own_td3_M$M.log 2>&1
done
echo "per-M learners at system level done $(date)" >> base_more_progress.log
# 3) HEVC inter, 48 traffic seeds
BUB_SCHEME=hevc_inter BUB_MS=60,80,100,120,160 BUB_LAMS=55,60,65 STRAT_BUFS=0,30,60,120 BUB_POLS=fixed2 CAP_SEEDS=48 $PY cap_strat.py sim_inputs.json main_hevc48.json 0.44 cap 12 > main_hevc48.log 2>&1
echo "HEVC 48 seeds done $(date)" >> base_more_progress.log
echo done > base_more.done
