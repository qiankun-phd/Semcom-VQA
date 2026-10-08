#!/bin/bash
# AutoDL server, ~/bub_work2: capacity against the recall requirement, proposed method RETRAINED for every requirement (final protocol,
# hyper-parameters as tuned at 0.44 and not retuned; three training seeds). The training rates bracket the capacity found with the 0.44
# policies on the development traffic (qbar*/qbarlow*): 0.45 -> 40-50, 0.46 -> 30-40, 0.42 / 0.43 -> 55-65 (ceiling of the airspace).
# Then on the test traffic (seeds 501-548), M = 4, buffers 0 and 30 s. The rules of the same sweep run on server 182 (run_182_qbar.sh).
cd ~/bub_work2
PY=~/anaconda3/envs/bub/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
BASE="BUB_IOT_DB=20.1 BUB_NTOK_CV=0 BUB_ZMODE=uav BUB_ALTS=39,69.5,100 BUB_PORTS=4 BUB_QMARGIN=0.008"
export CAP_SCHED_CACHE=$HOME/bub_work/sched_cache; mkdir -p $CAP_SCHED_CACHE
rm -f qbar_progress.log qbar.done
TR="BUB_V=0.3 BUB_BETA=300 SPPO_MODE=struct SPPO_LRS=1.5e-1"
train () {      # requirement, training rates, validation rates
  for sd in 0 1 2; do
    env $BASE $TR BUB_QBAR=$1 PPO_LAMS=$2 PPO_LAMS_TEST=$3 EVAL_LAMS=$2 PPO_SEED=$((30 + sd)) $PY sppo_hold.py sim_inputs.json sppoQ$1_s$sd.json 0.44 4 60 3 500 > sppoQ$1_s$sd.log 2>&1 &
  done
}
evalq () {      # requirement, rates
  L="r0=sppoQ$1_s0_final.pt,r1=sppoQ$1_s1_final.pt,r2=sppoQ$1_s2_final.pt"
  env $BASE BUB_QBAR=$1 BUB_MS=4 BUB_LAMS=$2 STRAT_BUFS=0,30 CAP_SEEDS=48 CAP_SEED0=501 BUB_POLS= CAP_LEARNED=$L $PY cap_strat.py sim_inputs.json qsweep$1_prop.json 0.44 cap 18 > qsweep$1_prop.log 2>&1
  echo "q = $1 on the test traffic done $(date)" >> qbar_progress.log
}
train 0.45 40,45,50 45,50; train 0.46 30,35,40 35,40; wait
echo "trainings 0.45 / 0.46 done $(date)" >> qbar_progress.log
train 0.42 55,60,65 60,65; train 0.43 55,60,65 60,65; wait
echo "trainings 0.42 / 0.43 done $(date)" >> qbar_progress.log
evalq 0.45 35,40,45,50,55; evalq 0.46 25,30,35,40,45; evalq 0.42 55,60,65; evalq 0.43 55,60,65
echo done > qbar.done
