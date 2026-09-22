#!/bin/bash
# After the M2 chain finishes, run the NAIVE fixed-rate LDPC digital baseline (cliffs) on 3 channels.
# Only service s2 (image) is needed -- s0/s1 are channel-mode-independent (already from main run).
set -u
cd /home/qiankun/phd_research/uav-vqa-semantic-rl || exit 1
PY=/home/qiankun/.conda/envs/uav_semcom/bin/python
LOG=outputs/logs
M2_PID="${1:-2100302}"
SNR="-5,0,5,10,15,20"

echo "[naivechain] $(date) waiting for M2 chain PID $M2_PID"
while kill -0 "$M2_PID" 2>/dev/null; do sleep 60; done
echo "[naivechain] $(date) M2 chain finished -> starting NAIVE"

for k in rician awgn rayleigh; do
  echo "[naivechain] $(date) START naive $k"
  $PY scripts/run_v1_detector_eval.py \
      --config configs/v2_0_${k}_naive.json \
      --evaluator qwen --service-levels 2 --snr-bins=${SNR} \
      > ${LOG}/naive_${k}.log 2>&1
  echo "[naivechain] $(date) END naive $k exit=$?"
done
echo "[naivechain] $(date) NAIVE ALL DONE"
