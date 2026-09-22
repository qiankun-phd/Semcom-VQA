#!/bin/bash
# Wait for the running Rician full eval, then run AWGN + Rayleigh full evals in sequence.
set -u
cd /home/qiankun/phd_research/uav-vqa-semantic-rl || exit 1
PY=/home/qiankun/.conda/envs/uav_semcom/bin/python
LOG=outputs/logs
RICIAN_PID="${1:-2098065}"
SNR="-5,0,5,10,15,20"

echo "[chain] $(date) waiting for rician PID $RICIAN_PID"
while kill -0 "$RICIAN_PID" 2>/dev/null; do sleep 60; done
echo "[chain] $(date) rician finished"

# tag the rician result (default predictions path) for consistent naming
if [ -f outputs/vlm/v2_0_snr_predictions.csv ]; then
  cp -f outputs/vlm/v2_0_snr_predictions.csv outputs/vlm/v2_0_rician_predictions.csv
  echo "[chain] tagged rician -> outputs/vlm/v2_0_rician_predictions.csv"
fi

for k in awgn rayleigh; do
  echo "[chain] $(date) START $k"
  $PY scripts/run_v1_detector_eval.py \
      --config configs/v2_0_${k}.json \
      --evaluator qwen --service-levels 0,1,2 --snr-bins=${SNR} \
      > ${LOG}/full_${k}_s012.log 2>&1
  echo "[chain] $(date) END $k exit=$?"
done

echo "[chain] $(date) ALL CHANNELS DONE"
