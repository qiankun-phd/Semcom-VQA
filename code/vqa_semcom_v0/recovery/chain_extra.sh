#!/bin/bash
# Run co_presence+threshold eval across 3 channels (s1/s2), then rebuild v3_0 = main+cmp+extra (5 qtypes).
set -u
cd /home/qiankun/phd_research/uav-vqa-semantic-rl || exit 1
PY=/home/qiankun/.conda/envs/uav_semcom/bin/python
LOG=outputs/logs
SNR="-5,0,5,10,15,20"

for ch in rician awgn rayleigh; do
  echo "[extra] $(date) START $ch"
  $PY scripts/run_v1_detector_eval.py --config configs/v2_0_${ch}_extra.json \
      --evaluator qwen --service-levels 1,2 --snr-bins=${SNR} \
      > ${LOG}/extra_${ch}.log 2>&1
  echo "[extra] $(date) END $ch exit=$?"
done

# rebuild v3_0 = main(presence+counting) + comparison + extra(co_presence+threshold)  -> 5 qtypes
for ch in rician awgn rayleigh; do
  main=outputs/vlm/v2_0_${ch}_predictions.csv
  cmp=outputs/vlm/v2_0_${ch}_cmp_predictions.csv
  ext=outputs/vlm/v2_0_${ch}_extra_predictions.csv
  out=outputs/vlm/v3_0_${ch}_predictions.csv
  cp -f "$main" "$out"
  [ -f "$cmp" ] && tail -n +2 "$cmp" >> "$out"
  [ -f "$ext" ] && tail -n +2 "$ext" >> "$out"
  echo "[extra] rebuilt $out"
done
echo "[extra] $(date) ALL DONE"
