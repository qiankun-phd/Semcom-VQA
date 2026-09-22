#!/bin/bash
# Run the comparison question-type eval across 3 channels (s0/s1/s2), then merge into v3_0 prefix.
set -u
cd /home/qiankun/phd_research/uav-vqa-semantic-rl || exit 1
PY=/home/qiankun/.conda/envs/uav_semcom/bin/python
LOG=outputs/logs
SNR="-5,0,5,10,15,20"

for ch in rician awgn rayleigh; do
  echo "[cmp] $(date) START $ch"
  $PY scripts/run_v1_detector_eval.py --config configs/v2_0_${ch}_cmp.json \
      --evaluator qwen --service-levels 0,1,2 --snr-bins=${SNR} \
      > ${LOG}/cmp_${ch}.log 2>&1
  echo "[cmp] $(date) END $ch exit=$?"
done

# merge presence+counting (main) + comparison -> v3_0 prefix for 3-qtype analysis
for ch in rician awgn rayleigh; do
  main=outputs/vlm/v2_0_${ch}_predictions.csv
  cmp=outputs/vlm/v2_0_${ch}_cmp_predictions.csv
  out=outputs/vlm/v3_0_${ch}_predictions.csv
  if [ -f "$main" ] && [ -f "$cmp" ]; then
    cp -f "$main" "$out"
    tail -n +2 "$cmp" >> "$out"
    echo "[cmp] merged -> $out ($(wc -l < "$out") lines)"
  fi
done
echo "[cmp] $(date) ALL DONE"
