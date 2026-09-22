#!/bin/bash
# After the main 3-channel chain finishes, run the M2 analog baseline (Qwen) on all 3 channels.
set -u
cd /home/qiankun/phd_research/uav-vqa-semantic-rl || exit 1
PY=/home/qiankun/.conda/envs/uav_semcom/bin/python
LOG=outputs/logs
MAIN_PID="${1:-2099399}"
SNR="-5,0,5,10,15,20"

echo "[m2chain] $(date) waiting for main chain PID $MAIN_PID"
while kill -0 "$MAIN_PID" 2>/dev/null; do sleep 60; done
echo "[m2chain] $(date) main chain finished -> starting M2"

for k in rician awgn rayleigh; do
  case $k in
    rician)   CFG=configs/v2_0_ldpc_channel.yaml ;;
    awgn)     CFG=configs/v2_0_awgn.json ;;
    rayleigh) CFG=configs/v2_0_rayleigh.json ;;
  esac
  echo "[m2chain] $(date) START M2 $k ($CFG)"
  $PY scripts/run_m2_analog_eval.py --config "$CFG" --evaluator qwen --snr-bins=${SNR} \
      --out outputs/vlm/m2_analog_${k}_predictions.csv > ${LOG}/m2_${k}.log 2>&1
  echo "[m2chain] $(date) END M2 $k exit=$?"
done
echo "[m2chain] $(date) M2 ALL DONE"
