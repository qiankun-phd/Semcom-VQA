#!/bin/bash
# Wait for the extra chain + the Qwen2.5-VL-3B download, then run the 2nd VLM on Rician.
set -u
cd /home/qiankun/phd_research/uav-vqa-semantic-rl || exit 1
PY=/home/qiankun/.conda/envs/uav_semcom/bin/python
LOG=outputs/logs
EXTRA_PID="${1:-2130912}"
SNR="-5,0,5,10,15,20"

echo "[v25] $(date) waiting for extra chain PID $EXTRA_PID"
while kill -0 "$EXTRA_PID" 2>/dev/null; do sleep 60; done
echo "[v25] $(date) extra chain done"

echo "[v25] $(date) waiting for Qwen2.5-VL-3B download"
while ! grep -q "DONE" "${LOG}/dl_qwen25vl3b.log" 2>/dev/null; do sleep 60; done
echo "[v25] $(date) download done"

# main = presence+counting (bounded subset for a cross-VLM generalization point), then comparison
echo "[v25] $(date) START main (presence+counting, max 800 tasks)"
$PY scripts/run_v1_detector_eval.py --config configs/v25_rician_main.json \
    --evaluator qwen --service-levels 1,2 --snr-bins=${SNR} --max-tasks 800 \
    > ${LOG}/v25_main.log 2>&1
echo "[v25] $(date) END main exit=$?"

echo "[v25] $(date) START cmp (comparison)"
$PY scripts/run_v1_detector_eval.py --config configs/v25_rician_cmp.json \
    --evaluator qwen --service-levels 1,2 --snr-bins=${SNR} \
    > ${LOG}/v25_cmp.log 2>&1
echo "[v25] $(date) END cmp exit=$?"
echo "[v25] $(date) ALL DONE"
