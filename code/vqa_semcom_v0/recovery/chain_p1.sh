#!/bin/bash
# P1 completion chain: error-free clean evals -> merge -> v25 extra -> rebuild CSVs + figures.
set -u
cd /home/qiankun/phd_research/uav-vqa-semantic-rl || exit 1
PY=/home/qiankun/.conda/envs/uav_semcom/bin/python
LOG=outputs/logs
SNR="-5,0,5,10,15,20"

for part in main cmp extra; do
  echo "[p1] $(date) START clean ${part}"
  $PY scripts/run_v1_detector_eval.py --config configs/v2_0_clean_${part}.json \
      --evaluator qwen --service-levels 2 --snr-bins=40 --resume \
      > ${LOG}/clean_${part}.log 2>&1
  echo "[p1] $(date) END clean ${part} exit=$?"
done

$PY - <<'PYEOF'
import csv
def cat(dst, srcs):
    rows, hdr = [], None
    for s in srcs:
        with open(s) as f:
            r = csv.reader(f); h = next(r)
            hdr = hdr or h
            rows += list(r)
    with open(dst, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
    print(dst, len(rows))
cat('outputs/vlm/v2_0_clean_predictions.csv',
    ['outputs/vlm/v2_0_clean_main_predictions.csv'])
cat('outputs/vlm/v3_0_clean_predictions.csv',
    ['outputs/vlm/v2_0_clean_main_predictions.csv',
     'outputs/vlm/v2_0_clean_cmp_predictions.csv',
     'outputs/vlm/v2_0_clean_extra_predictions.csv'])
PYEOF

echo "[p1] $(date) START v25 extra (Qwen2.5-VL-3B, 5-qtype completion)"
$PY scripts/run_v1_detector_eval.py --config configs/v25_rician_extra.json \
    --evaluator qwen --service-levels 1,2 --snr-bins=${SNR} --resume \
    > ${LOG}/v25_extra.log 2>&1
echo "[p1] $(date) END v25 extra exit=$?"

echo "[p1] $(date) rebuild CSVs + figures"
$PY scripts/build_comparison_v2.py --prefix v2_0 \
    --out outputs/reports/comparison_all.csv > ${LOG}/rebuild_v2_0.log 2>&1
$PY scripts/build_comparison_v2.py --prefix v3_0 --naive-prefix v2_0 \
    --clean-file outputs/vlm/v3_0_clean_predictions.csv \
    --out outputs/reports/comparison_v3_5qt.csv > ${LOG}/rebuild_v3_0.log 2>&1
$PY scripts/build_latency_breakdown.py --comparison-csv outputs/reports/comparison_all.csv \
    --prefix v2_0 --channel rayleigh --tag final > ${LOG}/latency.log 2>&1
$PY scripts/make_comparison_figures_v2.py --csv outputs/reports/comparison_all.csv \
    --tag final > ${LOG}/figs_final.log 2>&1
$PY scripts/make_comparison_figures_v2.py --csv outputs/reports/comparison_v3_5qt.csv \
    --tag v3_final > ${LOG}/figs_v3.log 2>&1
echo "[p1] $(date) ALL DONE"
