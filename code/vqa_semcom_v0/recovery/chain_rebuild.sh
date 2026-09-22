#!/bin/bash
# Full rebuild of the comparison experiment on 160 (182 lost). All stages resume-safe.
set -u
cd /home/qiankun/phd_research/uav-vqa-semantic-rl || exit 1
PY=/home/qiankun/.conda/envs/uav_semcom/bin/python
LOG=outputs/logs
SNR="-5,0,5,10,15,20"
mkdir -p ${LOG}

stamp(){ echo "[rebuild] $(date '+%F %T') $*"; }

# ---- Stage 1: main rician (base yaml -> v2_0_snr_predictions.csv) ----
stamp "START main rician"
$PY scripts/run_v1_detector_eval.py --config configs/v2_0_ldpc_channel.yaml \
    --evaluator qwen --service-levels 0,1,2 --snr-bins=${SNR} --resume \
    > ${LOG}/full_rician_s012.log 2>&1
stamp "END main rician exit=$?"
cp -f outputs/vlm/v2_0_snr_predictions.csv outputs/vlm/v2_0_rician_predictions.csv 2>/dev/null \
  && stamp "tagged rician predictions"

# ---- Stage 2: main awgn + rayleigh ----
for k in awgn rayleigh; do
  stamp "START main $k"
  $PY scripts/run_v1_detector_eval.py --config configs/v2_0_${k}.json \
      --evaluator qwen --service-levels 0,1,2 --snr-bins=${SNR} --resume \
      > ${LOG}/full_${k}_s012.log 2>&1
  stamp "END main $k exit=$?"
done

# ---- Stage 3: M2 analog x3 ----
for k in rician awgn rayleigh; do
  case $k in
    rician)   CFG=configs/v2_0_ldpc_channel.yaml ;;
    awgn)     CFG=configs/v2_0_awgn.json ;;
    rayleigh) CFG=configs/v2_0_rayleigh.json ;;
  esac
  stamp "START M2 $k"
  $PY scripts/run_m2_analog_eval.py --config "$CFG" --evaluator qwen --snr-bins=${SNR} \
      --out outputs/vlm/m2_analog_${k}_predictions.csv > ${LOG}/m2_${k}.log 2>&1
  stamp "END M2 $k exit=$?"
done

# ---- Stage 4: naive fixed-rate x3 (s2 only, the cliff baseline) ----
for k in rician awgn rayleigh; do
  stamp "START naive $k"
  $PY scripts/run_v1_detector_eval.py --config configs/v2_0_${k}_naive.json \
      --evaluator qwen --service-levels 2 --snr-bins=${SNR} --resume \
      > ${LOG}/naive_${k}.log 2>&1
  stamp "END naive $k exit=$?"
done

# ---- Stage 5: comparison qtype x3 (s1,s2) ----
for ch in rician awgn rayleigh; do
  stamp "START cmp $ch"
  $PY scripts/run_v1_detector_eval.py --config configs/v2_0_${ch}_cmp.json \
      --evaluator qwen --service-levels 1,2 --snr-bins=${SNR} --resume \
      > ${LOG}/cmp_${ch}.log 2>&1
  stamp "END cmp $ch exit=$?"
done

# ---- Stage 6: extra qtypes x3 (s1,s2) + merge v3_0 (5 qtypes) ----
for ch in rician awgn rayleigh; do
  stamp "START extra $ch"
  $PY scripts/run_v1_detector_eval.py --config configs/v2_0_${ch}_extra.json \
      --evaluator qwen --service-levels 1,2 --snr-bins=${SNR} --resume \
      > ${LOG}/extra_${ch}.log 2>&1
  stamp "END extra $ch exit=$?"
done
for ch in rician awgn rayleigh; do
  main=outputs/vlm/v2_0_${ch}_predictions.csv
  cmp=outputs/vlm/v2_0_${ch}_cmp_predictions.csv
  ext=outputs/vlm/v2_0_${ch}_extra_predictions.csv
  out=outputs/vlm/v3_0_${ch}_predictions.csv
  cp -f "$main" "$out"
  [ -f "$cmp" ] && tail -n +2 "$cmp" >> "$out"
  [ -f "$ext" ] && tail -n +2 "$ext" >> "$out"
  stamp "rebuilt $out"
done

# ---- Stage 7: error-free clean (40 dB awgn, s2) + merges ----
for part in main cmp extra; do
  stamp "START clean $part"
  $PY scripts/run_v1_detector_eval.py --config configs/v2_0_clean_${part}.json \
      --evaluator qwen --service-levels 2 --snr-bins=40 --resume \
      > ${LOG}/clean_${part}.log 2>&1
  stamp "END clean $part exit=$?"
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

# ---- Stage 8: 2nd VLM (Qwen2.5-VL-3B) on rician: main/cmp/extra ----
stamp "downloading Qwen2.5-VL-3B"
$PY - > ${LOG}/dl_qwen25vl3b.log 2>&1 <<'PYEOF'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct")
print("DONE", p)
PYEOF
stamp "download exit=$? (see dl_qwen25vl3b.log)"
stamp "START v25 main (max 800 tasks)"
$PY scripts/run_v1_detector_eval.py --config configs/v25_rician_main.json \
    --evaluator qwen --service-levels 1,2 --snr-bins=${SNR} --max-tasks 800 --resume \
    > ${LOG}/v25_main.log 2>&1
stamp "END v25 main exit=$?"
stamp "START v25 cmp"
$PY scripts/run_v1_detector_eval.py --config configs/v25_rician_cmp.json \
    --evaluator qwen --service-levels 1,2 --snr-bins=${SNR} --resume \
    > ${LOG}/v25_cmp.log 2>&1
stamp "END v25 cmp exit=$?"
stamp "START v25 extra"
$PY scripts/run_v1_detector_eval.py --config configs/v25_rician_extra.json \
    --evaluator qwen --service-levels 1,2 --snr-bins=${SNR} --resume \
    > ${LOG}/v25_extra.log 2>&1
stamp "END v25 extra exit=$?"

# ---- Stage 9: rebuild CSVs + analyses + figures ----
stamp "rebuild CSVs + figures"
$PY scripts/build_comparison_v2.py --prefix v2_0 \
    --out outputs/reports/comparison_all.csv > ${LOG}/rebuild_v2_0.log 2>&1
$PY scripts/build_comparison_v2.py --prefix v3_0 --naive-prefix v2_0 \
    --clean-file outputs/vlm/v3_0_clean_predictions.csv \
    --out outputs/reports/comparison_v3_5qt.csv > ${LOG}/rebuild_v3_0.log 2>&1
$PY scripts/build_ablation.py --prefix v3_0 --out outputs/reports/ablation_mechanism_v3.csv \
    > ${LOG}/ablation.log 2>&1
$PY scripts/build_evidence_complementarity.py > ${LOG}/complementarity.log 2>&1
$PY scripts/make_complementarity_fig.py > ${LOG}/complementarity_fig.log 2>&1
$PY scripts/build_latency_breakdown.py --comparison-csv outputs/reports/comparison_all.csv \
    --prefix v2_0 --channel rayleigh --tag final > ${LOG}/latency.log 2>&1
$PY scripts/make_comparison_figures_v2.py --csv outputs/reports/comparison_all.csv \
    --tag final > ${LOG}/figs_final.log 2>&1
$PY scripts/make_comparison_figures_v2.py --csv outputs/reports/comparison_v3_5qt.csv \
    --tag v3_final > ${LOG}/figs_v3.log 2>&1
stamp "ALL DONE"
