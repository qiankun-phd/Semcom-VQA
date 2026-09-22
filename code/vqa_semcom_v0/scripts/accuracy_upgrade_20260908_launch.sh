#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
upgrade_repo=/home/qiankun/phd_research/vqa_semcom
exec nice -n 10 /home/qiankun/.conda/envs/RA_DI/bin/python -u \
  "$upgrade_repo/outputs/accuracy_upgrade_20260908.py" --repo "$upgrade_repo" \
  --out "$upgrade_repo/outputs/accuracy_upgrade_20260908/phase_a" \
  > "$upgrade_repo/outputs/accuracy_upgrade_20260908_phase_a.log" 2>&1
