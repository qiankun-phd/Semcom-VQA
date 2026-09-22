#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
revision_root=/home/qiankun/phd_research/vqa_semcom/outputs/revision_20260907_independent
exec nice -n 10 /home/qiankun/.conda/envs/RA_DI/bin/python -u \
  "$revision_root/revision_20260907_crossreceiver.py" \
  --repo /home/qiankun/phd_research/vqa_semcom \
  --out "$revision_root/crossreceiver_v2" \
  > "$revision_root/crossreceiver_v2.log" 2>&1
