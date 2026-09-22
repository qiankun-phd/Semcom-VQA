#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
RUN_ROOT=/home/qiankun/phd_research/vqa_semcom/outputs/revision_20260907_independent
exec nice -n 10 /home/qiankun/.conda/envs/RA_DI/bin/python -u \
  "$RUN_ROOT/revision_20260907_supplement.py" \
  --repo /home/qiankun/phd_research/vqa_semcom \
  --formal "$RUN_ROOT/formal_v1" --out "$RUN_ROOT/supplement_v1" \
  > "$RUN_ROOT/supplement_v1.log" 2>&1
