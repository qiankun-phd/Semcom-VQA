#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
RUN_ROOT=/home/qiankun/phd_research/vqa_semcom/outputs/revision_20260907_independent
exec nice -n 10 /home/qiankun/.conda/envs/RA_DI/bin/python -u "$RUN_ROOT/revision_20260907_dronevehicle.py" \
  --repo /home/qiankun/phd_research/vqa_semcom --formal "$RUN_ROOT/formal_v1" \
  --supplement "$RUN_ROOT/supplement_v1" --out "$RUN_ROOT/dronevehicle_v1" --target-only \
  > "$RUN_ROOT/dronevehicle_v1.log" 2>&1
