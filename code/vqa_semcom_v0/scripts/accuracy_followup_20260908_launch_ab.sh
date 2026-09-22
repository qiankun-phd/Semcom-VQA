#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
followup_repo=/home/qiankun/phd_research/vqa_semcom
cd "$followup_repo"
exec nice -n 10 /home/qiankun/.conda/envs/RA_DI/bin/python -u outputs/accuracy_followup_20260908_ab.py \
  --repo "$followup_repo" --out outputs/accuracy_followup_20260908/phase_ab \
  > outputs/accuracy_followup_20260908/phase_ab.log 2>&1
