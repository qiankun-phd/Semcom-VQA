#!/usr/bin/env bash
set -euo pipefail
# Launch only after the preceding VLM owner explicitly releases the GPU.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
controls_repo=/home/qiankun/phd_research/vqa_semcom
cd "$controls_repo"
exec /home/qiankun/.conda/envs/uav_semcom/bin/python -u outputs/detector_decoder_controls_20260908_detector.py \
  --repo "$controls_repo" --out "$controls_repo/outputs/detector_decoder_controls_20260908/detector" \
  > outputs/detector_decoder_controls_20260908/detector.log 2>&1
