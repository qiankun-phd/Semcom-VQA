#!/usr/bin/env bash
set -euo pipefail
followup_repo=/home/qiankun/phd_research/vqa_semcom
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false
cd "$followup_repo"
exec outputs/accuracy_upgrade_20260908/quant_env/bin/python -u outputs/accuracy_followup_20260908_vlm.py \
  --repo "$followup_repo" --out outputs/accuracy_followup_20260908/phase_c --resume \
  > outputs/accuracy_followup_20260908/phase_c.log 2>&1
