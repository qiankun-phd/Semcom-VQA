#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
router_root=/home/qiankun/phd_research/vqa_semcom/outputs
exec nice -n 10 /home/qiankun/.conda/envs/RA_DI/bin/python -u \
  "$router_root/router_network_revision_20260908.py" \
  --source "$router_root/revision_20260907_independent/crossreceiver_v2" \
  --out "$router_root/router_network_revision_20260908" \
  > "$router_root/router_network_revision_20260908.log" 2>&1
