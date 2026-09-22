#!/usr/bin/env bash
# One-shot sequential experiment workflow, not a recurring monitor or scheduler.
set -euo pipefail
BASELINE_REPO=/home/qiankun/phd_research/vqa_semcom
BASELINE_ROOT=$BASELINE_REPO/outputs/literature_baselines_20260908
"$BASELINE_ROOT/env/bin/python" "$BASELINE_REPO/scripts/literature_baselines_20260908_augment.py" \
  --root "$BASELINE_ROOT" > "$BASELINE_ROOT/augment.log" 2>&1
bash "$BASELINE_REPO/scripts/literature_baselines_20260908_launch.sh" rsvqa 24 d4 > "$BASELINE_ROOT/rsvqa_d4.log" 2>&1
bash "$BASELINE_REPO/scripts/literature_baselines_20260908_launch.sh" tdeepsc 48 > "$BASELINE_ROOT/tdeepsc48.log" 2>&1
bash "$BASELINE_REPO/scripts/literature_baselines_20260908_launch.sh" tdeepsc 96 > "$BASELINE_ROOT/tdeepsc96.log" 2>&1
"$BASELINE_ROOT/env/bin/python" "$BASELINE_REPO/scripts/literature_baselines_20260908_analyze.py" --root "$BASELINE_ROOT"
