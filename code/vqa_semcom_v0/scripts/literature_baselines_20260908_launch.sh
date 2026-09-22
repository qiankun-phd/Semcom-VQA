#!/usr/bin/env bash
set -euo pipefail
BASELINE_REPO=/home/qiankun/phd_research/vqa_semcom
BASELINE_ROOT=$BASELINE_REPO/outputs/literature_baselines_20260908
BASELINE_KIND=${1:?Specify tdeepsc or rsvqa}
BASELINE_SYMBOLS=${2:-24}
BASELINE_AUGMENT=${3:-none}
case "$BASELINE_KIND" in
  tdeepsc|rsvqa) ;;
  *) exit 2 ;;
esac
# This is an immediate launch, not a polling scheduler. The coordinating agent
# must first receive the previous owner's GPU-release message.
"$BASELINE_ROOT/env/bin/python" "$BASELINE_REPO/scripts/literature_baselines_20260908_train.py" \
  --root "$BASELINE_ROOT" --repo "$BASELINE_REPO" --kind "$BASELINE_KIND" \
  --symbols "$BASELINE_SYMBOLS" --seed 7 --device cuda --rsvqa-augmentation "$BASELINE_AUGMENT" --smoke
exec "$BASELINE_ROOT/env/bin/python" "$BASELINE_REPO/scripts/literature_baselines_20260908_train.py" \
  --root "$BASELINE_ROOT" --repo "$BASELINE_REPO" --kind "$BASELINE_KIND" \
  --symbols "$BASELINE_SYMBOLS" --seed 7 --device cuda --rsvqa-augmentation "$BASELINE_AUGMENT"
