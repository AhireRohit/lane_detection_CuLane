#!/usr/bin/env bash
# Baseline dashcam run for all three models, sequentially on ONE GPU, then the visual comparison.
B="$(cd "$(dirname "$0")/.." && pwd)"
export DASH_VIDEO="${1:-$B/vendors/ufldv2/example.mp4}"
export DASH_CROP="${2:-}"
export DASH_BATCH="run_$(date -u +%Y%m%dT%H%M%SZ)_${HOSTNAME%%.*}${DASH_CROP:+_crop}"
echo "DASH_BATCH=$DASH_BATCH VIDEO=$DASH_VIDEO"
rc_all=0
for m in scnn ufldv2 clrernet; do
  bash "$B/scripts/gpu_entry.sh" "$m" dashcam; rc=$?; echo "DASHCAM_RC $m=$rc"; [[ $rc -eq 0 ]] || rc_all=1
done
if [[ $rc_all -eq 0 ]]; then
  "$B/.venvs/scnn/bin/python" "$B/scripts/dashcam_compare.py" "$B/results/comparison/$DASH_BATCH" \
    "$B/results/dashcam/scnn/$DASH_BATCH" "$B/results/dashcam/ufldv2/$DASH_BATCH" "$B/results/dashcam/clrernet/$DASH_BATCH" || rc_all=1
fi
exit $rc_all
