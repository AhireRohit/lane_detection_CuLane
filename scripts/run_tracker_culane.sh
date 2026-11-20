#!/usr/bin/env bash
# Tracker accuracy on CULane test clips (scripts/tracker_culane.py), then official scoring of each variant.
# Usage: run_tracker_culane.sh <model_name> <pred_dir>
set -euo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)"; M="$1"; PRED="${2%/}"
OUT="$B/results/tracker_culane/$M"; mkdir -p "$OUT"
"$B/.venvs/scnn/bin/python" "$B/scripts/tracker_culane.py" "$M" "$PRED" "$OUT"
for v in B0r B2 B2obs; do bash "$B/scripts/score_culane_official.sh" "tracker_culane/${M}_$v" "$OUT/$v"; done
echo "TRACKER CULANE DONE $M"
