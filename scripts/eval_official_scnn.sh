#!/usr/bin/env bash
# Invoke the upstream SCNN CULane C++ evaluator with official settings.
# Does NOT copy or edit GT and never substitutes local proxy metrics.
set -euo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)"
R="$(dirname "$B")"
DATA="$R/datasets/CULane"
SCNN="$B/vendors/scnn"
EVAL="$SCNN/utils/lane_evaluation/CULane/evaluate"
PRED="${1:-$B/results/scnn/full/coord_output}"
OUT="$B/results/scnn/official"
[[ -x "$EVAL" ]] || { echo "MISSING official evaluator executable: $EVAL" >&2; echo "Build upstream CULane C++ evaluator first (requires C++ OpenCV development package)" >&2; exit 3; }
[[ -d "$PRED" ]] || { echo "Missing predictions $PRED" >&2; exit 4; }
mkdir -p "$OUT"
for i in {0..8};do
  LIST=("$DATA"/list/test_split/test${i}_*.txt)
  [[ ${#LIST[@]} -eq 1 && -f ${LIST[0]} ]] || { echo "Missing scenario list $i" >&2;exit 5; }
  # In-place ground truth lines must exist for the upstream -a data_root convention.
  rel="$(head -1 "${LIST[0]}" | awk '{print $1}' | sed 's#^/##')"
  gt="$DATA/${rel%.jpg}.lines.txt"
  if [[ ! -e "$gt" ]];then
    echo "GT NOT MAPPED: $gt" >&2
    echo "CULane Kaggle annotations_new may be separate. Verify/normalize outside the protected dataset and update data_root." >&2
    exit 6
  fi
  "$EVAL" -a "$DATA" -d "$PRED" -i "$DATA" -l "${LIST[0]}" \
    -w 30 -t 0.5 -c 1640 -r 590 -f 1 -o "$OUT/test${i}.txt"
done
cat "$OUT"/test*.txt | tee "$OUT/all_scenarios.txt"
echo 'OFFICIAL CULANE SCNN EVALUATION COMPLETE (record build version and GT)' 
