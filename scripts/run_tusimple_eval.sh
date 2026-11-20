#!/usr/bin/env bash
# GPU job: scripts/tusimple_eval.py for one model/checkpoint, then the official CULane C++ evaluator on its lanes.
# Usage: run_tusimple_eval.sh <tag> <model: scnn|ufldv2|clrernet> <checkpoint|released> [--clips]
set -euo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)"; TAG="$1"; M="$2"; CK="$3"; shift 3
OUT="$B/results/tusimple/$TAG"; mkdir -p "$OUT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
source "$B/.venvs/$M/bin/activate"
if [[ "$M" == clrernet ]]; then
  export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$B/vendors/clrernet:$B/vendors/clrernet/libs/models/layers/nms/src:${PYTHONPATH:-}"
  [[ "$CK" == released ]] || export CLR_CKPT="$CK"
fi
cd "$B/scripts"; python tusimple_eval.py "$M" "$OUT" "$@"
EVAL="$B/vendors/ufldv2/evaluation/culane/build_conda_gcc/culane_evaluator"; export LD_LIBRARY_PATH="$B/.opencv_cpp/lib:${LD_LIBRARY_PATH:-}"
for d in "$OUT"/*/; do
  v=$(basename "$d"); [[ "$v" == gt_lines || "$v" == work_dir ]] && continue
  "$EVAL" -a "$OUT/gt_lines/" -d "$d" -i "$B/../datasets/TuSimple/" -l "$OUT/list.txt" -w 30 -t 0.5 -c 1280 -r 720 -f 1 -o "$OUT/culane_eval_$v.txt" >/dev/null
  echo "CULANE-STYLE $TAG $v: $(tr '\n' ' ' < "$OUT/culane_eval_$v.txt")"
done
echo "TUSIMPLE JOB DONE $TAG"
