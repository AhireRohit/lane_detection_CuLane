#!/usr/bin/env bash
# CPU job wrapper for scripts/onnx_clrernet.py. Usage: onnx_job.sh export | onnx_job.sh infer <variant> <i> <n>
set -euo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)"; PY="$B/.venvs/clrernet/bin/python"; X="$B/.venvs/onnx_extra"
if [[ "$1" == export ]]; then
  [[ -f "$X/.ok" ]] || { rm -rf "$X"; "$PY" -m pip install -q --target "$X" onnx==1.16.2 onnxruntime==1.19.2 && touch "$X/.ok"; }
  export ORIG_PRED="${ORIG_PRED:-$(ls -1d "$B"/results/clrernet/full/run_*73864/pred | tail -1)}"
fi
exec "$PY" "$B/scripts/onnx_clrernet.py" "$@"
