#!/usr/bin/env bash
# Dump the demo builders' lanes on the same 200 CULane test images, on GPU (cluster venvs) or CPU (Space env),
# then compare: isolates the device / torch-build difference (same code, same cv2 loader).
# Usage: verify_dump_job.sh gpu|cpu [N]
set -uo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)"; MODE="$1"; N="${2:-200}"; OUT=$B/results/verify_cpu; mkdir -p $OUT
CKPT=$(ls -1 $B/results/finetune/clrernet_ft_corrupt_2*-73992/epoch_3.pth | head -1)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 TORCH_HOME=$B/.cache/torch
pids=()
for m in scnn ufldv2 clrernet clrernet_robust; do
  (
    if [[ $MODE == gpu ]]; then
      case $m in scnn|ufldv2) V=$m;; *) V=clrernet;; esac
      source $B/.venvs/$V/bin/activate; PY=python
      [[ $V == clrernet ]] && export PYTHONPATH="$B/vendors/clrernet:$B/vendors/clrernet/libs/models/layers/nms/src:${PYTHONPATH:-}"
    else
      PY=$B/.venvs/space_cpu/bin/python; export LANE_DEVICE=cpu PYTHONPATH="$B/vendors/clrernet:${PYTHONPATH:-}"
    fi
    [[ $m == clrernet_robust ]] && export CLR_CKPT=$CKPT
    cd $B/scripts && $PY verify_cpu.py dump $m $N $OUT/dump_${MODE}_$m.jsonl
  ) > $OUT/dump_${MODE}_$m.log 2>&1 & pids+=($!)
  [[ $MODE == gpu ]] && wait $!   # GPU: one model at a time on the single GPU
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
for m in scnn ufldv2 clrernet clrernet_robust; do tail -1 $OUT/dump_${MODE}_$m.log; done
echo DUMP_RC=$rc; exit $rc
