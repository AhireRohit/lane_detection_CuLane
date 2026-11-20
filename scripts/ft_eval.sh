#!/usr/bin/env bash
# Evaluate one fine-tuned CLRerNet checkpoint, same protocol as the baseline:
#  (1) author tools/test.py on the full CULane test (34,680) -> predictions copied to $OUT/test_pred
#  (2) robustness subset (2,895 imgs x 16 conditions) into the shared robustness root as model dir <tag>
# Env setup mirrors gpu_entry.sh for clrernet (not edited while training jobs run it).
set -uo pipefail
B=/path/to/Lane_Detection/benchmark_culane; RUN="$1"; TAG="$2"; CKPT="$RUN/epoch_3.pth"
[ -s "$CKPT" ] || { echo "FAIL: no $CKPT"; exit 10; }
source $B/.venvs/clrernet/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCH_HOME=$B/.cache/torch CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$B/vendors/clrernet:$B/vendors/clrernet/libs/models/layers/nms/src:${PYTHONPATH:-}"
nvidia-smi --query-gpu=name --format=csv,noheader; sha256sum "$CKPT"
OUT="$RUN/eval"; mkdir -p "$OUT"
SHORT_TMP="$(mktemp -d /tmp/clrev.XXXX)"; export TMPDIR="$SHORT_TMP"
cd $B/vendors/clrernet
python tools/test.py configs/clrernet/culane/clrernet_culane_dla34_ema.py "$CKPT" --work-dir "$OUT/test_workdir" || exit 11
PRED_SRC=""; for d in "$SHORT_TMP"/tmp*; do [[ -d "$d/driver_100_30frame" ]] && PRED_SRC="$d"; done
[[ -n "$PRED_SRC" ]] || { echo "FAIL: no metric dir"; exit 12; }
cp -r "$PRED_SRC" "$OUT/test_pred"; n=$(find "$OUT/test_pred" -name '*.lines.txt' | wc -l); echo "TEST_PRED_FILES=$n"
[[ "$n" -eq 34680 ]] || exit 13
unset TMPDIR CUBLAS_WORKSPACE_CONFIG   # robustness = plain inference, as for the baseline
cd $B/scripts
CLR_CKPT="$CKPT" python robustness.py clrernet $B/results/robustness/full_20261119T154132Z/$TAG || exit 14
echo "FT_EVAL DONE $TAG"
