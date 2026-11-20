#!/usr/bin/env bash
# CPU-only HTCondor job (docker image python:3.10-slim, no GPU): build the Hugging Face Space's environment from
# hf_space/requirements.txt and compare its CPU lanes with the GPU predictions for all four demo models.
# Usage (jobs/verify/verify_cpu.sub): verify_cpu_job.sh <N images per model>
set -uo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)"; N="${1:-200}"
V=$B/.venvs/space_cpu; OUT=$B/results/verify_cpu; mkdir -p $OUT
python3 --version; nproc
if [[ ! -f $V/.deps_ok ]]; then
  python3 -m venv $V && $V/bin/pip install -q --upgrade pip && $V/bin/pip install -q -r $B/hf_space/requirements.txt || exit 2
  # imgaug pulls the GUI opencv-python (needs libGL); keep only the headless build, as on the cluster (gpu_entry.sh)
  $V/bin/pip uninstall -q -y opencv-python opencv-contrib-python; $V/bin/pip install -q --no-deps --force-reinstall opencv-python-headless==4.10.0.84 || exit 3
  $V/bin/python -c 'import cv2, torch, mmcv, mmdet; print(cv2.__version__, torch.__version__, mmcv.__version__, mmdet.__version__)' || exit 4
  touch $V/.deps_ok
fi
$V/bin/pip freeze > $OUT/pip_freeze.txt
CKPT=$(ls -1 $B/results/finetune/clrernet_ft_corrupt_2*-73992/epoch_3.pth | head -1)
export PYTHONPATH="$B/vendors/clrernet:${PYTHONPATH:-}" OMP_NUM_THREADS=2 LANE_DEVICE=cpu
pids=()
for m in scnn ufldv2 clrernet clrernet_robust; do
  ( [[ $m == clrernet_robust ]] && export CLR_CKPT=$CKPT
    cd $B/scripts && $V/bin/python verify_cpu.py e2e $m $N $OUT/e2e_$m.json > $OUT/e2e_$m.log 2>&1 ) & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
for m in scnn ufldv2 clrernet clrernet_robust; do echo "== $m"; tail -14 $OUT/e2e_$m.log; done
echo VERIFY_CPU_RC=$rc; exit $rc
