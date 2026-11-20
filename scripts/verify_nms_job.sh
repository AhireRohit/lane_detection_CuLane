#!/usr/bin/env bash
# GPU HTCondor job: compare the CPU lane-NMS port (scripts/nms_cpu.py) with CLRerNet's CUDA op on identical proposals.
# Usage (jobs/verify/verify_nms.sub): verify_nms_job.sh <N images>
set -uo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)"; N="${1:-2000}"; OUT=$B/results/verify_cpu; mkdir -p $OUT
source $B/.venvs/clrernet/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 TORCH_HOME=$B/.cache/torch
export PYTHONPATH="$B/vendors/clrernet:$B/vendors/clrernet/libs/models/layers/nms/src:${PYTHONPATH:-}"
nvidia-smi --query-gpu=name --format=csv,noheader
cd $B/scripts && python verify_cpu.py nms $N $OUT/nms_check.json
