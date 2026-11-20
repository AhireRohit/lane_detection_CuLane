#!/usr/bin/env bash
# Controlled A/B on ONE GPU in ONE job: CLRerNet latency with vs without CUBLAS_WORKSPACE_CONFIG (order: without, with, without).
set -uo pipefail
B=/path/to/Lane_Detection/benchmark_culane
source $B/.venvs/clrernet/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCH_HOME=$B/.cache/torch
export PYTHONPATH="$B/vendors/clrernet:$B/vendors/clrernet/libs/models/layers/nms/src:${PYTHONPATH:-}"
R=$B/results/speed/cublas_ab_$1; mkdir -p $R; nvidia-smi --query-gpu=name,clocks.sm --format=csv,noheader
cd $B/scripts
i=0
for mode in off on off2; do
  i=$((i+1))
  if [[ $mode == on ]]; then export CUBLAS_WORKSPACE_CONFIG=:4096:8; else unset CUBLAS_WORKSPACE_CONFIG; fi
  echo "RUN $i mode=$mode CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-unset}"
  python benchmark_speed.py clrernet $R/${i}_$mode > $R/${i}_$mode.log 2>&1 || echo "FAIL $mode"
  python -c "import json;j=json.load(open('$R/${i}_$mode/speed_manifest.json'));r=j['results'];print('$mode', 'fwd_med',round(r['forward_ms']['median_ms'],2),'full_med',round(r['full_ms']['median_ms'],2),'mem',round(j['peak_cuda_mem_allocated_mb']),'load',j['loadavg'][:9])"
done
