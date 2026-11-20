#!/usr/bin/env bash
# Runs the three latency benchmarks sequentially in ONE container on ONE GPU (same hardware, no co-scheduling).
B="$(cd "$(dirname "$0")/.." && pwd)"
export SPEED_BATCH="run_$(date -u +%Y%m%dT%H%M%SZ)_${HOSTNAME%%.*}"
echo "SPEED_BATCH=$SPEED_BATCH"; nvidia-smi
rc_all=0
for m in scnn ufldv2 clrernet; do
  bash "$B/scripts/gpu_entry.sh" "$m" speed; rc=$?
  echo "SPEED_RC $m=$rc"; [[ $rc -eq 0 ]] || rc_all=1
done
exit $rc_all
