#!/usr/bin/env bash
set -uo pipefail
B=/path/to/Lane_Detection/benchmark_culane
source $B/.venvs/ufldv2/bin/activate
export PYTHONUNBUFFERED=1 TORCH_HOME=$B/.cache/torch
cd $B/scripts
python verify_ufld_decode.py $B/results/ufld_decode_fix/verify_$1.json 50; v=$?
bash $B/scripts/gpu_entry.sh ufldv2 smoke5; s=$?
echo "VERIFY_RC=$v SMOKE_RC=$s"; exit $((v+s))
