#!/usr/bin/env bash
# GPU job for the Streamlit demo app (jobs/app/infer.sub): runs app_infer.py once per requested model,
# each in that model's own venv. APP_JOB = job directory holding request.json and the uploaded input.
set -uo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)"
J="${APP_JOB:?set APP_JOB}"
[[ -f "$J/request.json" ]] || { echo "no request.json in $J"; exit 2; }
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCH_HOME="$B/.cache/torch"
# Our corruption-fine-tuned CLRerNet (report §7.1); override with ROBUST_CKPT if needed.
ROBUST_CKPT="${ROBUST_CKPT:-$(ls -1 "$B"/results/finetune/clrernet_ft_corrupt_*/epoch_3.pth 2>/dev/null | head -1)}"
echo "HOST=$(hostname) GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader) JOB=$J"
MODELS=$(python -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['models']))" "$J/request.json")
rc=0
for m in $MODELS; do
  case "$m" in scnn|ufldv2) V=$m;; clrernet|clrernet_robust) V=clrernet;; *) echo "unknown model $m"; rc=3; continue;; esac
  (
    source "$B/.venvs/$V/bin/activate"
    [[ "$V" == clrernet ]] && export PYTHONPATH="$B/vendors/clrernet:$B/vendors/clrernet/libs/models/layers/nms/src:${PYTHONPATH:-}"
    if [[ "$m" == clrernet_robust ]]; then
      [[ -s "$ROBUST_CKPT" ]] || { echo "robust checkpoint not found"; exit 4; }
      export CLR_CKPT="$ROBUST_CKPT"
    fi
    cd "$B/scripts" && python app_infer.py "$J" "$m"
  ) > "$J/log_$m.txt" 2>&1 || { echo "FAIL $m (see log_$m.txt)"; tail -5 "$J/log_$m.txt"; rc=1; }
done
if [[ $rc -eq 0 ]]; then touch "$J/DONE"; else echo "$rc" > "$J/FAILED"; fi
echo "APP_JOB rc=$rc"
exit $rc
