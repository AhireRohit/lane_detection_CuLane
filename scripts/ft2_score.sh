#!/usr/bin/env bash
# CPU, overnight round: official C++ scoring of the 5 new fine-tunes' full-test predictions, and their robustness
# conditions, all in parallel; then one robustness summary across all 10 models.
# Usage: ft2_score.sh <run_dir> ... (each run_dir = results/finetune/<name>_<ts>_<host>, with eval/test_pred)
set -uo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)"; PY=$B/.venvs/scnn/bin/python; rc=0
R=$B/results/robustness/full_20261119T154132Z
[ -f $R/robustness_5models.csv ] || cp $R/robustness.csv $R/robustness_5models.csv
[ -f $R/robustness_summary_5models.json ] || cp $R/robustness_summary.json $R/robustness_summary_5models.json
pids=(); tags=()
for r in "$@"; do
  t=$(basename "$r" | sed -E 's/_[0-9]{8}T[0-9]{6}Z_.*$//')   # clrernet_ft_control_s1 etc.
  tags+=("$t")
  ( bash $B/scripts/score_culane_official.sh "$t" "$r/eval/test_pred" > "$r/eval/official_score.log" 2>&1 ) & pids+=($!)
  ( cd $B/scripts && $PY score_robustness_one.py $R "$t" > "$r/eval/robust_score.log" 2>&1 ) & pids+=($!)
done
for p in "${pids[@]}"; do wait $p || rc=1; done
for r in "$@"; do grep -h 'all_test' "$r/eval/official_score.log" | tail -1 | sed "s#^#$(basename $r): #"; tail -1 "$r/eval/robust_score.log"; done
# every model that has a robustness dir (baselines + all fine-tunes so far), deduplicated
M=$( { printf '%s\n' scnn ufldv2 clrernet; ls -1d $R/clrernet_ft_*/ 2>/dev/null | xargs -n1 basename; printf '%s\n' "${tags[@]}"; } | awk '!s[$0]++' | paste -sd, )
cd $B/scripts && $PY score_robustness.py $R --models $M || rc=1
echo FT2_SCORE_RC=$rc; exit $rc
