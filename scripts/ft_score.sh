#!/usr/bin/env bash
# CPU: official C++ scoring of both fine-tuned test predictions + robustness scoring next to the baselines.
set -uo pipefail
B=/path/to/Lane_Detection/benchmark_culane; rc=0
for r in $B/results/finetune/clrernet_ft_control_20261119T181813Z_*-73992 $B/results/finetune/clrernet_ft_corrupt_20261119T181814Z_*-73992; do
  t=$(basename $r | cut -d_ -f1-3)
  bash $B/scripts/score_culane_official.sh $t $r/eval/test_pred || rc=1
done
R=$B/results/robustness/full_20261119T154132Z
[ -f $R/robustness_3models.csv ] || cp $R/robustness.csv $R/robustness_3models.csv
[ -f $R/robustness_summary_3models.json ] || cp $R/robustness_summary.json $R/robustness_summary_3models.json
$B/.venvs/scnn/bin/python $B/scripts/score_robustness.py $R --models scnn,ufldv2,clrernet,clrernet_ft_control,clrernet_ft_corrupt || rc=1
echo FT_SCORE_RC=$rc; exit $rc
