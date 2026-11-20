#!/usr/bin/env bash
# Write the evaluation DAG for the overnight fine-tune round once its training jobs have created their run dirs.
# Usage: make_ft2_dag.sh <training ClusterId>   ->  jobs/ft_dag2/ft2_eval.dag (+ score2.sub); then condor_submit_dag it.
# Each EVAL node waits (PRE script on the access point) for its epoch_3.pth, then runs ft_eval.sh (full test + robustness).
set -euo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)"; CID="${1:?training ClusterId}"
D=$B/jobs/ft_dag2; mkdir -p $D $B/logs/ft_dag2
mapfile -t RUNS < <(ls -d $B/results/finetune/clrernet_ft_*_20*-"$CID")
[[ ${#RUNS[@]} -gt 0 ]] || { echo "no run dirs for $CID"; exit 1; }
sed -e "s#ft_score.sh#ft2_score.sh ${RUNS[*]}#" -e 's#logs/ft_dag/score#logs/ft_dag2/score#' -e 's#logs/ft_dag/ft_dag.log#logs/ft_dag2/ft_dag2.log#' \
    -e 's#request_cpus = 4#request_cpus = 10#' -e 's#OMP_NUM_THREADS=4#OMP_NUM_THREADS=1#' $B/jobs/ft_dag/score.sub > $D/score2.sub
sed -e 's#logs/ft_dag/#logs/ft_dag2/#g' -e 's#ft_dag.log#ft_dag2.log#' $B/jobs/ft_dag/eval.sub > $D/eval2.sub
{
  echo "# Overnight fine-tune round (training cluster $CID): evaluate each run, then score all in parallel."
  parents=()
  for r in "${RUNS[@]}"; do
    t=$(basename "$r" | sed -E 's/_[0-9]{8}T[0-9]{6}Z_.*$//'); n=EVAL_$(echo "$t" | sed 's/clrernet_ft_//' | tr a-z A-Z)
    echo "JOB $n $D/eval2.sub"
    echo "VARS $n run=\"$r\" tag=\"$t\""
    echo "SCRIPT PRE $n $B/scripts/wait_for_file.sh $r/epoch_3.pth 14"
    parents+=("$n")
  done
  echo "JOB SCORE2 $D/score2.sub"
  echo "PARENT ${parents[*]} CHILD SCORE2"
  echo "RETRY SCORE2 1"
} > $D/ft2_eval.dag
cat $D/ft2_eval.dag
