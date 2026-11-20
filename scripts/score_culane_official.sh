#!/usr/bin/env bash
# Official CULane C++ evaluator (SCNN/UFLDv2 upstream copy, unmodified source) on existing predictions.
# Usage: score_culane_official.sh <model> <pred_dir>
# Settings follow upstream run-full.sh: width 30, IoU 0.5, 1640x590, frame 1.
# Fix vs. earlier invalid run (job 73767): dirs passed WITH trailing slash, because
# evaluate.cpp builds paths as anno_dir + list_entry and list entries have no leading '/'.
set -euo pipefail
MODEL="$1"; PRED="${2%/}/"
R=/path/to/Lane_Detection
B="$R/benchmark_culane"
DATA="$R/datasets/CULane/"
EVAL="$B/vendors/ufldv2/evaluation/culane/build_conda_gcc/culane_evaluator"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_${_CONDOR_JOB_AD:+c}$$"
OUT="$B/results/$MODEL/official_eval/$RUN_ID"
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$B/.opencv_cpp/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
W=30; IOU=0.5; IW=1640; IH=590

exec > >(tee -a "$OUT/score_stdout.log") 2> >(tee -a "$OUT/score_stderr.log" >&2)
echo "MODEL=$MODEL PRED=$PRED HOST=$(hostname) DATE=$(date -u +%FT%TZ)"
echo "evaluator=$EVAL sha256=$(sha256sum "$EVAL" | cut -d' ' -f1)"
echo "ufldv2 vendor commit=$(git -C "$B/vendors/ufldv2" rev-parse HEAD)"
[[ -x "$EVAL" ]] || { echo "FAIL: evaluator missing" >&2; exit 3; }

# --- Preflight: exact paths the evaluator will build must exist ---
miss=0; n=0
for L in "$DATA"list/test_split/test*_*.txt; do
  while read -r e _; do
    n=$((n+1)); s="${e%.*}.lines.txt"
    [[ -f "$DATA$s" ]] || { echo "MISSING GT: $DATA$s" >&2; miss=$((miss+1)); }
    [[ -f "$PRED$s" ]] || { echo "MISSING PRED: $PRED$s" >&2; miss=$((miss+1)); }
  done < <(awk 'NR%200==1' "$L")
done
echo "preflight sampled=$n missing=$miss"
[[ $miss -eq 0 ]] || { echo "FAIL: preflight missing files" >&2; exit 4; }

run_eval() { # list pred out
  "$EVAL" -a "$DATA" -d "$2" -i "$DATA" -l "$1" -w $W -t $IOU -c $IW -r $IH -f 1 -o "$3" >/dev/null
  cat "$3"
}

# --- Self-check: GT scored against itself must give F1 = 1 ---
run_eval "$DATA"list/test_split/test6_curve.txt "$DATA" "$OUT/selfcheck_gt_vs_gt_curve.txt"
grep -q '^Fmeasure: 1$' "$OUT/selfcheck_gt_vs_gt_curve.txt" || { echo "FAIL: GT-vs-GT self-check is not F1=1" >&2; exit 5; }
echo "SELFCHECK PASS"

# --- Overall + nine scenarios ---
run_eval "$DATA"list/test.txt "$PRED" "$OUT/all_test.txt"
for L in "$DATA"list/test_split/test*_*.txt; do
  name="$(basename "$L" .txt)"
  run_eval "$L" "$PRED" "$OUT/$name.txt"
done

# --- Sanity gate: GT must be found in every scenario except cross (no GT lanes by design) ---
python3 - "$OUT" <<'PY'
import sys, re, glob, json, os
out = sys.argv[1]; rows = {}
for f in sorted(glob.glob(os.path.join(out, "*.txt"))):
    if "selfcheck" in f: continue
    t = open(f).read()
    tp, fp, fn = map(int, re.search(r"tp: (\d+) fp: (\d+) fn: (\d+)", t).groups())
    p = tp/(tp+fp) if tp+fp else None; r = tp/(tp+fn) if tp+fn else None
    f1 = 2*p*r/(p+r) if p and r else None
    rows[os.path.basename(f)[:-4]] = dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f1)
bad = [k for k, v in rows.items() if "cross" not in k and v["tp"] + v["fn"] == 0]
json.dump(dict(evaluator="CULane official C++ (ufldv2 vendor copy)", width_lane=30, iou=0.5,
               im_w=1640, im_h=590, results=rows, sanity_fail=bad), open(os.path.join(out, "official_eval.json"), "w"), indent=2)
for k, v in rows.items():
    f1 = "N/A (FP only)" if "cross" in k else (f"{100*v['f1']:.2f}" if v["f1"] is not None else "None")
    print(f"{k:16s} tp={v['tp']:6d} fp={v['fp']:6d} fn={v['fn']:6d} F1={f1}")
if bad: print("FAIL: no GT lanes found for", bad); sys.exit(6)
PY
echo "OFFICIAL EVAL COMPLETE: $OUT"
