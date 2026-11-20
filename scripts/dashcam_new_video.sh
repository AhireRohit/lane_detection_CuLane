#!/usr/bin/env bash
# ONE-STEP pipeline for a NEW dashcam video (independent test). Usage (via jobs/dashcam/new_video.sub):
#   VIDEO=<path to .mp4> TAG=<short name> condor_submit jobs/dashcam/new_video.sub
# Steps: squash baseline (3 models) -> vanishing-point crop calibration (label-free) -> crop baseline (3 models)
#        -> comparisons -> B0-B3 ablation with thresholds from CULane VAL predictions (nothing tuned on this video).
set -uo pipefail
B=/path/to/Lane_Detection/benchmark_culane; PY=$B/.venvs/scnn/bin/python
VIDEO="$(realpath "${VIDEO:?set VIDEO}")"; TAG="${TAG:-$(basename "${VIDEO%.*}")}"
export DASH_VIDEO="$VIDEO" DASH_PROVENANCE="${DASH_PROVENANCE:-user-provided dashcam recording ($TAG)}"
TS=$(date -u +%Y%m%dT%H%M%SZ); SQ=new_${TAG}_squash_$TS; CR=new_${TAG}_crop_$TS
# B1 thresholds from CULane VAL predictions (jobs/b1_recal/valpred.sub); newest run unless B1_CAL is set.
B1_CAL="${B1_CAL:-$(ls -1d $B/results/b1_recal/valpred_*/b1_calibration_pred.json 2>/dev/null | tail -1)}"
[ -s "$B1_CAL" ] || { echo "no B1 calibration found: run jobs/b1_recal/valpred.sub first or set B1_CAL"; exit 6; }
echo "VIDEO=$VIDEO SQ=$SQ CR=$CR B1_CAL=$B1_CAL"; sha256sum "$VIDEO"
for m in scnn ufldv2 clrernet; do DASH_BATCH=$SQ bash $B/scripts/gpu_entry.sh $m dashcam || { echo "FAIL squash $m"; exit 1; }; done
$PY $B/scripts/estimate_vp.py $SQ $B/results/adapter_v2/crop_calibration_${TAG}_$TS.json || exit 2
for m in scnn ufldv2 clrernet; do DASH_BATCH=$CR DASH_CROP=$B/results/adapter_v2/crop_calibration_${TAG}_$TS.json bash $B/scripts/gpu_entry.sh $m dashcam || { echo "FAIL crop $m"; exit 3; }; done
for b in $SQ $CR; do $PY $B/scripts/dashcam_compare.py $B/results/comparison/$b $B/results/dashcam/scnn/$b $B/results/dashcam/ufldv2/$b $B/results/dashcam/clrernet/$b || exit 4; done
$PY $B/scripts/temporal_ablation.py $CR $B/results/ablation/new_${TAG}_$TS "$B1_CAL" || exit 5
echo "NEW VIDEO PIPELINE DONE: comparison/$SQ comparison/$CR ablation/new_${TAG}_$TS"
