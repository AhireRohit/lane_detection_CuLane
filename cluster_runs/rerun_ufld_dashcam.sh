#!/usr/bin/env bash
set -uo pipefail
export DASH_VIDEO=/path/to/Lane_Detection/benchmark_culane/vendors/ufldv2/example.mp4
DASH_BATCH=run_20261119T144052Z_<cluster-user>-73882 bash /path/to/Lane_Detection/benchmark_culane/scripts/gpu_entry.sh ufldv2 dashcam; a=$?
DASH_BATCH=run_20261119T154149Z_<cluster-user>-73901_crop DASH_CROP=/path/to/Lane_Detection/benchmark_culane/results/adapter_v2/crop_calibration.json bash /path/to/Lane_Detection/benchmark_culane/scripts/gpu_entry.sh ufldv2 dashcam; b=$?
echo "RC squash=$a crop=$b"; exit $((a+b))
