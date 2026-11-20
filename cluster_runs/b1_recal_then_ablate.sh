#!/usr/bin/env bash
set -euo pipefail
/path/to/Lane_Detection/benchmark_culane/.venvs/scnn/bin/python /path/to/Lane_Detection/benchmark_culane/scripts/b1_calibrate_preds.py /path/to/Lane_Detection/benchmark_culane/results/b1_recal/valpred_20261119T145622Z
/path/to/Lane_Detection/benchmark_culane/.venvs/scnn/bin/python /path/to/Lane_Detection/benchmark_culane/scripts/temporal_ablation.py run_20261119T144052Z_<cluster-user>-73882 /path/to/Lane_Detection/benchmark_culane/results/ablation/temporal_v2_$1 /path/to/Lane_Detection/benchmark_culane/results/b1_recal/valpred_20261119T145622Z/b1_calibration_pred.json
/path/to/Lane_Detection/benchmark_culane/.venvs/scnn/bin/python - <<'P2'
import sys; sys.path.insert(0, "/path/to/Lane_Detection/benchmark_culane/scripts")
P2
