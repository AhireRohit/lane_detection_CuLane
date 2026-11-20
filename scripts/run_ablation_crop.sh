#!/usr/bin/env bash
/path/to/Lane_Detection/benchmark_culane/.venvs/scnn/bin/python /path/to/Lane_Detection/benchmark_culane/scripts/temporal_ablation.py run_20261119T154149Z_<cluster-user>-73901_crop /path/to/Lane_Detection/benchmark_culane/results/ablation/temporal_crop_$1 /path/to/Lane_Detection/benchmark_culane/results/b1_recal/valpred_20261119T145622Z/b1_calibration_pred.json
