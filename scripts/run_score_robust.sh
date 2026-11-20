#!/usr/bin/env bash
/path/to/Lane_Detection/benchmark_culane/.venvs/scnn/bin/python /path/to/Lane_Detection/benchmark_culane/scripts/score_robustness.py "$1" ${2:+--limit $2}
