#!/usr/bin/env bash
B=/path/to/Lane_Detection/benchmark_culane
"$B/.venvs/scnn/bin/python" "$B/scripts/env_manifest.py"; a=$?
"$B/.venvs/scnn/bin/python" "$B/scripts/compare_annotations_new.py"; b=$?
exit $((a+b))
