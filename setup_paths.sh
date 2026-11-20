#!/usr/bin/env bash
# Replace the placeholder project root in configs, scripts and job files with your own path.
# Usage: ./setup_paths.sh /absolute/path/to/Lane_Detection   (folder that contains benchmark_culane/ and datasets/CULane/)
set -euo pipefail
ROOT="${1:?usage: ./setup_paths.sh /absolute/path/to/Lane_Detection}"
grep -rl "/path/to/Lane_Detection" --include='*.py' --include='*.sh' --include='*.sub' --include='*.dag' --include='*.json' . \
  | xargs sed -i "s#/path/to/Lane_Detection#${ROOT//#/\#}#g"
echo "Paths set to $ROOT"
