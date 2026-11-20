#!/usr/bin/env bash
# Official C++ scoring of CLRerNet's full-test predictions. Uses CLR_PRED if set, otherwise the newest
# results/clrernet/full/run_*/pred (the original run was job 73864).
set -euo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)"
PRED="${CLR_PRED:-$(ls -1d "$B"/results/clrernet/full/run_*/pred 2>/dev/null | tail -1)}"
[[ -d "$PRED" ]] || { echo "no CLRerNet predictions found; run the full CLRerNet job first or set CLR_PRED"; exit 2; }
exec bash "$B/scripts/score_culane_official.sh" clrernet "$PRED"
