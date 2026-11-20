#!/usr/bin/env bash
# DAGMan PRE script (runs on the access point, sleeps only): wait until a file exists, at most MAX_H hours.
f="$1"; max_h="${2:-12}"; t=0
until [ -s "$f" ]; do sleep 300; t=$((t+300)); [ $t -ge $((max_h*3600)) ] && { echo "timeout waiting for $f"; exit 1; }; done
sleep 120   # let the training job finish writing / exit
echo "found $f"
