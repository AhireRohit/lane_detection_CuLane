#!/usr/bin/env bash
# Assemble the Hugging Face Space folder from this repository.
# Usage: hf_space/make_space.sh <space_dir> <robust_ckpt.pth>
#   space_dir      : a clone of your Space repo (git clone https://huggingface.co/spaces/<user>/<space>)
#   robust_ckpt.pth: output of scripts/export_robust_ckpt.py (jobs/verify/export_robust.sub)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; S="${1:?space dir}"; CK="${2:?robust checkpoint}"
mkdir -p "$S/benchmark_culane/scripts" "$S/benchmark_culane/config" "$S/weights"
cp "$REPO"/hf_space/{app.py,setup_space.sh,requirements.txt,packages.txt,VERIFICATION.md} "$S/"
cp "$REPO/hf_space/SPACE_README.md" "$S/README.md"
for f in common.py benchmark_speed.py scnn_infer.py app_infer.py estimate_vp.py temporal_ablation.py nms_cpu.py; do
  cp "$REPO/scripts/$f" "$S/benchmark_culane/scripts/"
done
cp "$REPO"/config/{sources.json,checkpoints.sha256,culane_res34_cluster.py} "$S/benchmark_culane/config/"
cp "$CK" "$S/weights/clrernet_culane_dla34_robust_ft.pth"
(cd "$S" && sha256sum weights/clrernet_culane_dla34_robust_ft.pth > weights/SHA256SUMS)
grep -q 'a90b78d306a510ee4cd775239c6a68e0eb01a0aa163a849daf03e480abcfa174' "$S/weights/SHA256SUMS" \
  || { echo "robust checkpoint is not the verified export (sha256 a90b78d3...)"; exit 2; }
# keep the Space's own LFS rules; only add *.pth if missing
grep -qF '*.pth filter=lfs' "$S/.gitattributes" 2>/dev/null || printf '*.pth filter=lfs diff=lfs merge=lfs -text\n' >> "$S/.gitattributes"
echo "Space folder ready: $S  (commit and push it to Hugging Face)"
