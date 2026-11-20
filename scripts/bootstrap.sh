#!/usr/bin/env bash
set -euo pipefail
B="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$(dirname "$B")"
cd "$R"
for model in scnn ufldv2 clrernet; do
  case "$model" in
    scnn) url=https://github.com/delldu/SCNN.git;;
    ufldv2) url=https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2.git;;
    clrernet) url=https://github.com/hirotomusiker/CLRerNet.git;;
  esac
  target="$B/vendors/$model"
  if [[ -d "$target/.git" ]]; then echo "KEEP existing $target";
  elif [[ -e "$target" ]]; then echo "ERROR non-git path exists $target" >&2; exit 2;
  else echo "CLONE $url"; git clone --depth 1 "$url" "$target"; fi
  git -C "$target" rev-parse HEAD | tee "$B/results/${model}_commit.txt"
done
# Carefully add read-only symlinks; never move or copy source dataset.
if [[ -e "$B/vendors/clrernet/dataset/culane" && ! -L "$B/vendors/clrernet/dataset/culane" ]]; then
  echo 'CLRerNet dataset/culane already exists, refusing to overwrite.' >&2; exit 3
fi
mkdir -p "$B/vendors/clrernet/dataset"
ln -sfn "$R/datasets/CULane" "$B/vendors/clrernet/dataset/culane"
# UFLDv2 config copy is outside vendor tree; CI keeps upstream unchanged.
cp "$B/vendors/ufldv2/configs/culane_res34.py" "$B/config/culane_res34_cluster.py"
export B R
python3 - <<'PY'
import os
from pathlib import Path
p=Path(os.environ['B'])/'config/culane_res34_cluster.py'
s=p.read_text()
import re
s,count=re.subn(r"(?m)^data_root\s*=.*$", "data_root = "+repr(str(Path(os.environ['R'])/'datasets/CULane')), s, count=1)
if count != 1: raise SystemExit('Cannot locate data_root in UFLD config')
p.write_text(s)
print('Wrote',p)
PY
python3 "$B/scripts/audit.py"
# Avoid a redundant ImageNet backbone download when fully trained UFLDv2 weights are loaded.
# Apply only after exact upstream code check; record patch as reproducibility evidence.
UFLD_MODEL="$B/vendors/ufldv2/model/model_culane.py"
if grep -q 'return parsingNet(pretrained = True,' "$UFLD_MODEL"; then
  sed -i 's/return parsingNet(pretrained = True,/return parsingNet(pretrained = False,/' "$UFLD_MODEL"
  echo 'Patched UFLDv2 pretrained backbone initialization OFF (full checkpoint loaded strictly).'
fi
(cd "$B/vendors/ufldv2" && git diff -- model/model_culane.py > "$B/results/ufldv2_no_imagenet_download.patch")
