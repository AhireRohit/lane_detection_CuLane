#!/usr/bin/env bash
# One-time setup on the cluster login node (light work only): clone the three model repos at the EXACT commits
# used for every result in reports/, link the dataset for CLRerNet, write the UFLDv2 config, audit the dataset,
# and create the small train-list used by the fine-tune smoke test. Vendor code is never modified.
set -euo pipefail
B="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$(dirname "$B")"
mkdir -p "$B/results" "$B/logs" "$B/data_aux"
cd "$R"
for model in scnn ufldv2 clrernet; do
  case "$model" in
    scnn)     url=https://github.com/delldu/SCNN.git;                          sha=a6fadfbd9e593a28684dcaf29ca8ad425c1c700f;;
    ufldv2)   url=https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2.git;    sha=c903880678454dfd9b55a63022368db05c00bc6d;;
    clrernet) url=https://github.com/hirotomusiker/CLRerNet.git;               sha=dae038f67da57e292e5293a68c9c1c2922de13c2;;
  esac
  target="$B/vendors/$model"
  if [[ -d "$target/.git" ]]; then echo "KEEP existing $target";
  elif [[ -e "$target" ]]; then echo "ERROR non-git path exists $target" >&2; exit 2;
  else
    echo "CLONE $url @ $sha"
    git clone --quiet "$url" "$target"
    git -C "$target" checkout --quiet "$sha"
  fi
  head=$(git -C "$target" rev-parse HEAD)
  [[ "$head" == "$sha" ]] || { echo "ERROR $model is at $head, expected pinned $sha" >&2; exit 4; }
  [[ -z "$(git -C "$target" status --porcelain --untracked-files=no)" ]] || { echo "ERROR $model has local modifications" >&2; exit 5; }
  echo "$head" > "$B/results/${model}_commit.txt"; echo "OK $model $head"
done
# Read-only symlink for CLRerNet's expected dataset location; never move or copy the dataset.
if [[ -e "$B/vendors/clrernet/dataset/culane" && ! -L "$B/vendors/clrernet/dataset/culane" ]]; then
  echo 'CLRerNet dataset/culane already exists, refusing to overwrite.' >&2; exit 3
fi
mkdir -p "$B/vendors/clrernet/dataset"
ln -sfn "$R/datasets/CULane" "$B/vendors/clrernet/dataset/culane"
# UFLDv2 test config lives outside the vendor tree (config/culane_res34_cluster.py is in this repo; setup_paths.sh
# fills in its data_root). Only if it is missing, recreate it from the upstream config.
if [[ ! -f "$B/config/culane_res34_cluster.py" ]]; then
  cp "$B/vendors/ufldv2/configs/culane_res34.py" "$B/config/culane_res34_cluster.py"
  export B R
  python3 - <<'PY'
import os, re
from pathlib import Path
p = Path(os.environ['B']) / 'config/culane_res34_cluster.py'
s, n = re.subn(r"(?m)^data_root\s*=.*$", "data_root = " + repr(str(Path(os.environ['R']) / 'datasets/CULane')), p.read_text(), count=1)
if n != 1: raise SystemExit('Cannot locate data_root in UFLD config')
p.write_text(s); print('Wrote', p)
PY
fi
python3 "$B/scripts/audit.py"
# Fine-tune smoke test list = first 240 lines of the official train list.
head -240 "$R/datasets/CULane/list/train_gt.txt" > "$B/data_aux/train_gt_smoke.txt"
echo "BOOTSTRAP OK"
