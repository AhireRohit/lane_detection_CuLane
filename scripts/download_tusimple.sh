#!/usr/bin/env bash
# TuSimple lane test set (cross-dataset evaluation only; never trained on, never redistributed) from the ungated
# Hugging Face mirror dotblue/TuSimple (Apache-2.0, copy of github.com/TuSimple/tusimple-benchmark), pinned to a commit.
# The mirror holds a 10% sample of the clips (233 of the 2782 test clips, complete with all 20 frames). Only those
# are used; their labels are written to test_label_available.json.
# Usage: download_tusimple.sh label   -> test_label.json + the labeled frame (20) of each available test clip
#        download_tusimple.sh clips   -> also frames 1..19 of those clips (20 fps) for the tracker test
set -uo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)"; T="$B/../datasets/TuSimple"; MODE="${1:-label}"
REV=3ab1eb751e3036a83b1e40c33a1da7e77b240c84
U="https://huggingface.co/datasets/dotblue/TuSimple/resolve/$REV"
mkdir -p "$T"; cd "$T"
[[ -s test_label.json ]] || curl -sSfL --retry 5 -o test_label.json "$U/test_label.json" || exit 2
curl -sSfL --retry 5 -o siblings.json "https://huggingface.co/api/datasets/dotblue/TuSimple/revision/$REV" || exit 2
python3 - "$MODE" > want.txt <<'PY'
import json, sys
files = {s['rfilename'] for s in json.load(open('siblings.json'))['siblings']}
keep = []
for line in open('test_label.json'):
    d = json.loads(line)['raw_file'].rsplit('/', 1)[0]          # clips/<date>/<id>/20.jpg
    if all(f'{d}/{k}.jpg' in files for k in range(1, 21)):
        keep.append(line)
        for k in (range(1, 21) if sys.argv[1] == 'clips' else [20]): print(f'{d}/{k}.jpg')
open('test_label_available.json', 'w').write(''.join(keep))
print(len(keep), 'complete test clips in the mirror', file=sys.stderr)
PY
get() { [[ -s "$1" ]] && return 0; mkdir -p "$(dirname "$1")"
  for t in 1 2 3 4 5 6; do curl -sSfL -o "$1.part" "$U/$1" && mv "$1.part" "$1" && return 0; sleep $((t * 15)); done
  echo "FAILED $1"; return 1; }
export -f get; export U
xargs -a want.txt -P 6 -I{} bash -c 'get "$@"' _ {} > fails.txt 2>&1
n=$(wc -l < want.txt); f=$(grep -c FAILED fails.txt); have=$(while read -r p; do [[ -s $p ]] && echo; done < want.txt | wc -l)
echo "TUSIMPLE $MODE wanted=$n have=$have failed=$f"; [[ $have -eq $n ]]
