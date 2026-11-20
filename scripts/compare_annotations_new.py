#!/usr/bin/env python3
"""Compare CULane annotations_new/*.lines.txt with the same paths under the dataset root. Read-only.
Writes reports/annotations_new_comparison.json: identical / different / missing counts per driver and per split."""
import json, hashlib
from pathlib import Path
B = Path(__file__).resolve().parents[1]; D = B.parent / 'datasets/CULane'; A = D / 'annotations_new'
split = {}
for name, f in (('train', 'train_gt.txt'), ('val', 'val.txt'), ('test', 'test.txt')):
    p = D / 'list' / f
    for l in open(p):
        if l.strip(): split[l.split()[0].lstrip('/')[:-3] + 'lines.txt'] = name
def h(p): return hashlib.md5(p.read_bytes()).hexdigest()
res, examples = {}, []
for f in sorted(A.rglob('*.lines.txt')):
    rel = str(f.relative_to(A)); drv = rel.split('/')[0]; o = D / rel
    k = 'missing_in_root' if not o.is_file() else ('identical' if h(f) == h(o) else 'different')
    s = split.get(rel, 'not_in_lists')
    r = res.setdefault(drv, {}); r.setdefault(k, {}); r[k][s] = r[k].get(s, 0) + 1
    if k == 'different' and len(examples) < 10:
        examples.append(dict(file=rel, split=s, lanes_new=len(f.read_text().splitlines()), lanes_root=len(o.read_text().splitlines())))
tot = {}
for drv, r in res.items():
    for k, d in r.items():
        for s, n in d.items(): tot.setdefault(k, {}); tot[k][s] = tot[k].get(s, 0) + n
out = dict(annotations_new=str(A.relative_to(D.parent)), per_driver=res, totals=tot, different_examples=examples,
           note='byte comparison only; no file was modified. Training/eval in this project used the dataset-root labels.')
(B / 'reports/annotations_new_comparison.json').write_text(json.dumps(out, indent=2)); print(json.dumps(dict(totals=tot, examples=examples[:5]), indent=1))
