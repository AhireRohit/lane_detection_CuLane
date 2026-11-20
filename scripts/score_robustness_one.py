#!/usr/bin/env python3
"""Compute the official-evaluator eval.txt for ONE model's 16 robustness conditions (for parallel scoring).

Same evaluator call and image list as score_robustness.py; the list file is written per model, so several of these can
run at the same time. Afterwards run score_robustness.py once: it reuses every eval.txt that is newer than its
predictions and writes robustness.csv / robustness_summary.json (including the identical-input checksum check).
Usage: score_robustness_one.py <root> <model>
"""
import sys
from pathlib import Path
import score_robustness as S
from robust_common import subset, CONDS, cond_name

root, m = Path(sys.argv[1]).resolve(), sys.argv[2]
items = subset()
lst = root / m / 'subset_list.txt'
lst.write_text(''.join(e + '\n' for _, e, _ in items))
for c, s in CONDS:
    cn = cond_name(c, s)
    n = sum(1 for _ in (root / m / cn / 'pred').rglob('*.lines.txt'))
    assert n == len(items), f'{m} {cn}: {n} prediction files, expected {len(items)}'
    r = S.run(lst, root / m / cn / 'pred', root / m / cn / 'eval.txt')
    print(m, cn, round(r['f1'], 2), flush=True)
print('ONE DONE', m)
