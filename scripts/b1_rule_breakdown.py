#!/usr/bin/env python3
"""Diagnostic only: which B1 rule rejects lanes on the video (per model). No parameter changes."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from temporal_ablation import resample, feats, crosses, CW, CH, MODELS, B
batch, out = sys.argv[1], Path(sys.argv[2])
th = json.loads((out / 'b1_calibration_culane_val.json').read_text())['thresholds']
res = {}
for m in MODELS:
    run = B / 'results/dashcam' / m / batch
    rep = json.loads((run / 'report.json').read_text()); sx, sy = rep['video_width'] / CW, rep['video_height'] / CH
    c = dict(lanes=0, extent=0, curv=0, slope=0, crossing_pairs=0); slopes = []
    for line in open(run / 'frames.jsonl'):
        L = [x for x in (resample([(a / sx, b / sy) for a, b in l]) for l in json.loads(line)['lanes']) if x is not None]
        for x in L:
            f = feats(x); c['lanes'] += 1; slopes.append(f['slope'])
            c['extent'] += f['extent'] < th['min_extent']; c['curv'] += f['curv'] > th['max_curv']; c['slope'] += f['slope'] > th['max_slope']
        c['crossing_pairs'] += sum(crosses(L[i], L[j]) for i in range(len(L)) for j in range(i + 1, len(L)))
    c['slope_quantiles_p50_p90_p99'] = np.percentile(slopes, [50, 90, 99]).round(2).tolist()
    res[m] = c
res['val_gt_threshold'] = th
(out / 'b1_rule_breakdown_video.json').write_text(json.dumps(res, indent=2, default=int)); print(json.dumps(res, indent=2, default=int))
