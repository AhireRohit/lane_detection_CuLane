#!/usr/bin/env python3
"""Verify the fixed in-memory UFLDv2 decoder against the author's own complete non-TTA test output
(results/ufldv2/full_notta/culane_eval_tmp, written by vendor run_test via test.py) on N test images.
(a) PIL loader (author's loader)  -> isolates the decoder;  (b) cv2 BGR loader (our video/robustness path).
Usage: verify_ufld_decode.py <out_json> [N]
"""
import sys, json
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
from common import B, D
from robust_common import subset
from benchmark_speed import build_ufld

out = Path(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
REF = B / 'results/ufldv2/full_notta/culane_eval_tmp'
pre, fwd, post, info, verify = build_ufld(out.parent)
import torchvision.transforms as T
from utils.common import merge_config   # already configured by build_ufld (sys.argv set there)
tf = T.Compose([T.Resize((533, 1600)), T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225))])


def read(p):
    return [list(zip(*[iter(map(float, r.split()))] * 2)) for r in Path(p).read_text().splitlines() if r.split()]


def cmp(ref, mine):
    if len(ref) != len(mine) or any(len(a) != len(b) for a, b in zip(ref, mine)):
        return False, None
    d = max((abs(x1 - x2) + abs(y1 - y2) for a, b in zip(ref, mine) for (x1, y1), (x2, y2) in zip(a, b)), default=0.0)
    return True, d


res = dict(pil=[], cv2=[]); counts = []
items = [x for x in subset() if x[0] != 'test7_cross'][:N]
with torch.no_grad():
    for _, _, rel in items:
        ref = read(REF / (rel[:-3] + 'lines.txt'))
        x_pil = tf(Image.open(D / rel).convert('RGB'))[:, -320:, :].unsqueeze(0).cuda()
        mine_pil = post(fwd(x_pil))
        mine_cv = post(fwd(pre(cv2.imread(str(D / rel)))))
        res['pil'].append(cmp(ref, mine_pil)); res['cv2'].append(cmp(ref, mine_cv))
        counts.append((len(ref), len(mine_pil), len(mine_cv)))
summ = {}
for k, v in res.items():
    ok = [d for same, d in v if same]
    summ[k] = dict(images=len(v), same_lane_and_point_counts=len(ok), max_abs_diff_px=max(ok) if ok else None,
                   median_max_diff_px=float(np.median(ok)) if ok else None)
summ['mean_lanes_ref_pil_cv2'] = np.mean(counts, 0).round(3).tolist()
summ['frame0_verify'] = verify(cv2.imread(str(D / items[0][2])))
summ['note'] = 'reference = author run_test output (row lanes 1,2 + column lanes 0,3), test.py job 73894'
out.write_text(json.dumps(summ, indent=2, default=float)); print(json.dumps(summ, indent=2, default=float))
ok = summ['pil']['same_lane_and_point_counts'] == summ['pil']['images'] and summ['pil']['max_abs_diff_px'] < 0.01
print('VERIFY', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
