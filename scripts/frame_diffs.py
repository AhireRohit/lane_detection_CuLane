#!/usr/bin/env python3
"""Same computation as vendors/clrernet/tools/calculate_frame_diff.py (mean abs pixel difference to the previous
train_gt frame), but the output goes to benchmark_culane/data_aux/train_diffs.npz so the CULane folder stays untouched."""
from pathlib import Path
import numpy as np, cv2
B = Path(__file__).resolve().parents[1]; D = B.parent / 'datasets/CULane'
lines = open(D / 'list/train_gt.txt', 'rb').readlines()
prev = np.zeros(1); diffs = []
for i, line in enumerate(lines):
    img = cv2.imread(str(D / line.decode('utf8').strip().split(' ')[0][1:]))
    diffs.append(np.abs(img.astype(np.float32) - prev.astype(np.float32)).sum() / (img.shape[0] * img.shape[1] * img.shape[2]))
    prev = img
    if i % 10000 == 0: print(i, flush=True)
np.savez(B / 'data_aux/train_diffs', data=diffs)
d = np.array(diffs); print('frames', len(d), 'kept (diff>=15)', int((d >= 15).sum()))
