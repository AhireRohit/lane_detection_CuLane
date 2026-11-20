#!/usr/bin/env python3
"""Chunked version of frame_diffs.py (identical values): diffs for train_gt lines [start, end), using line start-1 as the
previous frame (zeros for start 0, exactly as the vendor tool). Usage: frame_diffs_chunk.py <start> <end> | --merge <n_chunks>"""
import sys
from pathlib import Path
import numpy as np, cv2
B = Path(__file__).resolve().parents[1]; D = B.parent / 'datasets/CULane'; O = B / 'data_aux'
lines = open(D / 'list/train_gt.txt', 'rb').readlines()
rd = lambda i: cv2.imread(str(D / lines[i].decode('utf8').strip().split(' ')[0][1:]))
if sys.argv[1] == '--merge':
    n = int(sys.argv[2]); parts = [np.load(O / f'diff_chunk_{k:02d}.npz')['data'] for k in range(n)]
    d = np.concatenate(parts); assert len(d) == len(lines), (len(d), len(lines))
    np.savez(O / 'train_diffs', data=d); print('merged', len(d), 'kept (diff>=15)', int((d >= 15).sum())); sys.exit(0)
s, e = int(sys.argv[1]), int(sys.argv[2]); k = int(sys.argv[3])
prev = rd(s - 1) if s > 0 else np.zeros(1); out = []
for i in range(s, e):
    img = rd(i)
    out.append(np.abs(img.astype(np.float32) - prev.astype(np.float32)).sum() / (img.shape[0] * img.shape[1] * img.shape[2]))
    prev = img
np.savez(O / f'diff_chunk_{k:02d}', data=out); print('chunk', k, s, e, 'done')
