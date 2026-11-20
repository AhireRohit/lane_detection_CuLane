#!/usr/bin/env python3
"""Check: chunked (8 parallel jobs) frame diffs == single-pass frame_diffs.py output (now data_aux/train_diffs.npz)."""
from pathlib import Path
import numpy as np
O = Path(__file__).resolve().parents[1] / 'data_aux'
chunks = np.concatenate([np.load(O / f'diff_chunk_{k:02d}.npz')['data'] for k in range(8)])
single = np.load(O / 'train_diffs.npz')['data']
print('lengths', len(chunks), len(single), 'max_abs_diff', float(np.abs(chunks - single).max()),
      'identical_keep_mask', bool(((chunks >= 15) == (single >= 15)).all()), 'kept', int((single >= 15).sum()))
