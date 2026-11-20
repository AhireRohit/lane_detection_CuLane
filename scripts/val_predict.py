#!/usr/bin/env python3
"""Predict lanes on CULane VAL (list/val.txt) with the same verified pre/forward/decode used for the dashcam video
(benchmark_speed.py builders; UFLDv2 non-TTA). Output: <out>/pred/<rel>.lines.txt in 1640x590 coordinates.
Used only to calibrate the B1 geometry filter on model predictions; never on test labels or the video.
Usage: val_predict.py <model> <out_dir>
"""
import sys, json, time
from pathlib import Path
import cv2
import torch
from common import D
from benchmark_speed import build_scnn, build_ufld, build_clr


def main():
    name, out = sys.argv[1], Path(sys.argv[2]).resolve()
    pred = out / 'pred'; pred.mkdir(parents=True, exist_ok=True)
    rels = [l.split()[0].lstrip('/') for l in open(D / 'list/val.txt') if l.strip()]
    if name == 'scnn': pre, fwd, post, info, _ = build_scnn()
    elif name == 'ufldv2': pre, fwd, post, info, _ = build_ufld(out)
    else: pre, fwd, post, info, _ = build_clr(out)
    t0 = time.time(); bad = 0
    with torch.no_grad():
        for i, rel in enumerate(rels):
            img = cv2.imread(str(D / rel))
            if img is None or img.shape != (590, 1640, 3):
                bad += 1; continue
            lanes = post(fwd(pre(img)))
            dst = pred / (rel[:-3] + 'lines.txt'); dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(''.join(' '.join(f'{float(x):.2f} {float(y):.2f}' for x, y in l) + '\n' for l in lanes if len(l) > 1))
            if i % 1000 == 0: print(f'{name} val {i}/{len(rels)} {time.time() - t0:.0f}s', flush=True)
    n = sum(1 for _ in pred.rglob('*.lines.txt'))
    (out / 'val_predict_manifest.json').write_text(json.dumps(dict(model=name, val_list_entries=len(rels), files=n, unreadable=bad,
                                                                   seconds=time.time() - t0, model_info=info), indent=2))
    print(f'VALPRED files={n} entries={len(rels)} unreadable={bad}')
    if n != len(rels): sys.exit(3)


if __name__ == '__main__':
    main()
