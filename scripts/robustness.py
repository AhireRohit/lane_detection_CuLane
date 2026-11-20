#!/usr/bin/env python3
"""Corruption-robustness inference on a fixed CULane test subset (see robust_common.py).
Same verified pre/forward/decode as speed + dashcam (benchmark_speed.py builders; UFLDv2 non-TTA).
Writes <out>/<cond>/pred/<rel>.lines.txt (1640x590 coords), input checksums, and example images.
Usage: robustness.py <model> <out_dir> [--limit N]
"""
import sys, json, time, hashlib
from pathlib import Path
import numpy as np
import cv2
import torch
from robust_common import D, subset, CONDS, cond_name, corrupt, SEVERITY, STRIDE
from benchmark_speed import build_scnn, build_ufld, build_clr


def main():
    name, out = sys.argv[1], Path(sys.argv[2]).resolve()
    limit = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else None
    only = set(sys.argv[sys.argv.index('--conds') + 1].replace(';', ',').split(',')) if '--conds' in sys.argv else None
    if only is not None:
        unknown = only - {cond_name(c, s) for c, s in CONDS}
        assert not unknown, f'unknown conditions: {unknown}'
    tag = '_' + sys.argv[sys.argv.index('--tag') + 1] if '--tag' in sys.argv else ''
    out.mkdir(parents=True, exist_ok=True)
    items = subset()
    if limit: items = items[:limit]
    conds = [(c, s) for c, s in CONDS if only is None or cond_name(c, s) in only]
    if name == 'scnn': pre, fwd, post, info, _ = build_scnn()
    elif name == 'ufldv2': pre, fwd, post, info, _ = build_ufld(out)
    else: pre, fwd, post, info, _ = build_clr(out)
    md5 = {}; ex = []; t0 = time.time()
    with torch.no_grad():
        for c, s in conds:
            cn = cond_name(c, s); pdir = out / cn / 'pred'
            done = pdir.is_dir() and sum(1 for _ in pdir.rglob('*.lines.txt')) == len(items)
            for idx, (scen, _, rel) in enumerate(items):
                if done and idx >= 3: break          # condition complete from an earlier job: only re-record input md5
                img = cv2.imread(str(D / rel))
                assert img is not None and img.shape == (590, 1640, 3), rel
                x = corrupt(img, c, s, idx)
                if idx < 3: md5[f'{cn}/{idx}'] = hashlib.md5(x.tobytes()).hexdigest()
                if idx == 0: ex.append((cn, x))
                if done: continue
                lanes = post(fwd(pre(x)))
                dst = pdir / (rel[:-3] + 'lines.txt'); dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(''.join(' '.join(f'{float(a):.2f} {float(b):.2f}' for a, b in l) + '\n' for l in lanes if len(l) > 1))
            n = sum(1 for _ in pdir.rglob('*.lines.txt'))
            assert n == len(items), f'{cn}: {n} files for {len(items)} items'
            print(f'{name} {cn} {"already complete (md5 re-recorded)" if done else "done"} ({n} files) {time.time() - t0:.0f}s', flush=True)
    tiles = [cv2.putText(cv2.resize(x, (410, 148)), cn, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2) for cn, x in ex]
    rows = [np.hstack(tiles[i:i + 4] + [np.zeros_like(tiles[0])] * (4 - len(tiles[i:i + 4]))) for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(out / f'corruption_examples{tag}.jpg'), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    (out / f'robustness_manifest{tag}.json').write_text(json.dumps(dict(model=name, images=len(items), stride=STRIDE, conditions=[cond_name(*c) for c in conds],
        severity=SEVERITY, input_md5_first3=md5, seconds=time.time() - t0, model_info=info,
        note='UFLDv2 non-TTA; inputs identical across models (compare input_md5_first3)'), indent=2))
    print('ROBUST DONE', out)


if __name__ == '__main__':
    main()
