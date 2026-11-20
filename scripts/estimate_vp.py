#!/usr/bin/env python3
"""Camera-geometry calibration for the aspect-preserving video adapter (label-free on the video side).

Vanishing row = median intersection height of pairs of lanes, each lane fitted as x = a*y + b on its lower part.
- CULane reference: val ground-truth lanes (1640x590), no test labels.
- Video: the saved baseline predicted lanes of all three models (source pixels).
Crop band: full video width, height W*590/1640, placed so the video vanishing row sits at the same relative height
as CULane's. No evaluation result is used to choose it.
Usage: estimate_vp.py <dash_batch> <out_json>
"""
import sys, json
from pathlib import Path
import numpy as np

B = Path(__file__).resolve().parents[1]; D = B.parent / 'datasets/CULane'


def fit(lane, H):
    a = np.asarray(lane, float)
    a = a[a[:, 1] >= 0.55 * H] if (a[:, 1] >= 0.55 * H).sum() >= 3 else a
    if len(a) < 3 or np.ptp(a[:, 1]) < 0.05 * H: return None
    A, Bc = np.polyfit(a[:, 1], a[:, 0], 1)
    return A, Bc


def vrows(lanes_per_image, H):
    ys = []
    for lanes in lanes_per_image:
        F = [f for f in (fit(l, H) for l in lanes) if f is not None]
        for i in range(len(F)):
            for j in range(i + 1, len(F)):
                (a1, b1), (a2, b2) = F[i], F[j]
                if abs(a1 - a2) > 0.2:                      # skip near-parallel pairs (unstable intersection)
                    y = (b2 - b1) / (a1 - a2)
                    if 0 <= y <= H: ys.append(y)
    return np.array(ys)


def read(p):
    out = []
    for row in p.read_text().splitlines():
        v = [float(t) for t in row.split()]
        if len(v) >= 6: out.append(list(zip(v[0::2], v[1::2])))
    return out


def main():
    batch, outp = sys.argv[1], Path(sys.argv[2])
    gt = []
    for l in open(D / 'list/val.txt'):
        if l.strip():
            p = D / (l.split()[0].lstrip('/')[:-3] + 'lines.txt')
            if p.is_file(): gt.append(read(p))
    cy = vrows(gt, 590)
    rep = json.loads((B / 'results/dashcam/clrernet' / batch / 'report.json').read_text())
    W, H = rep['video_width'], rep['video_height']
    vid = []
    for m in ['scnn', 'ufldv2', 'clrernet']:
        for line in open(B / 'results/dashcam' / m / batch / 'frames.jsonl'):
            vid.append([[tuple(p) for p in l] for l in json.loads(line)['lanes'] if len(l) >= 3])
    vy = vrows(vid, H)
    r = float(np.median(cy)) / 590
    band_h = int(round(W * 590 / 1640))
    top = int(round(float(np.median(vy)) - r * band_h))
    top_c = min(max(top, 0), H - band_h)
    res = dict(culane_val_vanishing_row_px=float(np.median(cy)), culane_relative=r, culane_pairs=int(cy.size),
               culane_iqr=np.percentile(cy, [25, 75]).round(1).tolist(),
               video_vanishing_row_px=float(np.median(vy)), video_pairs=int(vy.size), video_iqr=np.percentile(vy, [25, 75]).round(1).tolist(),
               video_size=[W, H], band_height=band_h, band_top_unclamped=top, band_top=top_c, band=[top_c, top_c + band_h],
               scale_x=1640 / W, scale_y=590 / band_h,
               rows_lost_below_band=H - (top_c + band_h), rows_lost_above_band=top_c,
               method='median pairwise intersection of lanes fitted x=a*y+b on lower 45%; CULane from val GT, video from baseline predictions')
    outp.write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
