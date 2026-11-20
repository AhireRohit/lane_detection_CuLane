#!/usr/bin/env python3
"""Accuracy of the B2 temporal tracker on LABELED video: the CULane test set (419 clips, frames 1 s apart).

The tracker (scripts/temporal_ablation.py, parameters fixed before any dashcam or CULane run, unchanged here) is run
causally over each test clip in frame order on a model's saved full-test predictions, restarting at every clip.
Variants (all declared before scoring, none selected afterwards):
  B0r     raw predictions passed through the tracker's lane representation (rows 250..580, 10 px) with no
          tracking: the fair control, since B2 outputs that representation
  B2      tracker output, confirmed tracks including held (missed-detection) lanes, as the demo draws them
  B2obs   tracker output without held lanes (smoothing + confirmation only)
Each variant is written as CULane .lines.txt files and scored by the official evaluator (score_culane_official.sh).
Caveat: CULane annotates 1 frame per second; the tracker was designed for 10-30 fps video, where its EMA and
3-frame hold span 0.1-0.3 s instead of 1-3 s here.
Usage: tracker_culane.py <model_name> <pred_dir> <out_dir>
"""
import sys, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import temporal_ablation as ta   # Tracker, resample, ROWS, P (unchanged)

VARIANTS = ('B0r', 'B2', 'B2obs')


def read_lanes(p):
    out = []
    if not p.is_file(): return out
    for row in p.read_text().splitlines():
        v = [float(t) for t in row.split()]
        if len(v) >= 4:
            x = ta.resample(list(zip(v[0::2], v[1::2])))
            if x is not None: out.append(x)
    return out


def write_lanes(p, xs):
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for x in xs:
        m = np.isfinite(x) & (x >= 0) & (x < ta.CW)
        if m.sum() >= 2:
            rows.append(' '.join(f'{xx:.2f} {yy:.0f}' for xx, yy in zip(x[m][::-1], ta.ROWS[m][::-1])))
    p.write_text('\n'.join(rows) + ('\n' if rows else ''))


def main():
    name, pred, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    clips = defaultdict(list)
    for line in open(ta.D / 'list/test.txt'):
        if line.strip():
            rel = line.split()[0].lstrip('/'); clips[str(Path(rel).parent)].append(rel)
    t0, n, stats = time.time(), 0, defaultdict(int)
    for clip, frames in clips.items():
        frames.sort(key=lambda r: int(Path(r).stem))
        tr = ta.Tracker()
        for rel in frames:
            s = rel[:-3] + 'lines.txt'
            dets = read_lanes(pred / s)
            disp = tr.step(dets)
            write_lanes(out / 'B0r' / s, dets)
            write_lanes(out / 'B2' / s, [d['x'] for d in disp])
            write_lanes(out / 'B2obs' / s, [d['x'] for d in disp if not d['held']])
            stats['frames'] += 1; stats['raw_lanes'] += len(dets); stats['b2_lanes'] += len(disp)
            stats['b2_held'] += sum(d['held'] for d in disp); n += 1
    meta = dict(model=name, pred_dir=str(pred), clips=len(clips), tracker_params=ta.P, rows=[ta.ROWS[0], ta.ROWS[-1], 10],
                variants=VARIANTS, **stats, seconds=round(time.time() - t0, 1),
                caveat='CULane test frames are 1 s apart; tracker designed for 10-30 fps')
    (out / 'tracker_culane_meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
