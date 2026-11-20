#!/usr/bin/env python3
"""Cross-dataset test: CULane-trained models on TuSimple test clips (never trained on). The public mirror holds a 10%
sample: the 233 test clips present with all 20 frames (test_label_available.json, written by download_tusimple.sh).

Protocol, fixed before any TuSimple result was seen:
  * Adapter: the whole 1280x720 frame is resized to the CULane canvas 1640x590 ("squash", as in the demo), then the
    model's own verified inference path runs (benchmark_speed.py builders, the code behind every reported number).
  * Rows: only label rows y >= 330 are scored, for ground truth and predictions alike. y = 330 is CLRerNet's crop
    line (cut_height 270 on the CULane canvas) mapped back to TuSimple, i.e. the part of the frame every model sees.
    Ground-truth lanes left with fewer than 2 points are dropped.
  * Metrics: (1) the official TuSimple metric (tusimple-benchmark LaneEval: 20 px angle-adjusted threshold, 85 % of
    points; accuracy, FP rate, FN rate) re-implemented line by line, plus the TuSimple F1 used by CLRNet/CLRerNet,
    F1 = 2(1-FP)(1-FN)/((1-FP)+(1-FN)); (2) the official CULane C++ evaluator on the same lanes at 1280x720
    (width 30, IoU 0.5), run by the wrapper script.
  * Tracker (--clips): frames 1..20 of every test clip (20 fps) are run in order and the B2 tracker
    (temporal_ablation.py, parameters unchanged) is scored on the labeled frame 20, with the same B0r / B2 / B2obs
    variants as scripts/tracker_culane.py.
Usage: tusimple_eval.py <model: scnn|ufldv2|clrernet> <out_dir> [--clips]   (CLR_CKPT selects a CLRerNet checkpoint)
"""
import sys, os, json, time
from pathlib import Path
import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
T = HERE.parent.parent / 'datasets' / 'TuSimple'
SW, SH, CW, CH = 1280, 720, 1640, 590
SX, SY = SW / CW, SH / CH
Y_MIN = 330
PIX, PT = 20, 0.85


# ---------------- official TuSimple LaneEval (tusimple-benchmark evaluate/lane.py), same logic ----------------
def get_angle(xs, ys):
    xs, ys = xs[xs >= 0], ys[xs >= 0]
    return float(np.arctan(np.polyfit(ys, xs, 1)[0])) if len(xs) > 1 else 0.0   # == LinearRegression slope


def line_accuracy(pred, gt, thresh):
    pred = np.array([p if p >= 0 else -100 for p in pred]); gt = np.array([g if g >= 0 else -100 for g in gt])
    return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)


def bench(pred, gt, ys):
    if len(gt) + 2 < len(pred): return 0., 0., 1.
    threshs = [PIX / np.cos(get_angle(np.array(g, float), np.array(ys, float))) for g in gt]
    accs_all, matched = [], 0.
    fn = 0.
    for g, th in zip(gt, threshs):
        accs = [line_accuracy(np.array(p), np.array(g), th) for p in pred]
        m = np.max(accs) if accs else 0.
        if m < PT: fn += 1
        else: matched += 1
        accs_all.append(m)
    fp = len(pred) - matched
    if len(gt) > 4 and fn > 0: fn -= 1
    s = sum(accs_all)
    if len(gt) > 4: s -= min(accs_all)
    return s / max(min(4.0, len(gt)), 1.), (fp / len(pred) if pred else 0.), fn / max(min(len(gt), 4.), 1.)


def _f1(fp, fn):
    p, r = 1 - np.asarray(fp, float), 1 - np.asarray(fn, float)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(p + r > 0, 2 * p * r / np.where(p + r > 0, p + r, 1), 0.0)   # 0 when FP = FN = 100 %


def score(preds, labels, boot=2000):
    per = np.array([bench(preds[lab['raw_file']], lab['lanes'], lab['h_samples']) for lab in labels])   # acc, fp, fn
    acc, fp, fn = per.mean(0)
    idx = np.random.default_rng(0).integers(0, len(per), (boot, len(per)))   # clip-level bootstrap, fixed seed
    m = per[idx].mean(1)
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return dict(frames=len(per), accuracy=float(acc), fp_rate=float(fp), fn_rate=float(fn), f1=float(_f1(fp, fn)),
                ci95=dict(accuracy=ci(m[:, 0]), f1=ci(_f1(m[:, 1], m[:, 2]))), per_frame=per.round(4).tolist())


# ---------------- protocol helpers ----------------
def truncate_gt(lab):
    ys = lab['h_samples']; lanes = []
    for xs in lab['lanes']:
        xs = [x if (y >= Y_MIN and x >= 0) else -2 for x, y in zip(xs, ys)]
        if sum(x >= 0 for x in xs) >= 2: lanes.append(xs)
    return dict(lab, lanes=lanes)


def to_samples(pts, ys):
    """Lane points in source pixels -> x at each label row (-2 outside the lane's extent or above Y_MIN)."""
    a = np.asarray(pts, float); a = a[np.isfinite(a).all(1)]
    a = a[(a[:, 0] >= 0) & (a[:, 0] < SW)]
    if len(a) < 2: return None
    a = a[np.argsort(a[:, 1])]; _, i = np.unique(a[:, 1], return_index=True); a = a[i]
    if len(a) < 2: return None
    out = [float(np.interp(y, a[:, 1], a[:, 0])) if (y >= Y_MIN and a[0, 1] <= y <= a[-1, 1]) else -2 for y in ys]
    return out if sum(x >= 0 for x in out) >= 2 else None


def canvas_to_src(lane):
    return [(x * SX, y * SY) for x, y in lane]


def write_lines(path, lanes_xs, ys):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [' '.join(f'{x:.2f} {y}' for x, y in sorted(zip(xs, ys), key=lambda t: -t[1]) if x >= 0) for xs in lanes_xs]
    rows = [r for r in rows if r.count(' ') >= 3]
    # no lanes -> empty file, as the official CULane writer does (v1 wrote a lone newline, which the C++ evaluator
    # counts as one lane on that side; the TuSimple-metric numbers never used these files)
    path.write_text('\n'.join(rows) + '\n' if rows else '')


def rescore(out):
    """Rewrite the lane files from the saved predictions (pred_<variant>.json) and check the TuSimple metric is
    unchanged; the wrapper then re-runs the CULane evaluator. No inference."""
    labels = {}
    for l in open(T / 'test_label_available.json'):
        if l.strip():
            lab = truncate_gt(json.loads(l)); labels[lab['raw_file']] = lab
    res = json.loads((out / 'tusimple_metrics.json').read_text())
    for lab in labels.values():
        write_lines(out / 'gt_lines' / (lab['raw_file'][:-3] + 'lines.txt'), lab['lanes'], lab['h_samples'])
    for v in res['results']:
        preds = {}
        for l in open(out / f'pred_{v}.json'):
            d = json.loads(l); preds[d['raw_file']] = d['lanes']
        new = score(preds, list(labels.values()))
        assert abs(new['f1'] - res['results'][v]['f1']) < 1e-9, (v, new['f1'], res['results'][v]['f1'])
        for r, lanes in preds.items():
            write_lines(out / v / (r[:-3] + 'lines.txt'), lanes, labels[r]['h_samples'])
    print('RESCORED', out)


def main():
    model, out, clips = sys.argv[1], Path(sys.argv[2]), '--clips' in sys.argv
    out.mkdir(parents=True, exist_ok=True)
    if '--rescore' in sys.argv: return rescore(out)
    import torch
    from benchmark_speed import build_scnn, build_ufld, build_clr
    if clips: import temporal_ablation as ta   # scipy; only the tracker needs it
    torch.backends.cudnn.benchmark = False
    if model == 'scnn': pre, fwd, post, _, _ = build_scnn()
    elif model == 'ufldv2': pre, fwd, post, _, _ = build_ufld(out)
    else: pre, fwd, post, _, _ = build_clr(out)

    def infer(path):
        img = cv2.imread(str(T / path)); assert img is not None and img.shape == (SH, SW, 3), path
        with torch.no_grad():
            lanes = post(fwd(pre(cv2.resize(img, (CW, CH), interpolation=cv2.INTER_LINEAR))))
        return [l for l in lanes if len(l) > 1]

    labels = [truncate_gt(json.loads(l)) for l in open(T / 'test_label_available.json') if l.strip()]
    variants = ['raw'] + (['B0r', 'B2', 'B2obs'] if clips else [])
    preds = {v: {} for v in variants}; t0 = time.time()
    for k, lab in enumerate(labels):
        ys, raw = lab['h_samples'], lab['raw_file']
        if clips:
            tr = ta.Tracker(); d = raw.rsplit('/', 1)[0]
            for f in range(1, 21):
                lanes = infer(f'{d}/{f}.jpg')
                dets = [x for x in (ta.resample(l) for l in lanes) if x is not None]
                disp = tr.step(dets)
            to_rows = lambda xs: [(x, y) for x, y in zip(xs, ta.ROWS) if np.isfinite(x)]
            for v, ls in (('B0r', dets), ('B2', [q['x'] for q in disp]), ('B2obs', [q['x'] for q in disp if not q['held']])):
                preds[v][raw] = [s for s in (to_samples(canvas_to_src(to_rows(x)), ys) for x in ls) if s is not None]
        else:
            lanes = infer(raw)
        preds['raw'][raw] = [s for s in (to_samples(canvas_to_src(l), ys) for l in lanes) if s is not None]
        if k % 200 == 0: print(model, k, len(labels), round(time.time() - t0), 's', flush=True)

    res = dict(model=model, checkpoint=os.environ.get('CLR_CKPT', 'released') if model == 'clrernet' else 'released',
               protocol=dict(adapter='squash 1280x720 -> 1640x590', rows_scored=f'y >= {Y_MIN}', metric='TuSimple LaneEval',
                             f1='2(1-FP)(1-FN)/((1-FP)+(1-FN)) (CLRNet/CLRerNet convention)'),
               seconds=round(time.time() - t0), results={})
    gt_dir = out / 'gt_lines'
    for lab in labels:
        write_lines(gt_dir / (lab['raw_file'][:-3] + 'lines.txt'), lab['lanes'], lab['h_samples'])
    (out / 'list.txt').write_text(''.join(lab['raw_file'] + '\n' for lab in labels))
    for v in variants:
        res['results'][v] = score(preds[v], labels)
        for lab in labels:
            write_lines(out / v / (lab['raw_file'][:-3] + 'lines.txt'), preds[v][lab['raw_file']], lab['h_samples'])
        with open(out / f'pred_{v}.json', 'w') as fh:
            for lab in labels:
                fh.write(json.dumps(dict(raw_file=lab['raw_file'], lanes=preds[v][lab['raw_file']], h_samples=lab['h_samples'])) + '\n')
    (out / 'tusimple_metrics.json').write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))
    print('TUSIMPLE EVAL DONE', model)


if __name__ == '__main__':
    main()
