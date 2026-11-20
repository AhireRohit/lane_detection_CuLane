#!/usr/bin/env python3
"""Phase F: model-agnostic geometry filter + causal temporal lane tracker, with ablation.

B0 raw baseline | B1 + geometric validity filter | B2 + temporal matching/smoothing | B3 geometry + temporal.
Operates offline on each model's saved baseline per-frame lanes (results/dashcam/<model>/<batch>/frames.jsonl),
so checkpoint and score threshold are identical to the baseline. Processing is strictly causal (frame t uses
frames <= t only), so offline == online behaviour.
All tracker parameters are fixed a priori (no search). B1 thresholds are calibrated on CULane VAL ground truth
only; the false-rejection rate of true lanes is measured on that set. Video halves (frames < 630 and >= 630)
are reported separately; nothing is tuned on either. The video is unlabeled: no accuracy claims.
Usage: temporal_ablation.py <dash_batch> <out_dir> [per_model_thresholds.json]
v2: optional third argument = b1_calibrate_preds.py output (thresholds from each model's own CULane VAL predictions).
"""
import sys, json, time, random, csv
from pathlib import Path
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

B = Path(__file__).resolve().parents[1]
D = B.parent / 'datasets' / 'CULane'
CW, CH = 1640, 590
ROWS = np.arange(250, 590, 10, dtype=float)          # common image rows, CULane canvas coordinates
P = dict(gate_px=40.0, alpha=0.5, confirm_hits=2, max_hold=3, min_common_rows=5, cross_margin_px=10.0)
PCT = dict(extent_low=0.5, curv_high=99.5, slope_high=99.5)
MODELS = ['scnn', 'ufldv2', 'clrernet']
SPLIT = 630
COLORS = [(0, 255, 0), (255, 80, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0), (0, 255, 255)]


# ---------------- common lane representation ----------------
def resample(pts):
    a = np.asarray(pts, float)
    if len(a) < 2: return None
    a = a[np.argsort(a[:, 1])]
    _, idx = np.unique(a[:, 1], return_index=True); a = a[idx]
    if len(a) < 2: return None
    x = np.full(len(ROWS), np.nan)
    m = (ROWS >= a[0, 1]) & (ROWS <= a[-1, 1])
    x[m] = np.interp(ROWS[m], a[:, 1], a[:, 0])
    return x if np.isfinite(x).sum() >= 2 else None


def feats(x):
    xi = x[np.isfinite(x)]
    d1, d2 = np.diff(xi), np.diff(xi, 2)
    return dict(extent=10.0 * (xi.size - 1), curv=float(np.abs(d2).max()) if d2.size else 0.0,
                slope=float(np.median(np.abs(d1)) / 10.0) if d1.size else 0.0)


def crosses(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3: return False
    d = a[m] - b[m]; d = d[np.abs(d) > P['cross_margin_px']]
    return d.size > 0 and d.max() > 0 and d.min() < 0


# ---------------- B1: geometric validity filter ----------------
def calibrate():
    F, n_img, cross_img, n_lanes_img = [], 0, 0, []
    for line in open(D / 'list/val.txt'):
        if not line.strip(): continue
        rel = line.split()[0].lstrip('/')
        p = D / (rel[:-3] + 'lines.txt')
        if not p.is_file(): continue
        lanes = []
        for row in p.read_text().splitlines():
            v = [float(t) for t in row.split()]
            if len(v) >= 4:
                x = resample(list(zip(v[0::2], v[1::2])))
                if x is not None: lanes.append(x)
        n_img += 1; n_lanes_img.append(len(lanes))
        F += [feats(x) for x in lanes]
        if any(crosses(lanes[i], lanes[j]) for i in range(len(lanes)) for j in range(i + 1, len(lanes))): cross_img += 1
    ext = np.array([f['extent'] for f in F]); cur = np.array([f['curv'] for f in F]); slo = np.array([f['slope'] for f in F])
    th = dict(min_extent=float(np.percentile(ext, PCT['extent_low'])), max_curv=float(np.percentile(cur, PCT['curv_high'])),
              max_slope=float(np.percentile(slo, PCT['slope_high'])))
    rej = (ext < th['min_extent']) | (cur > th['max_curv']) | (slo > th['max_slope'])
    cal = dict(source='CULane list/val.txt ground truth (no test labels, no video)', percentiles=PCT, thresholds=th,
               val_images=n_img, val_gt_lanes=int(len(F)),
               gt_lane_false_rejection_rate_single_lane_rules=float(rej.mean()),
               gt_lane_rejection_by_rule=dict(extent=float((ext < th['min_extent']).mean()), curv=float((cur > th['max_curv']).mean()),
                                              slope=float((slo > th['max_slope']).mean())),
               gt_images_with_crossing_pair=int(cross_img), gt_image_crossing_rate=cross_img / max(n_img, 1),
               feature_quantiles={k: np.percentile(a, [1, 50, 99]).round(3).tolist() for k, a in (('extent', ext), ('curv', cur), ('slope', slo))},
               caveat='video frames are squashed vertically (aspect factor 0.64) before inference, which makes lanes appear '
                      'shallower than in CULane; slope-rule false rejections on video may exceed the val-GT rate')
    return th, cal


def geo_filter(lanes, th):
    keep, rej = [], []
    for x in lanes:
        f = feats(x)
        bad = f['extent'] < th['min_extent'] or f['curv'] > th['max_curv'] or f['slope'] > th['max_slope']
        (rej if bad else keep).append(x)
    changed = True
    while changed:                       # remove the shorter lane of any crossing pair, repeatedly
        changed = False
        for i in range(len(keep)):
            for j in range(i + 1, len(keep)):
                if crosses(keep[i], keep[j]):
                    k = i if np.isfinite(keep[i]).sum() < np.isfinite(keep[j]).sum() else j
                    rej.append(keep.pop(k)); changed = True; break
            if changed: break
    return keep, rej


# ---------------- B2: causal tracker with EMA smoothing and explicit holds ----------------
def cost(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.median(np.abs(a[m] - b[m]))) if m.sum() >= P['min_common_rows'] else np.inf


def match(A, Bs):
    if not A or not Bs: return []
    C = np.array([[cost(a, b) for b in Bs] for a in A])
    r, c = linear_sum_assignment(np.where(np.isfinite(C), C, 1e9))
    return [(i, j) for i, j in zip(r, c) if C[i, j] < P['gate_px']]


class Tracker:
    def __init__(self): self.tracks, self.next_id = [], 0

    def step(self, dets):
        T = self.tracks; pairs = match([t['x'] for t in T], dets)
        mt, md = {i for i, _ in pairs}, {j for _, j in pairs}
        for i, j in pairs:
            t, d = T[i], dets[j]
            both = np.isfinite(t['x']) & np.isfinite(d)
            x = d.copy(); x[both] = P['alpha'] * d[both] + (1 - P['alpha']) * t['x'][both]
            t.update(x=x, hits=t['hits'] + 1, misses=0, obs=d)
            if t['hits'] >= P['confirm_hits']: t['confirmed'] = True
        for i, t in enumerate(T):
            if i not in mt: t['misses'] += 1; t['obs'] = None
        self.tracks = [t for t in T if t['misses'] == 0 or (t['confirmed'] and t['misses'] <= P['max_hold'])]
        for j, d in enumerate(dets):
            if j not in md:
                self.tracks.append(dict(id=self.next_id, x=d.copy(), hits=1, misses=0, confirmed=P['confirm_hits'] <= 1, obs=d))
                self.next_id += 1
        return [dict(x=t['x'], held=t['misses'] > 0, id=t['id'], obs=t['obs']) for t in self.tracks if t['confirmed']]


def run_variant(frames, v, th):
    tr = Tracker() if v in ('B2', 'B3') else None
    res, lat = [], []
    for lanes in frames:
        t0 = time.perf_counter()
        dets, rej = (geo_filter(lanes, th) if v in ('B1', 'B3') else (lanes, []))
        disp = tr.step(dets) if tr else [dict(x=x, held=False, id=None, obs=x) for x in dets]
        lat.append(1000 * (time.perf_counter() - t0))
        res.append(dict(disp=disp, rej=rej))
    return res, lat


# ---------------- label-free stability metrics (identical for every variant) ----------------
def metrics(res, lat, lo, hi, sx):
    R = res[lo:hi]; n = len(R)
    cnt = np.array([len(r['disp']) for r in R]); obs = np.array([sum(not d['held'] for d in r['disp']) for r in R])
    held = int(sum(sum(d['held'] for d in r['disp']) for r in R))
    births, jit, dev = 0, {500: [], 400: []}, []
    ri = {y: int(np.where(ROWS == y)[0][0]) for y in jit}
    for k in range(1, n):
        A = [d['x'] for d in R[k - 1]['disp']]; Bx = [d['x'] for d in R[k]['disp']]
        pr = match(A, Bx); births += len(Bx) - len(pr)
        for i, j in pr:
            for y, r in ri.items():
                if np.isfinite(A[i][r]) and np.isfinite(Bx[j][r]): jit[y].append(abs(A[i][r] - Bx[j][r]))
    for r in R:
        for d in r['disp']:
            if not d['held'] and d['obs'] is not None:
                m = np.isfinite(d['x']) & np.isfinite(d['obs'])
                if m.any(): dev.append(float(np.median(np.abs(d['x'][m] - d['obs'][m]))))
    q = lambda a, p: float(np.percentile(a, p)) if len(a) else None
    L = lat[lo:hi]
    return dict(frames=n, availability=float((cnt > 0).mean()), observed_availability=float((obs > 0).mean()),
                held_lane_fraction=held / max(int(cnt.sum()), 1), mean_displayed_lanes=float(cnt.mean()),
                mean_observed_lanes=float(obs.mean()),
                lane_count_change_events=int((np.diff(cnt) != 0).sum()),
                lane_chain_starts_per_100_frames=100 * births / max(n - 1, 1),
                jitter_y500_median_px=q(jit[500], 50), jitter_y500_p95_px=q(jit[500], 95),
                jitter_y400_median_px=q(jit[400], 50), jitter_y400_p95_px=q(jit[400], 95),
                smoothing_offset_from_observation_median_px=q(dev, 50), smoothing_offset_p95_px=q(dev, 95),
                b1_rejected_lanes=int(sum(len(r['rej']) for r in R)),
                postproc_ms_mean=float(np.mean(L)), postproc_ms_p95=float(np.percentile(L, 95)),
                px_unit=f'CULane canvas px (x 1640); multiply by {sx:.4f} for source-video px')


# ---------------- rendering ----------------
def affine(rep):
    """canvas -> source: x*sx, y*sy + y0 (v1 squash adapter: y0=0; v2 crop adapter: y0=band top)."""
    a = rep.get('adapter_affine')
    if a: return a['sx'], a['sy'], a['y0']
    return rep['video_width'] / CW, rep['video_height'] / CH, 0


def to_src(x, sx, sy, y0=0):
    m = np.isfinite(x)
    return np.round(np.stack([x[m] * sx, ROWS[m] * sy + y0], 1)).astype(np.int32)


def draw(img, r, sx, sy, label, show_rej, y0=0):
    for k, d in enumerate(r['disp']):
        pts = to_src(d['x'], sx, sy, y0)
        if len(pts) < 2: continue
        col = COLORS[(d['id'] if d['id'] is not None else k) % len(COLORS)]
        if d['held']:
            for s in range(0, len(pts) - 1, 4):
                cv2.line(img, tuple(pts[s]), tuple(pts[min(s + 2, len(pts) - 1)]), col, 3, cv2.LINE_AA)
        else:
            cv2.polylines(img, [pts], False, col, 3, cv2.LINE_AA)
    if show_rej:
        for x in r['rej']:
            pts = to_src(x, sx, sy, y0)
            if len(pts) >= 2: cv2.polylines(img, [pts], False, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)


def main():
    batch, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    th, cal = calibrate()
    (out / 'b1_calibration_culane_val.json').write_text(json.dumps(cal, indent=2))
    print('B1 calibration:', json.dumps(cal['thresholds']), 'GT false rejection', cal['gt_lane_false_rejection_rate_single_lane_rules'], flush=True)
    th_by_model = {m: th for m in MODELS}
    if len(sys.argv) > 3:   # v2: per-model thresholds calibrated on the model's own VAL predictions (b1_calibrate_preds.py)
        J = json.loads(Path(sys.argv[3]).read_text())
        th_by_model = {m: J[m]['thresholds'] for m in MODELS}
        (out / 'b1_thresholds_used.json').write_text(json.dumps(dict(source=sys.argv[3], thresholds=th_by_model), indent=2))
        print('B1 thresholds per model (val predictions):', json.dumps(th_by_model), flush=True)
    rows, keep = [], {}
    for m in MODELS:
        th = th_by_model[m]
        run = B / 'results/dashcam' / m / batch
        rep = json.loads((run / 'report.json').read_text())
        sx, sy, y0 = affine(rep)
        frames = []
        for line in open(run / 'frames.jsonl'):
            fr = json.loads(line)
            lanes = [resample([(x / sx, (y - y0) / sy) for x, y in l]) for l in fr['lanes']]
            frames.append([x for x in lanes if x is not None])
        allm = {}
        for v in ['B0', 'B1', 'B2', 'B3']:
            res, lat = run_variant(frames, v, th)
            n = len(frames); half = n // 2   # = 630 for example.mp4 (1260 frames); generic for new videos
            allm[v] = {part: metrics(res, lat, lo, hi, sx) for part, (lo, hi) in
                       (('first_half', (0, half)), ('second_half', (half, n)), ('all', (0, n)))}
            for part in ('first_half', 'second_half'):
                rows.append(dict(model=m, variant=v, part=part, **{k: v2 for k, v2 in allm[v][part].items() if k != 'px_unit'}))
            if m == 'clrernet' and v in ('B0', 'B3'): keep[v] = res
            if v == 'B3':
                with open(out / f'{m}_B3_frames.jsonl', 'w') as f:
                    for i, r in enumerate(res):
                        f.write(json.dumps(dict(frame=i, observed=[d['id'] for d in r['disp'] if not d['held']],
                                                held=[d['id'] for d in r['disp'] if d['held']], n_rejected=len(r['rej']))) + '\n')
        (out / f'{m}_metrics.json').write_text(json.dumps(dict(model=m, params=P, b1_thresholds=th, metrics=allm), indent=2))
        print(m, 'done', flush=True)
    with open(out / 'ablation.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # CLRerNet visuals: B0 | B3 side-by-side video, review sheet (second half, random), B1-rejection sheet
    run = B / 'results/dashcam/clrernet' / batch
    rep = json.loads((run / 'report.json').read_text()); sx, sy, y0 = affine(rep)
    cap = cv2.VideoCapture(rep['video_path']); fps = rep['video_fps']
    n = len(keep['B0']); rnd = random.Random(0)
    review = set(rnd.sample(range(n // 2, n), min(12, n - n // 2)))
    rej_frames = [i for i in range(n) if keep['B3'][i]['rej']]
    rej_pick = set(rnd.sample(rej_frames, min(12, len(rej_frames))))
    vw, sheet_r, sheet_j = None, [], []
    for i in range(n):
        ok, fr = cap.read()
        if not ok: break
        a, b = fr.copy(), fr.copy()
        draw(a, keep['B0'][i], sx, sy, f'B0 raw  frame {i}', False, y0)
        draw(b, keep['B3'][i], sx, sy, f'B3 geo+temporal  frame {i}  (dashed=held, thin red=rejected)', True, y0)
        row = np.hstack([cv2.resize(a, (640, 360), interpolation=cv2.INTER_AREA), cv2.resize(b, (640, 360), interpolation=cv2.INTER_AREA)])
        if vw is None: vw = cv2.VideoWriter(str(out / 'clrernet_B0_vs_B3.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (row.shape[1], row.shape[0]))
        vw.write(row)
        if i in review: sheet_r.append((i, row))
        if i in rej_pick: sheet_j.append((i, row))
    vw.release()
    for name, sh in (('clrernet_review_sheet_second_half.jpg', sheet_r), ('clrernet_b1_rejection_sheet.jpg', sheet_j)):
        if sh:
            sh.sort(); grid = [np.vstack([r for _, r in sh[k:k + 6]]) for k in range(0, len(sh), 6)]
            h = max(g.shape[0] for g in grid)
            grid = [np.vstack([g, np.zeros((h - g.shape[0], g.shape[1], 3), np.uint8)]) for g in grid]
            cv2.imwrite(str(out / name), np.hstack(grid), [cv2.IMWRITE_JPEG_QUALITY, 88])
    (out / 'review_frames.json').write_text(json.dumps(dict(review_second_half=sorted(review), b1_rejection_frames_sampled=sorted(rej_pick),
                                                             clrernet_frames_with_b1_rejection=len(rej_frames)), indent=2))
    print('ABLATION DONE', out)


if __name__ == '__main__':
    main()
