#!/usr/bin/env python3
"""Calibrate the B1 geometry filter per model on that model's own CULane VAL predictions.

1. Label each predicted lane TP/FP against val GT: lanes drawn as 30 px wide polylines (CULane convention),
   computed at half resolution (820x295, width 15) for speed, Hungarian matching, IoU >= 0.5 -> TP.
   (Polyline rasterization approximates the official evaluator's spline drawing; this is a calibration aid,
   not a reported benchmark score.)
2. Decision rule fixed BEFORE looking at results: per model, thresholds from the TP-lane feature distribution
   (extent p0.5, curvature p99.5, slope p99.5), i.e. ~1% true-lane rejection per model by construction.
3. Report per-rule rejection rates on TP lanes (cost) and FP lanes (benefit), and the crossing rule's TP/FP split.
Usage: b1_calibrate_preds.py <valpred_root> (expects <root>/<model>/pred/...)
"""
import sys, json
from pathlib import Path
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, str(Path(__file__).parent))
from temporal_ablation import resample, feats, crosses, MODELS, D

S = 0.5; WIDTH = 30; IOU = 0.5; HW = (int(590 * S), int(1640 * S))
PCT = dict(extent_low=0.5, curv_high=99.5, slope_high=99.5)


def read(p):
    out = []
    if p.is_file():
        for row in p.read_text().splitlines():
            v = [float(t) for t in row.split()]
            if len(v) >= 4: out.append(list(zip(v[0::2], v[1::2])))
    return out


def mask(l):
    m = np.zeros(HW, np.uint8)
    cv2.polylines(m, [np.round(np.array(l) * S).astype(np.int32)], False, 1, max(1, int(WIDTH * S)))
    return m.astype(bool)


def main():
    root = Path(sys.argv[1])
    rels = [l.split()[0].lstrip('/') for l in open(D / 'list/val.txt') if l.strip()]
    report = {}
    for m in MODELS:
        F, lab, tp, fp, fn, cross_tp, cross_fp = [], [], 0, 0, 0, 0, 0
        for k, rel in enumerate(rels):
            stem = rel[:-3] + 'lines.txt'
            P = [l for l in read(root / m / 'pred' / stem) if len(l) > 1]
            G = [l for l in read(D / stem) if len(l) > 1]
            ok = np.zeros(len(P), bool)
            if P and G:
                pm = [mask(l) for l in P]; gm = [mask(l) for l in G]
                iou = np.array([[(a & b).sum() / max((a | b).sum(), 1) for b in gm] for a in pm])
                r, c = linear_sum_assignment(-iou)
                for i, j in zip(r, c):
                    if iou[i, j] >= IOU: ok[i] = True
            tp += int(ok.sum()); fp += int(len(P) - ok.sum()); fn += int(len(G) - ok.sum())
            X = [resample(l) for l in P]
            for i, x in enumerate(X):
                if x is None: continue
                F.append(feats(x)); lab.append(bool(ok[i]))
            for i in range(len(X)):
                for j in range(i + 1, len(X)):
                    if X[i] is not None and X[j] is not None and crosses(X[i], X[j]):
                        s = i if np.isfinite(X[i]).sum() < np.isfinite(X[j]).sum() else j   # same choice as geo_filter
                        cross_tp += int(ok[s]); cross_fp += int(not ok[s])
            if k % 2000 == 0: print(m, k, flush=True)
        lab = np.array(lab)
        ext = np.array([f['extent'] for f in F]); cur = np.array([f['curv'] for f in F]); slo = np.array([f['slope'] for f in F])
        th = dict(min_extent=float(np.percentile(ext[lab], PCT['extent_low'])), max_curv=float(np.percentile(cur[lab], PCT['curv_high'])),
                  max_slope=float(np.percentile(slo[lab], PCT['slope_high'])))
        rules = dict(extent=ext < th['min_extent'], curv=cur > th['max_curv'], slope=slo > th['max_slope'])
        anyr = rules['extent'] | rules['curv'] | rules['slope']
        rate = lambda r, s: float(r[s].mean()) if s.any() else None
        p_ = tp / max(tp + fp, 1); r_ = tp / max(tp + fn, 1)
        report[m] = dict(thresholds=th, val_images=len(rels), pred_lanes=int(lab.size), tp_lanes=int(lab.sum()), fp_lanes=int((~lab).sum()),
                         approx_val_f1=2 * p_ * r_ / max(p_ + r_, 1e-9),
                         tp_rejection_by_rule={k: rate(v, lab) for k, v in rules.items()}, fp_rejection_by_rule={k: rate(v, ~lab) for k, v in rules.items()},
                         tp_rejection_any=rate(anyr, lab), fp_rejection_any=rate(anyr, ~lab),
                         crossing_rule_removed=dict(tp=cross_tp, fp=cross_fp),
                         gt_calibrated_thresholds_for_reference=json.loads((Path(__file__).resolve().parents[1] / 'results/ablation/temporal_73883/b1_calibration_culane_val.json').read_text())['thresholds'])
        print(m, json.dumps({k: report[m][k] for k in ('thresholds', 'tp_rejection_any', 'fp_rejection_any', 'crossing_rule_removed', 'approx_val_f1')}), flush=True)
    report['method'] = dict(matching=f'polyline width {WIDTH}px at scale {S}, Hungarian, IoU>={IOU}', percentiles=PCT,
                            note='thresholds from TP-lane distributions on CULane VAL predictions; no test labels; no video')
    (root / 'b1_calibration_pred.json').write_text(json.dumps(report, indent=2))
    print('CALIB DONE', root / 'b1_calibration_pred.json')


if __name__ == '__main__':
    main()
