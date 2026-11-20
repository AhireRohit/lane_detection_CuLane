#!/usr/bin/env python3
"""CULane test failure gallery from the saved full-test predictions (no new inference).

Per image and model: approximate TP/FP/FN with the CULane rule (30 px lanes, IoU >= 0.5, Hungarian), computed at half
resolution for speed (approximation of the official spline drawing; totals are compared to the official evaluator).
Per scenario: the 4 images with the most errors summed over the three models (deterministic, ties by list order),
drawn as rows of [SCNN | UFLDv2 | CLRerNet]: GT thin white, matched prediction green, unmatched prediction red.
Usage: failure_gallery.py <out_dir>
"""
import sys, json, glob
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

B = Path(__file__).resolve().parents[1]; D = B.parent / 'datasets/CULane'
PRED = dict(scnn=B / 'results/scnn/full/coord_output', ufldv2=B / 'results/ufldv2/full/culane_eval_tmp',
            clrernet=Path(sorted(glob.glob(str(B / 'results/clrernet/full/run_*73864/pred')))[0]))
SCEN = ['test0_normal', 'test1_crowd', 'test2_hlight', 'test3_shadow', 'test4_noline', 'test5_arrow', 'test6_curve', 'test7_cross', 'test8_night']
S = 0.5; HW = (295, 820)


def read(p):
    out = []
    if p.is_file():
        for row in p.read_text().splitlines():
            v = [float(t) for t in row.split()]
            if len(v) >= 4: out.append(list(zip(v[0::2], v[1::2])))
    return out


def mask(l):
    m = np.zeros(HW, np.uint8)
    cv2.polylines(m, [np.round(np.array(l) * S).astype(np.int32)], False, 1, 15)
    return m.astype(bool)


def match(P, G):
    ok = np.zeros(len(P), bool)
    if P and G:
        pm = [mask(l) for l in P]; gm = [mask(l) for l in G]
        iou = np.array([[(a & b).sum() / max((a | b).sum(), 1) for b in gm] for a in pm])
        r, c = linear_sum_assignment(-iou)
        for i, j in zip(r, c):
            if iou[i, j] >= 0.5: ok[i] = True
    return ok


def score(rel):
    stem = rel[:-3] + 'lines.txt'; G = [l for l in read(D / stem) if len(l) > 1]; out = {}
    for m, pd in PRED.items():
        P = [l for l in read(pd / stem) if len(l) > 1]; ok = match(P, G)
        out[m] = (int(ok.sum()), int(len(P) - ok.sum()), int(len(G) - ok.sum()))
    return rel, out


def draw(rel, m):
    img = cv2.imread(str(D / rel)); stem = rel[:-3] + 'lines.txt'
    G = [l for l in read(D / stem) if len(l) > 1]; P = [l for l in read(PRED[m] / stem) if len(l) > 1]; ok = match(P, G)
    for l in G: cv2.polylines(img, [np.round(np.array(l)).astype(np.int32)], False, (255, 255, 255), 3, cv2.LINE_AA)
    for l, k in zip(P, ok): cv2.polylines(img, [np.round(np.array(l)).astype(np.int32)], False, (0, 220, 0) if k else (0, 0, 255), 7, cv2.LINE_AA)
    tp, fp, fn = int(ok.sum()), int(len(P) - ok.sum()), int(len(G) - ok.sum())
    img = cv2.resize(img, (546, 196), interpolation=cv2.INTER_AREA)
    cv2.putText(img, f'{m} tp{tp} fp{fp} fn{fn}', (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    lists = {s: [l.split()[0].lstrip('/') for l in open(D / 'list/test_split' / f'{s}.txt') if l.strip()] for s in SCEN}
    allrel = sorted({r for v in lists.values() for r in v})
    with Pool(4) as p: res = dict(p.map(score, allrel, chunksize=200))
    summary = {}
    for s, rels in lists.items():
        tot = {m: [sum(res[r][m][k] for r in rels) for k in range(3)] for m in PRED}
        summary[s] = {m: dict(tp=t[0], fp=t[1], fn=t[2], f1_approx=round(100 * 2 * t[0] / max(2 * t[0] + t[1] + t[2], 1), 2)) for m, t in tot.items()}
        err = sorted(rels, key=lambda r: -sum(res[r][m][1] + res[r][m][2] for m in PRED))[:4]
        rows = [np.hstack([draw(r, m) for m in PRED]) for r in err]
        hdr = np.full((28, rows[0].shape[1], 3), 255, np.uint8)
        cv2.putText(hdr, f'{s}: 4 images with most errors (sum over models). white=GT green=matched red=unmatched pred', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f'gallery_{s}.jpg'), np.vstack([hdr] + rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
        summary[s]['gallery_images'] = err
    tot_all = {m: [sum(res[r][m][k] for r in allrel) for k in range(3)] for m in PRED}
    summary['all_test'] = {m: dict(tp=t[0], fp=t[1], fn=t[2], f1_approx=round(100 * 2 * t[0] / max(2 * t[0] + t[1] + t[2], 1), 2)) for m, t in tot_all.items()}
    summary['note'] = 'approximate matcher (half-res polylines); compare all_test with reports/culane_by_scenario.csv (official C++)'
    (out / 'gallery_summary.json').write_text(json.dumps(summary, indent=2)); print(json.dumps(summary['all_test'], indent=1)); print('GALLERY DONE', out)


if __name__ == '__main__':
    main()
