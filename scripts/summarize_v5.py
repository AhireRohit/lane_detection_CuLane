#!/usr/bin/env python3
"""Collect the v5 results into report tables (standard library only; safe on the login node).

  reports/tusimple_cross_dataset.csv   TuSimple (233-clip public subset): official metric + CULane-style F1
  reports/tracker_labeled.csv          B0r vs B2 vs B2obs: CULane test clips (1 Hz) and TuSimple clips (20 fps)
  reports/onnx_clrernet.csv            ONNX FP32 / INT8: CULane test F1 (official evaluator) and CPU latency
  reports/v5_comparisons.csv           paired tracker difference (TuSimple) and robust vs control on TuSimple
Usage: summarize_v5.py [--print]
"""
import csv, glob, json, os, re, sys
from pathlib import Path

B = Path(__file__).resolve().parents[1]
RES, REP = B / 'results', B / 'reports'


def culane_f1(path):
    try: t = open(path).read()
    except OSError: return None
    m = re.search(r'Fmeasure: ([\d.]+)', t)
    return round(100 * float(m.group(1)), 2) if m else None


def newest(pattern):
    c = sorted(glob.glob(str(pattern)))
    return c[-1] if c else None


def tusimple():
    rows = []
    for d in sorted((RES / 'tusimple').glob('*/tusimple_metrics.json')):
        r = json.load(open(d)); tag = d.parent.name
        for v, m in r['results'].items():
            rows.append(dict(model=tag, variant=v, frames=m['frames'], accuracy=round(100 * m['accuracy'], 2),
                             fp_rate=round(100 * m['fp_rate'], 2), fn_rate=round(100 * m['fn_rate'], 2),
                             f1=round(100 * m['f1'], 2), f1_ci95_low=round(100 * m['ci95']['f1'][0], 2),
                             f1_ci95_high=round(100 * m['ci95']['f1'][1], 2),
                             culane_style_f1=culane_f1(d.parent / f'culane_eval_{v}.txt'), checkpoint=r['checkpoint'].split('/')[-2] if '/' in r['checkpoint'] else r['checkpoint']))
    return rows


def tracker():
    rows = []
    for m in ('clrernet', 'scnn', 'ufldv2_notta'):
        for v in ('B0r', 'B2', 'B2obs'):
            f = newest(RES / 'tracker_culane' / f'{m}_{v}' / 'official_eval' / '*' / 'all_test.txt')
            rows.append(dict(dataset='CULane test (419 clips, 1 frame/s, 34680 frames)', model=m, variant=v,
                             f1=culane_f1(f) if f else None, metric='official CULane F1 (IoU 0.5, width 30)'))
    for m in ('clrernet', 'scnn', 'ufldv2'):
        p = RES / 'tusimple' / m / 'tusimple_metrics.json'
        if not p.is_file(): continue
        r = json.load(open(p))['results']
        for v in ('B0r', 'B2', 'B2obs'):
            if v in r:
                rows.append(dict(dataset='TuSimple subset (233 clips, 20 fps, scored on frame 20)', model=m, variant=v,
                                 f1=round(100 * r[v]['f1'], 2), metric='TuSimple F1 (official LaneEval FP/FN)'))
    return rows


def onnx():
    rows = []
    ex = RES / 'onnx' / 'export.json'
    sp = json.load(open(ex))['speed'] if ex.is_file() else {}
    for v in ('torch_fp32', 'onnx_fp32', 'int8_all', 'int8_bbneck'):
        f = newest(RES / 'onnx_eval' / v / 'official_eval' / '*' / 'all_test.txt') if v != 'torch_fp32' else None
        rows.append(dict(variant=v, culane_test_f1=culane_f1(f) if f else (81.55 if v == 'torch_fp32' else None),
                         cpu_forward_ms_1thread=sp.get(f'{v}_threads1', {}).get('forward_ms_median'),
                         cpu_forward_ms_4threads=sp.get(f'{v}_threads4', {}).get('forward_ms_median'),
                         cpu_full_ms_4threads=sp.get(f'{v}_threads4', {}).get('full_ms_median'),
                         note='torch F1 = released GPU run (report 5.1)' if v == 'torch_fp32' else ''))
    return rows


def _f1(fp, fn):
    p, r = 1 - fp, 1 - fn
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def _paired(a, b, boot=2000, seed=0):
    """Clip-level paired bootstrap of TuSimple F1(b) - F1(a); a, b = per-frame [acc, fp, fn] lists, same clips."""
    import random
    rnd = random.Random(seed); n = len(a); d = []
    f = lambda rows, idx: _f1(sum(rows[i][1] for i in idx) / n, sum(rows[i][2] for i in idx) / n)
    full = f(b, range(n)) - f(a, range(n))
    for _ in range(boot):
        idx = [rnd.randrange(n) for _ in range(n)]; d.append(f(b, idx) - f(a, idx))
    d.sort()
    return round(100 * full, 2), round(100 * d[int(0.025 * boot)], 2), round(100 * d[int(0.975 * boot)], 2)


def comparisons():
    rows = []
    for m in ('clrernet', 'scnn', 'ufldv2'):
        p = RES / 'tusimple' / m / 'tusimple_metrics.json'
        if not p.is_file(): continue
        r = json.load(open(p))['results']
        if 'B2' in r:
            d, lo, hi = _paired(r['B0r']['per_frame'], r['B2']['per_frame'])
            rows.append(dict(comparison=f'{m}: tracker B2 vs B0r, TuSimple 20 fps (233 clips)', delta_f1=d, ci95_low=lo,
                             ci95_high=hi, method='paired clip bootstrap, 2000 resamples'))
    arms = {}
    for arm in ('control', 'corrupt'):
        vals = []
        for s in (0, 1, 2):
            p = RES / 'tusimple' / f'ft_{arm}_s{s}' / 'tusimple_metrics.json'
            if p.is_file(): vals.append(100 * json.load(open(p))['results']['raw']['f1'])
        arms[arm] = vals
    if len(arms['control']) == 3 and len(arms['corrupt']) == 3:
        import statistics as st
        mc, mr = st.mean(arms['control']), st.mean(arms['corrupt'])
        rows.append(dict(comparison='robust (corruption) fine-tune vs same-schedule control, TuSimple F1, 3 seeds each',
                         delta_f1=round(mr - mc, 2), ci95_low=None, ci95_high=None,
                         method=f"control {mc:.2f} +- {st.stdev(arms['control']):.2f}, corrupt {mr:.2f} +- "
                                f"{st.stdev(arms['corrupt']):.2f} (mean +- sd over seeds); per seed control "
                                f"{[round(v, 2) for v in arms['control']]}, corrupt {[round(v, 2) for v in arms['corrupt']]}"))
    return rows


def write(name, rows):
    if not rows: return
    REP.mkdir(exist_ok=True)
    with open(REP / name, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if '--print' in sys.argv:
        print(f'== {name}')
        for r in rows: print('  ' + '  '.join(f'{k}={v}' for k, v in r.items()))


if __name__ == '__main__':
    write('tusimple_cross_dataset.csv', tusimple())
    write('tracker_labeled.csv', tracker())
    write('onnx_clrernet.csv', onnx())
    write('v5_comparisons.csv', comparisons())
