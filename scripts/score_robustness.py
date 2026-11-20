#!/usr/bin/env python3
"""Score robustness predictions with the official CULane C++ evaluator (width 30, IoU 0.5, 1640x590).
Checks that all models received identical corrupted inputs (md5 of first 3 images per condition).
Usage: score_robustness.py <root> [--limit N]   (root contains <model>/<cond>/pred)
"""
import sys, os, re, json, csv, subprocess
from pathlib import Path
from robust_common import B, D, subset, CONDS, cond_name, SEVERITY, STRIDE

EVAL = B / 'vendors/ufldv2/evaluation/culane/build_conda_gcc/culane_evaluator'
MODELS = ['scnn', 'ufldv2', 'clrernet']


def run(lst, pred, outf):
    # Reuse an existing eval.txt only if it is newer than every prediction file in that condition (unchanged inputs).
    if outf.is_file():
        newest = max((p.stat().st_mtime for p in Path(pred).rglob('*.lines.txt')), default=0)
        if outf.stat().st_mtime > newest:
            tp, fp, fn = map(int, re.search(r'tp: (\d+) fp: (\d+) fn: (\d+)', outf.read_text()).groups())
            p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
            return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=100 * 2 * p * r / (p + r) if p + r else 0.0)
    env = dict(os.environ, LD_LIBRARY_PATH=f"{B / '.opencv_cpp/lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}")
    subprocess.run([str(EVAL), '-a', f'{D}/', '-d', f'{pred}/', '-i', f'{D}/', '-l', str(lst), '-w', '30', '-t', '0.5',
                    '-c', '1640', '-r', '590', '-f', '1', '-o', str(outf)], env=env, stdout=subprocess.DEVNULL, check=True)
    tp, fp, fn = map(int, re.search(r'tp: (\d+) fp: (\d+) fn: (\d+)', outf.read_text()).groups())
    p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=100 * 2 * p * r / (p + r) if p + r else 0.0)


def main():
    global MODELS
    root = Path(sys.argv[1]).resolve()
    limit = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else None
    if '--models' in sys.argv:   # e.g. add fine-tuned variants: scnn,ufldv2,clrernet,clrernet_ft_control,clrernet_ft_corrupt
        MODELS = sys.argv[sys.argv.index('--models') + 1].split(',')
    items = subset()[:limit] if limit else subset()
    lst = root / 'subset_list.txt'; lst.write_text(''.join(e + '\n' for _, e, _ in items))
    md5 = {}
    for m in MODELS:   # merge manifests of split (chunked) jobs: robustness_manifest*.json
        md5[m] = {}
        for f in sorted((root / m).glob('robustness_manifest*.json')):
            md5[m].update(json.loads(f.read_text())['input_md5_first3'])
    want = {f'{cond_name(c, s)}/{i}' for c, s in CONDS for i in range(min(3, len(items)))}
    for m in MODELS:
        assert set(md5[m]) >= want, f'{m}: input checksums missing for {sorted(want - set(md5[m]))[:5]}'
    same = all(md5[m][k] == md5['scnn'][k] for m in MODELS for k in want)
    assert same, 'models did not receive identical corrupted inputs'
    for m in MODELS:
        for c, s in CONDS:
            n = sum(1 for _ in (root / m / cond_name(c, s) / 'pred').rglob('*.lines.txt'))
            assert n == len(items), f'{m} {cond_name(c, s)}: {n} prediction files, expected {len(items)}'
    rows, res = [], {}
    for m in MODELS:
        res[m] = {}
        for c, s in CONDS:
            cn = cond_name(c, s)
            r = run(lst, root / m / cn / 'pred', root / m / cn / 'eval.txt'); res[m][cn] = r
        clean = res[m]['clean']['f1']
        for c, s in CONDS:
            cn = cond_name(c, s); r = res[m][cn]
            rows.append(dict(model=m, condition=c, severity=s, images=len(items), tp=r['tp'], fp=r['fp'], fn=r['fn'],
                             precision=round(r['precision'], 4), recall=round(r['recall'], 4), f1=round(r['f1'], 2),
                             f1_drop_vs_clean=round(clean - r['f1'], 2), relative_f1=round(r['f1'] / clean, 4) if clean else None))
        print(m, {k: round(v['f1'], 2) for k, v in res[m].items()}, flush=True)
    with open(root / 'robustness.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = {}
    for m in MODELS:
        corr = [r for r in rows if r['model'] == m and r['condition'] != 'clean']
        summary[m] = dict(clean_f1=res[m]['clean']['f1'], mean_corrupted_f1=sum(r['f1'] for r in corr) / len(corr),
                          mean_relative_f1=sum(r['relative_f1'] for r in corr) / len(corr),
                          per_corruption_mean_f1={c: sum(r['f1'] for r in corr if r['condition'] == c) / 3 for c in {r['condition'] for r in corr}})
    (root / 'robustness_summary.json').write_text(json.dumps(dict(images=len(items), stride=STRIDE, severity=SEVERITY,
        identical_inputs_verified=same, evaluator='CULane official C++ (width 30, IoU 0.5)', summary=summary), indent=2))
    print(json.dumps(summary, indent=1)); print('SCORE DONE', root)


if __name__ == '__main__':
    main()
