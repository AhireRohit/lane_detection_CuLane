#!/usr/bin/env python3
"""Check that the CPU path used by the Hugging Face Space gives the same lanes as the GPU path behind every reported number.

nms <N> <out.json>
    GPU job, clrernet venv. Runs CLRerNet on N CULane test images on the GPU; at every NMS call, also runs the CPU port
    (scripts/nms_cpu.py) on the identical proposals and compares the kept indices.
e2e <model> <N> <out.json>
    CPU-only job in the Space's Python environment (hf_space/requirements.txt), LANE_DEVICE=cpu. Runs the full
    pipeline (same builders as the demo) on N CULane test images and compares each image's lanes with the GPU
    prediction files that were scored for the report. model: scnn | ufldv2 | clrernet | clrernet_robust.
Images: every k-th entry of list/test.txt (fixed stride), identical for all models.
"""
import sys, os, json, glob, time
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import B, D, test_images

GPU_PRED = {
    'scnn': B / 'results/scnn/full/coord_output',
    'ufldv2': B / 'results/ufldv2/full_notta/culane_eval_tmp',          # non-TTA author run (the demo is non-TTA)
    'clrernet': None, 'clrernet_robust': None,
}


def pred_root(model):
    if model == 'clrernet':
        return Path(sorted(glob.glob(str(B / 'results/clrernet/full/run_*/pred')))[-1])
    if model == 'clrernet_robust':
        return Path(sorted(glob.glob(str(B / 'results/finetune/clrernet_ft_corrupt_2*-73992/eval/test_pred')))[-1])
    return GPU_PRED[model]


def sample(n):
    allimgs = list(test_images())
    step = max(1, len(allimgs) // n)
    return allimgs[::step][:n]


def read_lines(p):
    lanes = []
    for row in Path(p).read_text().splitlines():
        v = [float(t) for t in row.split()]
        if len(v) >= 4: lanes.append(np.array(v).reshape(-1, 2))
    return lanes


def lane_diff(a, b):
    """max |x difference| over the rows both lanes cover (x interpolated in y); inf if they share < 3 rows."""
    a = a[np.argsort(a[:, 1])]; b = b[np.argsort(b[:, 1])]
    lo, hi = max(a[0, 1], b[0, 1]), min(a[-1, 1], b[-1, 1])
    ys = b[(b[:, 1] >= lo) & (b[:, 1] <= hi), 1]
    if len(ys) < 3: return np.inf
    return float(np.max(np.abs(np.interp(ys, a[:, 1], a[:, 0]) - np.interp(ys, b[:, 1], b[:, 0]))))


def compare(cpu, gpu):
    """Greedy one-to-one matching of lanes by max x difference; returns (same_count, max diff over matched lanes)."""
    if len(cpu) != len(gpu): return False, np.inf
    if not cpu: return True, 0.0
    C = np.array([[lane_diff(np.asarray(c, float), g) for g in gpu] for c in cpu])
    used, worst = set(), 0.0
    for i in np.argsort(C.min(1)):
        j = min((j for j in range(len(gpu)) if j not in used), key=lambda j: C[i, j]); used.add(j); worst = max(worst, C[i, j])
    return True, worst


def e2e(model, n, outp):
    os.environ['LANE_DEVICE'] = 'cpu'
    from app_infer import build
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    pre, fwd, post, info = build(model, Path(outp).parent / f'work_{model}')
    root = pred_root(model)
    rows, t0 = [], time.time()
    for k, rel in enumerate(sample(n)):
        img = cv2.imread(str(D / rel))
        t = time.perf_counter(); lanes = [l for l in post(fwd(pre(img))) if len(l) > 1]; ms = 1000 * (time.perf_counter() - t)
        gpu = read_lines(root / (rel[:-4] + '.lines.txt'))
        same, worst = compare(lanes, gpu)
        rows.append(dict(img=rel, cpu_lanes=len(lanes), gpu_lanes=len(gpu), same_count=same, max_px=worst, ms=round(ms, 1)))
        if k % 25 == 0: print(model, k, len(lanes), len(gpu), round(worst, 3), round(ms), flush=True)
    px = np.array([r['max_px'] for r in rows if r['same_count']])
    res = dict(model=model, images=len(rows), gpu_predictions=str(root), torch=torch.__version__,
               same_lane_count=float(np.mean([r['same_count'] for r in rows])),
               max_px_median=float(np.median(px)) if len(px) else None, max_px_p95=float(np.percentile(px, 95)) if len(px) else None,
               within_1px=float(np.mean(px <= 1.0)) if len(px) else None, within_5px=float(np.mean(px <= 5.0)) if len(px) else None,
               cpu_ms_median=float(np.median([r['ms'] for r in rows])), wall_s=round(time.time() - t0),
               mismatched=[r for r in rows if not r['same_count'] or r['max_px'] > 5][:20])
    Path(outp).write_text(json.dumps(res, indent=2)); print(json.dumps({k: v for k, v in res.items() if k != 'mismatched'}, indent=1))


def nms_check(n, outp):
    import torch
    import nms_cpu
    from app_infer import build
    pre, fwd, post, info = build('clrernet', Path(outp).parent / 'work_nms')
    import libs.models.dense_heads.clrernet_head as Hd
    orig, stats = Hd.nms, dict(calls=0, mismatches=0, max_proposals=0, examples=[])

    def both(boxes, scores, overlap, top_k):
        k1, n1, p1 = orig(boxes, scores, overlap=overlap, top_k=top_k)
        k2, n2, _ = nms_cpu.nms(boxes, scores, overlap, top_k)
        a, b = k1[:int(n1)].cpu().tolist(), k2[:int(n2)].cpu().tolist()
        stats['calls'] += 1; stats['max_proposals'] = max(stats['max_proposals'], int(boxes.shape[0]))
        if a != b:
            stats['mismatches'] += 1
            if len(stats['examples']) < 10: stats['examples'].append(dict(cuda=a, cpu=b, n=int(boxes.shape[0])))
        return k1, n1, p1
    Hd.nms = both
    for k, rel in enumerate(sample(n)):
        post(fwd(pre(cv2.imread(str(D / rel)))))
        if k % 250 == 0: print('nms', k, stats['calls'], stats['mismatches'], flush=True)
    stats['images'] = n
    Path(outp).write_text(json.dumps(stats, indent=2)); print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'nms': nms_check(int(sys.argv[2]), sys.argv[3])
    else: e2e(sys.argv[2], int(sys.argv[3]), sys.argv[4])
