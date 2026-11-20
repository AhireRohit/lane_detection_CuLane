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


def _pt_to_polyline(p, L):
    """Euclidean distance from points p (n,2) to the polyline L (m,2)."""
    if len(L) == 1: return np.linalg.norm(p - L[0], axis=1)
    a, b = L[:-1], L[1:]; ab = b - a; den = np.maximum((ab ** 2).sum(1), 1e-12)
    t = np.clip(((p[:, None, :] - a[None]) * ab[None]).sum(2) / den[None], 0, 1)
    return np.linalg.norm(p[:, None, :] - (a[None] + t[..., None] * ab[None]), axis=2).min(1)


def lane_diff(a, b):
    """Symmetric max point-to-curve distance (px) between two lanes. Works for any lane direction: UFLDv2's side lanes
    are column-anchored (x fixed, y predicted), where comparing x at equal y would blow up small differences."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(max(_pt_to_polyline(a, b).max(), _pt_to_polyline(b, a).max()))


def shape_diff(a, b):
    """One-sided version: how far the points of either lane lie from the other's curve, whichever fits better
    (ignores a lane being drawn a little longer or shorter)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(min(_pt_to_polyline(a, b).max(), _pt_to_polyline(b, a).max()))


def compare(cpu, gpu):
    """Greedy one-to-one matching of lanes by max x difference; returns (same_count, max diff over matched lanes)."""
    if len(cpu) != len(gpu): return False, np.inf, np.inf
    if not cpu: return True, 0.0, 0.0
    C = np.array([[lane_diff(np.asarray(c, float), g) for g in gpu] for c in cpu])
    used, worst, shape = set(), 0.0, 0.0
    for i in np.argsort(C.min(1)):
        j = min((j for j in range(len(gpu)) if j not in used), key=lambda j: C[i, j]); used.add(j)
        worst = max(worst, C[i, j]); shape = max(shape, shape_diff(cpu[i], gpu[j]))
    return True, worst, shape


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
        same, worst, shape = compare(lanes, gpu)
        rows.append(dict(img=rel, cpu_lanes=len(lanes), gpu_lanes=len(gpu), same_count=same, max_px=worst, shape_px=shape, ms=round(ms, 1)))
        if k % 25 == 0: print(model, k, len(lanes), len(gpu), round(worst, 3), round(ms), flush=True)
    px = np.array([r['max_px'] for r in rows if r['same_count']]); sp = np.array([r['shape_px'] for r in rows if r['same_count']])
    res = dict(model=model, images=len(rows), gpu_predictions=str(root), torch=torch.__version__,
               same_lane_count=float(np.mean([r['same_count'] for r in rows])),
               max_px_median=float(np.median(px)) if len(px) else None, max_px_p95=float(np.percentile(px, 95)) if len(px) else None,
               within_1px=float(np.mean(px <= 1.0)) if len(px) else None, within_5px=float(np.mean(px <= 5.0)) if len(px) else None,
               shape_px_median=float(np.median(sp)) if len(sp) else None, shape_px_p95=float(np.percentile(sp, 95)) if len(sp) else None,
               shape_within_5px=float(np.mean(sp <= 5.0)) if len(sp) else None,
               metric='max point-to-curve distance between matched lanes, in 1640x590 px (shape_*: one-sided, ignores lane length)',
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


def dump(model, n, outp):
    """Run the demo's builder on the sampled test images on the current device (LANE_DEVICE) and save the lanes."""
    from app_infer import build
    import torch
    if os.environ.get('LANE_DEVICE', 'cuda') == 'cpu': torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    pre, fwd, post, info = build(model, Path(outp).parent / f'work_dump_{model}')
    with open(outp, 'w') as f:
        for k, rel in enumerate(sample(n)):
            lanes = [[[float(x), float(y)] for x, y in l] for l in post(fwd(pre(cv2.imread(str(D / rel))))) if len(l) > 1]
            f.write(json.dumps(dict(img=rel, lanes=lanes)) + '\n')
            if k % 50 == 0: print(model, k, len(lanes), flush=True)
    print('DUMP DONE', model, outp, 'torch', torch.__version__, 'device', os.environ.get('LANE_DEVICE', 'cuda'))


def cmp(a_path, b_path, outp):
    """Compare two dumps image by image (same code and loader; only the device / torch build differs)."""
    A = [json.loads(l) for l in open(a_path)]; Bd = {r['img']: r['lanes'] for r in map(json.loads, open(b_path))}
    rows = []
    for r in A:
        same, worst, shape = compare(r['lanes'], [np.asarray(l, float) for l in Bd[r['img']]])
        rows.append(dict(img=r['img'], same_count=same, max_px=worst, shape_px=shape, n=len(r['lanes'])))
    px = np.array([r['max_px'] for r in rows if r['same_count']])
    res = dict(a=a_path, b=b_path, images=len(rows), same_lane_count=float(np.mean([r['same_count'] for r in rows])),
               max_px_median=float(np.median(px)), max_px_p95=float(np.percentile(px, 95)), max_px_max=float(px.max()),
               within_0p1px=float(np.mean(px <= 0.1)), within_1px=float(np.mean(px <= 1)), within_5px=float(np.mean(px <= 5)),
               worst=sorted(rows, key=lambda r: -r['max_px'] if r['same_count'] else -1e9)[:10])
    Path(outp).write_text(json.dumps(res, indent=2)); print(json.dumps({k: v for k, v in res.items() if k != 'worst'}, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'nms': nms_check(int(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == 'dump': dump(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif sys.argv[1] == 'cmp': cmp(sys.argv[2], sys.argv[3], sys.argv[4])
    else: e2e(sys.argv[2], int(sys.argv[3]), sys.argv[4])
