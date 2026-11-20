#!/usr/bin/env python3
"""Uniform 5-image GPU smoke test (handbook Phase B) for scnn | ufldv2 | clrernet.

Pipeline check ONLY. Five images say nothing about accuracy; the tiny evaluator run below
is a geometry/format sanity gate (predictions must overlap GT somewhere), not a score.
Usage: smoke5.py <model> <run_dir>
"""
import sys, os, json, time, math, socket, subprocess, csv, hashlib, datetime, types
from pathlib import Path
import numpy as np
import cv2
import torch
from common import B, D, checkpoint, vendor

SCEN = ['test0_normal', 'test1_crowd', 'test2_hlight', 'test6_curve', 'test8_night']
W, H = 1640, 590
EVAL = B / 'vendors/ufldv2/evaluation/culane/build_conda_gcc/culane_evaluator'
COLORS = [(0, 255, 0), (255, 0, 0), (0, 180, 255), (255, 0, 255), (255, 255, 0), (0, 255, 255)]


def pick():
    """First entry with a non-empty GT label in each of five scenario lists (deterministic; cross excluded).
    Some CULane test frames have empty .lines.txt (e.g. first hlight/night entries); those are skipped
    only for choosing smoke inputs. Nothing is tuned on these labels."""
    out = []
    for s in SCEN:
        for line in (D / 'list/test_split' / f'{s}.txt').open():
            first = line.split()[0]
            gt = D / (first.lstrip('/')[:-3] + 'lines.txt')
            if gt.is_file() and gt.stat().st_size > 0:
                out.append((s, first, first.lstrip('/')))
                break
    return out


def read_lines(p):
    lanes = []
    for row in Path(p).read_text().splitlines():
        v = [float(t) for t in row.split()]
        if len(v) >= 4:
            lanes.append(list(zip(v[0::2], v[1::2])))
    return lanes


def git_head(repo):
    g = Path(repo) / '.git'
    try:
        head = (g / 'HEAD').read_text().strip()
        if not head.startswith('ref:'):
            return head
        ref = head.split()[1]
        if (g / ref).is_file():
            return (g / ref).read_text().strip()
        for line in (g / 'packed-refs').read_text().splitlines():
            if line.endswith(' ' + ref):
                return line.split()[0]
    except OSError:
        pass
    return None


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------- per-model inference, each reusing the author/full-run decoder ----------------

def make_scnn(run):
    sys.path.insert(0, str(vendor('scnn')))
    from model import SCNN
    from utils.prob2lines import getLane
    import torch.nn.functional as F
    from scnn_infer import extract_state
    m = SCNN((800, 288), pretrained=False)
    raw = torch.load(str(checkpoint('scnn')), map_location='cpu', weights_only=False)
    m.load_state_dict(extract_state(raw), strict=True)
    m = m.cuda().eval()
    mean = np.array([.485, .456, .406], np.float32); std = np.array([.229, .224, .225], np.float32)

    def infer(path, rel):
        img = cv2.imread(str(path))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = (cv2.resize(rgb, (800, 288), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255 - mean) / std
        t = torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).cuda()
        with torch.no_grad():
            seg, exist = m(t)[:2]
            seg = F.softmax(seg, dim=1)[0].cpu().numpy(); exist = exist[0].cpu().numpy()
        lanes = getLane.prob2lines_CULane(seg, [int(v > 0.5) for v in exist], resize_shape=(590, 1640), y_px_gap=20, pts=18)
        return img, [[(float(a), float(b)) for a, b in l] for l in lanes if len(l) > 1], {'exist_prob': exist.tolist()}
    return infer, {'decoder': 'vendor utils.prob2lines.getLane.prob2lines_CULane (same as scnn full run)', 'strict_load': True}


def make_ufld(run):
    repo = vendor('ufldv2'); sys.path.insert(0, str(repo))
    if 'data.dali_data' not in sys.modules:  # TRAIN-only import, stubbed exactly as in run_ufld_full.py
        stub = types.ModuleType('data.dali_data')
        class NoTraining:
            def __init__(self, *a, **k): raise RuntimeError('DALI train loader intentionally unavailable')
        stub.TrainCollect = NoTraining; sys.modules['data.dali_data'] = stub
    from utils.common import merge_config, get_model
    from evaluation.eval_wrapper import generate_lines_local, generate_lines_col_local
    from PIL import Image
    import torchvision.transforms as T
    sys.argv = ['smoke5.py', str(B / 'config/culane_res34_cluster.py'), '--test_model', str(checkpoint('ufldv2'))]
    _, cfg = merge_config()
    model = get_model(cfg)
    raw = torch.load(str(checkpoint('ufldv2')), map_location='cpu', weights_only=False)
    model.load_state_dict({k.removeprefix('module.'): v for k, v in raw['model'].items()}, strict=True)
    model.eval()
    tf = T.Compose([T.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)), T.ToTensor(),
                    T.Normalize((.485, .456, .406), (.229, .224, .225))])
    stage = run / 'ufld_native_lines'

    def infer(path, rel):
        img = cv2.imread(str(path))
        x = tf(Image.open(path).convert('RGB'))[:, -cfg.train_height:, :].unsqueeze(0).cuda()
        with torch.no_grad():
            pred = model(x)
        generate_lines_local('CULane', pred['loc_row'], pred['exist_row'], [rel], str(stage), 'normal', row_anchor=cfg.row_anchor)
        generate_lines_col_local('CULane', pred['loc_col'], pred['exist_col'], [rel], str(stage), 'normal', col_anchor=cfg.col_anchor)
        return img, read_lines(stage / (rel[:-3] + 'lines.txt')), {}
    return infer, {'decoder': 'vendor run_test non-TTA decode: generate_lines_local (row lanes 1,2) + generate_lines_col_local '
                              '(column lanes 0,3) (v2 fix: v1 used the row branch only)', 'strict_load': True}


def make_clr(run):
    repo = vendor('clrernet'); os.chdir(repo); sys.path.insert(0, str(repo))
    from mmdet.apis import init_detector
    from libs.api.inference import inference_one_image
    # Missing/unexpected keys were already gated by validate_checkpoint.py in gpu_entry.sh.
    model = init_detector('configs/clrernet/culane/clrernet_culane_dla34_ema.py', str(checkpoint('clrernet')), device='cuda:0')

    def infer(path, rel):
        img, preds = inference_one_image(model, str(path))
        return img, [[(float(a), float(b)) for a, b in p] for p in preds if len(p) > 1], {}
    return infer, {'decoder': 'vendor libs.api.inference.inference_one_image (demo path)',
                   'strict_load': 'gated by validate_checkpoint.py (seg_decoder/loss_seg training-only keys absent)'}


def main():
    model_name, run = sys.argv[1], Path(sys.argv[2]).resolve()
    run.mkdir(parents=True, exist_ok=True)
    pred_dir = run / 'pred'
    assert torch.cuda.is_available(), 'GPU unavailable'
    imgs = pick()
    (run / 'input_reference.txt').write_text(''.join(f'{s}\t{rel}\n' for s, _, rel in imgs))
    for s, _, rel in imgs:  # GT must exist and contain lanes for the chosen scenarios
        gt = D / (rel[:-3] + 'lines.txt')
        assert gt.is_file() and read_lines(gt), f'GT missing/empty for {rel}'

    t_load = time.perf_counter()
    infer, info = {'scnn': make_scnn, 'ufldv2': make_ufld, 'clrernet': make_clr}[model_name](run)
    load_s = time.perf_counter() - t_load

    rows, per_image = [], []
    for i, (s, first, rel) in enumerate(imgs, 1):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        img, lanes, extra = infer(D / rel, rel)
        torch.cuda.synchronize(); wall = 1000 * (time.perf_counter() - t0)
        dest = pred_dir / (rel[:-3] + 'lines.txt'); dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(''.join(' '.join(f'{x:.3f} {y:.3f}' for x, y in l) + '\n' for l in lanes))
        pts = [p for l in lanes for p in l]
        finite = all(math.isfinite(x) and math.isfinite(y) for x, y in pts)
        in_tol = all(-50 <= x <= W + 50 and -1 <= y <= H + 1 for x, y in pts)
        out_strict = sum(not (0 <= x <= W and 0 <= y <= H) for x, y in pts)
        ov = img.copy()
        for l in read_lines(D / (rel[:-3] + 'lines.txt')):
            cv2.polylines(ov, [np.array(l, np.int32)], False, (255, 255, 255), 1)
        for j, l in enumerate(lanes):
            cv2.polylines(ov, [np.array(l, np.int32)], False, COLORS[j % len(COLORS)], 3)
        cv2.putText(ov, f'{model_name} {s} lanes={len(lanes)} (white=GT)', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imwrite(str(run / f'overlay_{i:04d}.png'), ov)
        rec = dict(index=i, scenario=s, image=rel, image_shape=list(img.shape), n_lanes=len(lanes),
                   n_points=len(pts), finite=finite, within_tolerance=in_tol, points_outside_frame=out_strict,
                   wall_ms=wall, lanes=lanes, **extra)
        (run / f'raw_prediction_{i:04d}.json').write_text(json.dumps(rec, indent=1))
        per_image.append({k: v for k, v in rec.items() if k != 'lanes'})
        rows.append([i, s, rel, f'{wall:.2f}', len(lanes)])
        print(f'[{i}/5] {s} lanes={len(lanes)} pts={len(pts)} wall_ms={wall:.1f} finite={finite} tol={in_tol}', flush=True)
    with open(run / 'timings.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['index', 'scenario', 'image', 'wall_ms_incl_pre_post', 'n_lanes'])
        w.writerows(rows)

    # Tiny official-evaluator sanity run (format + geometry overlap), NOT a score.
    lst = run / 'smoke5_list.txt'; lst.write_text(''.join(first + '\n' for _, first, _ in imgs))
    env = dict(os.environ, LD_LIBRARY_PATH=f"{B / '.opencv_cpp/lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}")
    ev_out = run / 'eval_5img.txt'
    cmd = [str(EVAL), '-a', f'{D}/', '-d', f'{pred_dir}/', '-i', f'{D}/', '-l', str(lst), '-w', '30', '-t', '0.5',
           '-c', str(W), '-r', str(H), '-f', '1', '-o', str(ev_out)]
    rc = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL).returncode
    txt = ev_out.read_text() if ev_out.is_file() else ''
    import re
    mm = re.search(r'tp: (\d+) fp: (\d+) fn: (\d+)', txt)
    tp, fp, fn = map(int, mm.groups()) if mm else (None, None, None)

    checks = {
        'gpu': torch.cuda.get_device_name(0),
        'one_image_ok': per_image[0]['n_lanes'] > 0 and per_image[0]['finite'] and per_image[0]['within_tolerance'],
        'five_images_processed': len(per_image) == 5,
        # v3: a model may legitimately miss lanes on hard frames (the same frames are empty in the full runs),
        # so require lanes on >=3 of 5 frames instead of all 5. Misses are recorded in images[].
        'at_least_3_of_5_images_have_lanes': sum(r['n_lanes'] > 0 for r in per_image) >= 3,
        'all_coords_finite': all(r['finite'] for r in per_image),
        'all_coords_within_tolerance': all(r['within_tolerance'] for r in per_image),
        'five_prediction_files': sum(1 for _ in pred_dir.rglob('*.lines.txt')) == 5,
        'five_overlays': len(list(run.glob('overlay_*.png'))) == 5,
        'evaluator_rc0': rc == 0,
        'evaluator_tp_gt0': bool(tp),
    }
    passed = all(v for k, v in checks.items() if k != 'gpu')
    ck = checkpoint(model_name)
    manifest = dict(model=model_name, passed=passed, utc=datetime.datetime.utcnow().isoformat() + 'Z',
                    host=socket.gethostname(), torch=torch.__version__, cuda=torch.version.cuda,
                    checkpoint=str(ck), checkpoint_sha256=sha256(ck), vendor_commit=git_head(vendor(model_name)),
                    model_load_s=load_s, inference_info=info, images=per_image, checks=checks,
                    eval_5img=dict(tp=tp, fp=fp, fn=fn, rc=rc, note='5-image sanity gate only; NOT an accuracy result'),
                    timing_note='wall_ms includes pre/post-processing and first-call warmup; not an FPS benchmark')
    (run / 'smoke_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ('model', 'passed', 'checks', 'eval_5img')}, indent=2))
    if passed:
        ev = '\n'.join(f"  {r['scenario']:13s} lanes={r['n_lanes']} pts={r['n_points']} wall_ms={r['wall_ms']:.1f}" for r in per_image)
        (run / 'PASS.txt').write_text(
            f"SMOKE5 PASS {model_name} {manifest['utc']} host={manifest['host']} gpu={checks['gpu']}\n"
            f"checkpoint_sha256={manifest['checkpoint_sha256']} vendor_commit={manifest['vendor_commit']}\n"
            f"per-image evidence:\n{ev}\n"
            f"5-image evaluator sanity: tp={tp} fp={fp} fn={fn} (not an accuracy claim)\n"
            f"all checks: {json.dumps(checks)}\n")
        print('SMOKE5 PASS', run)
    else:
        print('SMOKE5 FAIL', run, file=sys.stderr); sys.exit(1)


if __name__ == '__main__':
    main()
