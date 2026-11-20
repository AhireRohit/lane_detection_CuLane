#!/usr/bin/env python3
"""Same-hardware latency benchmark (handbook Phase D): one P100, batch 1, FP32, cudnn.benchmark=False (default).

Groups measured per frame (CUDA-synchronized at every stage boundary):
  (1) forward_ms : network forward only (CLRerNet: backbone+neck+head, before NMS).
  (2) full_ms    : preprocess from an in-memory BGR 1640x590 frame + forward + decode/NMS
                   to source-frame lane coordinates. Disk read/JPEG decode EXCLUDED (frames preloaded),
                   no file writing, no drawing.
  (3) video end-to-end is measured later in the dashcam pipeline.
Input: the first WARMUP+MEASURE entries of list/test.txt, identical for all models.
Usage: benchmark_speed.py <model> <run_dir>
"""
import sys, os, json, time, csv, socket, datetime, types, subprocess
from pathlib import Path
import numpy as np
import cv2
import torch
from common import B, D, checkpoint, vendor, test_images

WARMUP, MEASURE = 50, 200
# LANE_DEVICE=cpu runs the same builders without a GPU (Hugging Face Space demo; equivalence checked by
# scripts/verify_cpu.py). Every benchmark number in reports/ used the default, cuda.
DEV = os.environ.get('LANE_DEVICE', 'cuda')


def build_scnn():
    sys.path.insert(0, str(vendor('scnn')))
    from model import SCNN
    from utils.prob2lines import getLane
    import torch.nn.functional as F
    from scnn_infer import extract_state
    m = SCNN((800, 288), pretrained=False)
    m.load_state_dict(extract_state(torch.load(str(checkpoint('scnn')), map_location='cpu', weights_only=False)), strict=True)
    m = m.to(DEV).eval()
    mean = np.array([.485, .456, .406], np.float32); std = np.array([.229, .224, .225], np.float32)

    def pre(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = (cv2.resize(rgb, (800, 288), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255 - mean) / std
        return torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(DEV)

    def fwd(t):
        return m(t)[:2]

    def post(o):
        seg, exist = o
        seg = F.softmax(seg, dim=1)[0].cpu().numpy(); exist = exist[0].cpu().numpy()
        lanes = getLane.prob2lines_CULane(seg, [int(v > 0.5) for v in exist], resize_shape=(590, 1640), y_px_gap=20, pts=18)
        return [l for l in lanes if len(l) > 1]

    info = dict(input_resolution='800x288 (full frame resized)',
                decode='softmax + GPU->CPU + vendor prob2lines_CULane (CPU, same as scored full run)')
    return pre, fwd, post, info, None


def build_ufld(run):
    repo = vendor('ufldv2'); sys.path.insert(0, str(repo))
    if 'data.dali_data' not in sys.modules:
        stub = types.ModuleType('data.dali_data')
        class NoTraining:
            def __init__(self, *a, **k): raise RuntimeError('DALI train loader intentionally unavailable')
        stub.TrainCollect = NoTraining; sys.modules['data.dali_data'] = stub
    from utils.common import merge_config, get_model
    from evaluation.eval_wrapper import generate_lines_local, generate_lines_col_local
    from PIL import Image
    import torchvision.transforms as T
    sys.argv = ['benchmark_speed.py', str(B / 'config/culane_res34_cluster.py'), '--test_model', str(checkpoint('ufldv2'))]
    _, cfg = merge_config()
    if DEV == 'cuda':
        model = get_model(cfg)
    else:   # vendor get_model() calls .cuda(); build the same network on CPU (backbone init is overwritten by the strict load)
        from model.model_culane import parsingNet
        model = parsingNet(pretrained=False, backbone=cfg.backbone, num_grid_row=cfg.num_cell_row, num_cls_row=cfg.num_row,
                           num_grid_col=cfg.num_cell_col, num_cls_col=cfg.num_col, num_lane_on_row=cfg.num_lanes,
                           num_lane_on_col=cfg.num_lanes, use_aux=cfg.use_aux, input_height=cfg.train_height,
                           input_width=cfg.train_width, fc_norm=cfg.fc_norm)
    raw = torch.load(str(checkpoint('ufldv2')), map_location='cpu', weights_only=False)
    model.load_state_dict({k.removeprefix('module.'): v for k, v in raw['model'].items()}, strict=True)
    model.eval()
    tf = T.Compose([T.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)), T.ToTensor(),
                    T.Normalize((.485, .456, .406), (.229, .224, .225))])
    ra, ca = cfg.row_anchor, cfg.col_anchor

    def pre(img):
        return tf(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))[:, -cfg.train_height:, :].unsqueeze(0).to(DEV)

    def fwd(x):
        return model(x)

    def post(pred):
        # In-memory copy of the author's non-TTA CULane decode in eval_wrapper.run_test (item 0), returning
        # coordinates instead of writing a file:
        #   generate_lines_local     -> row-anchor ego lanes 1, 2   (valid if > num_cls/2 anchors)
        #   generate_lines_col_local -> column-anchor side lanes 0, 3 (valid if > num_cls/4 anchors), appended after.
        # v2 FIX (2026-11-19): v1 implemented only the row branch, so UFLDv2 lost its two outer lanes.
        lanes = []
        out, ext = pred['loc_row'], pred['exist_row']
        mi = out.argmax(1).cpu(); valid = ext.argmax(1).cpu(); out = out.cpu()
        num_cls, G = out.shape[2], out.shape[1]
        for i in [1, 2]:
            if valid[0, :, i].sum() > num_cls / 2:
                lane = []
                for k in range(valid.shape[1]):
                    if valid[0, k, i]:
                        ind = torch.tensor(list(range(max(0, mi[0, k, i] - 1), min(G - 1, mi[0, k, i] + 1) + 1)))
                        v = (out[0, ind, k, i].softmax(0) * ind.float()).sum() + 0.5
                        lane.append((float(v / (G - 1) * 1640), float(ra[k] * 590)))
                lanes.append(lane)
        out, ext = pred['loc_col'], pred['exist_col']
        mi = out.argmax(1).cpu(); valid = ext.argmax(1).cpu(); out = out.cpu()
        num_cls, G = out.shape[2], out.shape[1]
        for i in [0, 3]:
            if valid[0, :, i].sum() > num_cls / 4:
                lane = []
                for k in range(valid.shape[1]):
                    if valid[0, k, i]:
                        ind = torch.tensor(list(range(max(0, mi[0, k, i] - 1), min(G - 1, mi[0, k, i] + 1) + 1)))
                        v = (out[0, ind, k, i].softmax(0) * ind.float()).sum() + 0.5
                        lane.append((float(ca[k] * 1640), float(v / (G - 1) * 590)))
                lanes.append(lane)
        return lanes

    def verify(img):
        stage = run / 'verify_ufld'; name = 'frame0.jpg'
        with torch.no_grad():
            p = fwd(pre(img))
        # Reference = the author's complete non-TTA output: row lanes written, then column lanes appended.
        generate_lines_local('CULane', p['loc_row'], p['exist_row'], [name], str(stage), 'normal', row_anchor=ra)
        generate_lines_col_local('CULane', p['loc_col'], p['exist_col'], [name], str(stage), 'normal', col_anchor=ca)
        ref = [list(zip(*[iter(map(float, r.split()))] * 2)) for r in (stage / 'frame0.lines.txt').read_text().splitlines() if r.split()]
        mine = post(p)
        if len(ref) != len(mine) or any(len(a) != len(b) for a, b in zip(ref, mine)):
            return dict(match=False, reason=f'lane/point count differs ref={[len(a) for a in ref]} mine={[len(b) for b in mine]}')
        d = max((abs(x1 - x2) + abs(y1 - y2) for a, b in zip(ref, mine) for (x1, y1), (x2, y2) in zip(a, b)), default=0.0)
        return dict(match=d < 0.01, max_abs_diff_px=d, lanes=len(ref),
                    reference='vendor generate_lines_local + generate_lines_col_local file output (%.3f), as in run_test')

    info = dict(input_resolution='1600x320 (resized to 1600x533, bottom 320 rows)',
                decode='in-memory copy of vendor run_test CULane decode: row lanes 1,2 + column lanes 0,3, NO TTA (v2 fix)',
                note='scored full run used cfg tta=True (extra forward passes); latency here is the non-TTA path')
    return pre, fwd, post, info, verify


def build_clr(run):
    repo = vendor('clrernet'); os.chdir(repo); sys.path.insert(0, str(repo))
    if DEV != 'cuda':   # the compiled lane-NMS op is CUDA-only; use the exact CPU port
        import nms_cpu; sys.modules['nms'] = nms_cpu
    from mmdet.apis import init_detector
    from libs.api.inference import get_prediction
    from libs.datasets.pipelines import Compose
    # CLR_CKPT (optional) evaluates a fine-tuned checkpoint with the same verified pipeline; default = released EMA weights.
    ckpt = os.environ.get('CLR_CKPT') or str(checkpoint('clrernet'))
    # init_detector reads the CULane test list only to fetch a colour palette (palette='none'). Where no dataset exists
    # (the public demo) pass palette='random' instead: dataset_meta/palette only affect visualisation, not predictions.
    palette = 'none' if (repo / 'dataset/culane/list/test.txt').is_file() else 'random'
    model = init_detector('configs/clrernet/culane/clrernet_culane_dla34_ema.py', ckpt, palette=palette,
                          device='cuda:0' if DEV == 'cuda' else 'cpu')
    model.bbox_head.test_cfg.as_lanes = False
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    head = model.bbox_head

    def pack(img):
        d = pipeline(dict(filename='in_memory', sub_img_name=None, img=img, gt_points=[], id_classes=[],
                          id_instances=[], img_shape=img.shape, ori_shape=img.shape))
        return dict(inputs=[d['inputs']], data_samples=[d['data_samples']])

    def pre(img):
        return model.data_preprocessor(pack(img), False)['inputs'], img.shape

    def fwd(p):
        x, shape = p
        return head(model.extract_feat(x))[-1], shape

    def post(o):
        pred, shape = o
        lanes, _ = head.get_lanes(pred, as_lanes=False, extend_bottom=head.test_cfg.extend_bottom)
        return get_prediction(lanes[0], shape[0], shape[1])

    def verify(img):
        with torch.no_grad():
            ref = get_prediction(model.test_step(pack(img))[0]['lanes'], img.shape[0], img.shape[1])
            mine = post(fwd(pre(img)))
        if len(ref) != len(mine) or any(len(a) != len(b) for a, b in zip(ref, mine)):
            return dict(match=False, reason=f'lane/point count differs ref={[len(a) for a in ref]} mine={[len(b) for b in mine]}')
        d = max((float(np.abs(np.asarray(a) - np.asarray(b)).max()) for a, b in zip(ref, mine)), default=0.0)
        return dict(match=d < 0.01, max_abs_diff_px=d, reference='vendor model.test_step + get_prediction (inference_one_image path)')

    info = dict(input_resolution='vendor val pipeline (configs/clrernet/culane/dataset_culane_clrernet.py val_al_pipeline)',
                checkpoint=ckpt,
                decode='head.get_lanes (NMS + lane decode, GPU) + vendor get_prediction (interp to source frame)',
                forward_definition='data_preprocessor counted in preprocess; forward = extract_feat + bbox_head forward')
    return pre, fwd, post, info, verify


def stats(a):
    a = np.asarray(a)
    return dict(n=int(a.size), mean_ms=float(a.mean()), median_ms=float(np.median(a)), p90_ms=float(np.percentile(a, 90)),
                p95_ms=float(np.percentile(a, 95)), min_ms=float(a.min()), max_ms=float(a.max()),
                fps_from_median=float(1000 / np.median(a)), fps_from_mean=float(1000 / a.mean()))


def main():
    name, run = sys.argv[1], Path(sys.argv[2]).resolve()
    run.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = False
    rels = []
    for r in test_images():
        rels.append(r)
        if len(rels) == WARMUP + MEASURE: break
    t = time.perf_counter()
    frames = [cv2.imread(str(D / r)) for r in rels]
    assert all(f is not None and f.shape == (590, 1640, 3) for f in frames), 'bad input frame'
    print(f'preloaded {len(frames)} frames in {time.perf_counter() - t:.1f}s', flush=True)
    (run / 'input_frames.txt').write_text('\n'.join(rels) + '\n')

    if name == 'scnn': pre, fwd, post, info, verify = build_scnn()
    elif name == 'ufldv2': pre, fwd, post, info, verify = build_ufld(run)
    else: pre, fwd, post, info, verify = build_clr(run)

    ver = verify(frames[0]) if verify else dict(match=True, note='decode path identical to scored full run code')
    print('decode equivalence:', ver, flush=True)

    torch.cuda.reset_peak_memory_stats()
    rows = []
    with torch.no_grad():
        for i, img in enumerate(frames):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            x = pre(img); torch.cuda.synchronize(); t1 = time.perf_counter()
            o = fwd(x); torch.cuda.synchronize(); t2 = time.perf_counter()
            lanes = post(o); torch.cuda.synchronize(); t3 = time.perf_counter()
            rows.append(dict(idx=i, phase='warmup' if i < WARMUP else 'measure', image=rels[i],
                             pre_ms=1000 * (t1 - t0), forward_ms=1000 * (t2 - t1), post_ms=1000 * (t3 - t2),
                             full_ms=1000 * (t3 - t0), n_lanes=len(lanes)))
            if i % 50 == 0: print(f'{name} frame {i} full_ms={rows[-1]["full_ms"]:.1f}', flush=True)
    with open(run / 'timing_samples.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    m = [r for r in rows if r['phase'] == 'measure']
    res = {k: stats([r[k] for r in m]) for k in ('forward_ms', 'full_ms', 'pre_ms', 'post_ms')}
    smi = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,clocks.sm,clocks.max.sm,temperature.gpu,utilization.gpu',
                          '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
    manifest = dict(model=name, utc=datetime.datetime.utcnow().isoformat() + 'Z', host=socket.gethostname(),
                    gpu=torch.cuda.get_device_name(0), nvidia_smi=smi, torch=torch.__version__, cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version(), cudnn_benchmark=False, precision='FP32', batch_size=1,
                    torch_threads=torch.get_num_threads(), omp_num_threads=os.environ.get('OMP_NUM_THREADS'),
                    loadavg=open('/proc/loadavg').read().strip(), warmup=WARMUP, measured=len(m),
                    input='first %d entries of CULane list/test.txt, preloaded BGR 1640x590' % len(frames),
                    timing_policy='perf_counter with torch.cuda.synchronize() at every stage boundary; '
                                  'disk read, JPEG decode, drawing and file writing excluded',
                    peak_cuda_mem_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
                    decode_equivalence=ver, model_info=info, results=res,
                    checkpoint=str(checkpoint(name)))
    (run / 'speed_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(model=name, results=res, decode_equivalence=ver), indent=2))
    if not ver.get('match'):
        print('SPEED FAIL: decode path not equivalent to vendor reference', file=sys.stderr); sys.exit(2)
    print('SPEED DONE', run)


if __name__ == '__main__':
    main()
