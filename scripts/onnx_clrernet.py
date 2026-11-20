#!/usr/bin/env python3
"""CLRerNet (DLA34, released EMA weights) -> ONNX FP32 and INT8, with CPU speed and full CULane test accuracy.

The exported graph is the network forward exactly as timed in benchmark_speed.py (backbone + neck + head up to the
final refinement stage's raw lane tensors). Preprocessing (vendor test pipeline + data_preprocessor) and decoding
(head.get_lanes: NMS, here the exact CPU port nms_cpu.py, + the vendor CULaneMetric string writer, y_step 2) stay in
Python and are shared by every variant, so only the network numerics differ.

Variants, all declared before any test-set scoring (none is chosen afterwards):
  torch_fp32    the PyTorch model on CPU (reference)
  onnx_fp32     ONNX Runtime CPU, FP32
  int8_all      ONNX Runtime static INT8 (QDQ, per-channel S8 weights, S8 activations, MinMax calibration)
  int8_bbneck   same, but only backbone + neck quantized; the lane head stays FP32
Calibration: 300 CULane TRAIN images (every k-th of list/train_gt.txt). No test image is used for calibration.

Modes:
  export                 export + quantize, equivalence checks on 50 VAL images, writer check against the original
                         GPU prediction files, CPU speed on 200 VAL images (1 and 4 threads) -> results/onnx/export.json
  infer <variant> <i> <n> write CULane .lines.txt for chunk i of n of list/test.txt -> results/onnx/<variant>/pred
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
B = HERE.parent
sys.path.insert(0, str(HERE))
from common import D, checkpoint, vendor   # noqa: E402

EXTRA = B / '.venvs' / 'onnx_extra'          # onnx + onnxruntime, appended last so the venv's numpy/torch win
sys.path.append(str(EXTRA))
OUT = B / 'results' / 'onnx'
VARIANTS = ('onnx_fp32', 'int8_all', 'int8_bbneck')
ONNX = {'onnx_fp32': OUT / 'clrernet_fp32.onnx', 'int8_all': OUT / 'clrernet_int8_all.onnx',
        'int8_bbneck': OUT / 'clrernet_int8_bbneck.onnx'}
OUTPUTS = ('cls_logits', 'anchor_params', 'lengths', 'xs')
ORI_W, ORI_H, Y_STEP = 1640, 590, 2


def build():
    import torch, nms_cpu
    sys.modules['nms'] = nms_cpu                      # CUDA lane-NMS op -> exact CPU port
    repo = vendor('clrernet'); os.chdir(repo); sys.path.insert(0, str(repo))
    from mmdet.apis import init_detector
    from libs.datasets.pipelines import Compose
    palette = 'none' if (repo / 'dataset/culane/list/test.txt').is_file() else 'random'
    model = init_detector('configs/clrernet/culane/clrernet_culane_dla34_ema.py', str(checkpoint('clrernet')),
                          palette=palette, device='cpu').eval()
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    head = model.bbox_head

    def pre(img):
        d = pipeline(dict(filename='in_memory', sub_img_name=None, img=img, gt_points=[], id_classes=[],
                          id_instances=[], img_shape=img.shape, ori_shape=img.shape))
        return model.data_preprocessor(dict(inputs=[d['inputs']], data_samples=[d['data_samples']]), False)['inputs']

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.m = model

        def forward(self, x):
            p = self.m.bbox_head(self.m.extract_feat(x))[-1]
            return tuple(p[k] for k in OUTPUTS)

    def decode(outs):
        pred = {k: torch.as_tensor(np.asarray(v)) for k, v in zip(OUTPUTS, outs)}
        lanes, _ = head.get_lanes(pred, as_lanes=True, extend_bottom=head.test_cfg.extend_bottom)
        return lanes[0]

    return torch, Net().eval(), pre, decode


def lane_string(lanes):
    """CULaneMetric.get_prediction_string (vendor), ori 1640x590, y_step 2."""
    ys = np.arange(0, ORI_H, Y_STEP) / ORI_H; out = []
    for lane in lanes:
        xs = lane(ys); m = (xs >= 0) & (xs < 1)
        lx, ly = (xs * ORI_W)[m][::-1], (ys[m] * ORI_H)[::-1]
        if len(lx) >= 2: out.append(' '.join('{:.5f} {:.5f}'.format(x, y) for x, y in zip(lx, ly)))
    return '\n'.join(out)


def parse(s):
    return [np.array(r.split(), float).reshape(-1, 2) for r in s.strip().splitlines() if r.strip()]


def lanes_diff(a, b):
    A, Bb = parse(a), parse(b)
    if len(A) != len(Bb) or any(x.shape != y.shape for x, y in zip(A, Bb)): return None
    return max((float(np.abs(x - y).max()) for x, y in zip(A, Bb)), default=0.0)


def image_list(name, k=1, limit=None):
    rels = [l.split()[0].lstrip('/') for l in open(D / 'list' / name) if l.strip()]
    rels = rels[::k]
    return rels[:limit] if limit else rels


def session(variant, threads):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(ONNX[variant]), so, providers=['CPUExecutionProvider'])


def export():
    import cv2
    torch, net, pre, decode = build()
    torch.set_grad_enabled(False)
    OUT.mkdir(parents=True, exist_ok=True); rep = dict(versions={}, checks={}, speed={})
    import onnx, onnxruntime as ort
    from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationDataReader
    from onnxruntime.quantization.shape_inference import quant_pre_process
    rep['versions'] = dict(torch=torch.__version__, onnx=onnx.__version__, onnxruntime=ort.__version__)
    x0 = pre(cv2.imread(str(D / image_list('val.txt', limit=1)[0])))
    torch.onnx.export(net, (x0,), str(ONNX['onnx_fp32']), opset_version=17, input_names=['img'],
                      output_names=list(OUTPUTS), do_constant_folding=True)
    onnx.checker.check_model(str(ONNX['onnx_fp32']))
    prep = OUT / 'clrernet_fp32_prep.onnx'
    # skip_optimization keeps the exported node names (v1 optimized the graph first, which renamed every node, so the
    # int8_bbneck head exclusion matched nothing and both INT8 variants were identical: results/onnx/export_v1.json)
    quant_pre_process(str(ONNX['onnx_fp32']), str(prep), skip_optimization=True)

    calib = image_list('train_gt.txt', k=max(1, sum(1 for _ in open(D / 'list/train_gt.txt')) // 300), limit=300)

    class Reader(CalibrationDataReader):
        def __init__(self): self.it = iter(calib)

        def get_next(self):
            r = next(self.it, None)
            return None if r is None else {'img': pre(cv2.imread(str(D / r))).numpy()}

    head_nodes = [n.name for n in onnx.load(str(prep)).graph.node if 'bbox_head' in n.name]
    assert head_nodes, 'no lane-head nodes found to exclude'
    for v, excl in (('int8_all', []), ('int8_bbneck', head_nodes)):
        quantize_static(str(prep), str(ONNX[v]), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, nodes_to_exclude=excl)
    rep['calibration'] = dict(source='CULane list/train_gt.txt', images=len(calib), first=calib[0], method='MinMax')
    rep['excluded_head_nodes_int8_bbneck'] = len(head_nodes)
    rep['model_mb'] = {v: round(p.stat().st_size / 2**20, 1) for v, p in ONNX.items()}

    # equivalence on VAL images: raw outputs and decoded lanes vs torch
    val = image_list('val.txt', k=97, limit=50); S = {v: session(v, 4) for v in VARIANTS}
    for v in VARIANTS:
        raw, ld, cnt = [], [], 0
        for r in val:
            x = pre(cv2.imread(str(D / r)))
            t = net(x); o = S[v].run(None, {'img': x.numpy()})
            raw.append(max(float(np.abs(a.numpy() - b).max()) for a, b in zip(t, o)))
            d = lanes_diff(lane_string(decode([a.numpy() for a in t])), lane_string(decode(o)))
            if d is None: cnt += 1
            else: ld.append(d)
        rep['checks'][v] = dict(images=len(val), raw_max_abs=float(np.max(raw)), raw_median_abs=float(np.median(raw)),
                                lane_count_or_shape_mismatch_images=cnt, lane_max_px=float(max(ld, default=0)),
                                lane_median_px=float(np.median(ld)) if ld else None)
    # writer check: torch CPU through this script's writer vs the original GPU full-test prediction files
    orig = Path(os.environ['ORIG_PRED'])
    diffs, miss = [], 0
    for r in image_list('test.txt', k=113, limit=300):
        s = lane_string(decode([a.numpy() for a in net(pre(cv2.imread(str(D / r))))]))
        d = lanes_diff(s, (orig / (r[:-3] + 'lines.txt')).read_text())
        if d is None: miss += 1
        else: diffs.append(d)
    rep['checks']['writer_torch_cpu_vs_gpu_files'] = dict(images=300, mismatched_images=miss, max_px=float(max(diffs, default=0)),
                                                         median_px=float(np.median(diffs)) if diffs else None)
    # CPU speed on VAL images: forward only, and full (preprocess + forward + decode + string)
    imgs = [cv2.imread(str(D / r)) for r in image_list('val.txt', k=47, limit=200)]
    for th in (1, 4):
        torch.set_num_threads(th); S = {v: session(v, th) for v in VARIANTS}
        runners = dict(torch_fp32=lambda x: [a.numpy() for a in net(x)], **{v: (lambda s: lambda x: s.run(None, {'img': x.numpy()}))(S[v]) for v in VARIANTS})
        for name, run in runners.items():
            for img in imgs[:10]: run(pre(img))          # warm-up
            fw, full = [], []
            for img in imgs:
                t0 = time.perf_counter(); x = pre(img); t1 = time.perf_counter(); o = run(x); t2 = time.perf_counter()
                lane_string(decode(o)); t3 = time.perf_counter()
                fw.append(1000 * (t2 - t1)); full.append(1000 * (t3 - t0))
            rep['speed'][f'{name}_threads{th}'] = dict(forward_ms_median=round(float(np.median(fw)), 1),
                                                      forward_ms_p90=round(float(np.percentile(fw, 90)), 1),
                                                      full_ms_median=round(float(np.median(full)), 1), images=len(imgs))
    import platform, subprocess
    rep['host'] = dict(node=platform.node(), cpu=subprocess.run("lscpu | grep 'Model name' | cut -d: -f2", shell=True,
                                                               capture_output=True, text=True).stdout.strip())
    (OUT / 'export.json').write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=2)); print('EXPORT DONE')


def infer(variant, i, n):
    import cv2
    torch, net, pre, decode = build()
    torch.set_grad_enabled(False); torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 4)))
    s = session(variant, int(os.environ.get('OMP_NUM_THREADS', 4)))
    rels = image_list('test.txt')[i::n]; out = OUT / variant / 'pred'; t0 = time.time()
    for k, r in enumerate(rels):
        o = s.run(None, {'img': pre(cv2.imread(str(D / r))).numpy()})
        p = out / (r[:-3] + 'lines.txt'); p.parent.mkdir(parents=True, exist_ok=True)
        txt = lane_string(decode(o)); p.write_text(txt + '\n' if txt else '')
        if k % 500 == 0: print(variant, i, k, len(rels), round(time.time() - t0), 's', flush=True)
    print('INFER DONE', variant, i, n, len(rels), round(time.time() - t0), 's')


if __name__ == '__main__':
    export() if sys.argv[1] == 'export' else infer(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
