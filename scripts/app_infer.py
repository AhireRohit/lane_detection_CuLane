#!/usr/bin/env python3
"""Lane detection on one image or video for the demo apps.

Used two ways, with the same code path:
- cluster app: inside an HTCondor GPU job (scripts/app_job.sh), one process per model (each model has its own venv);
- Hugging Face Space: imported by hf_space/app.py and run on CPU (LANE_DEVICE=cpu).
Reuses the verified builders from benchmark_speed.py (pre -> forward -> decode, checked against the vendor code),
the crop adapter (vanishing-point placement, as estimate_vp.py) and the B2 temporal tracker from temporal_ablation.py.

CLI: app_infer.py <job_dir> <model>
  job_dir/request.json: {"input": "input.mp4", "kind": "video"|"image", "adapter": "auto"|"squash", "tracker": bool}
  model: scnn | ufldv2 | clrernet | clrernet_robust  (clrernet_robust = our corruption-fine-tuned checkpoint via CLR_CKPT)
  writes job_dir/out/<model>/{overlay.mp4|overlay.png, lanes.jsonl, summary.json} and job_dir/status_<model>.json.
Uploaded media are unlabeled: the summary reports detections and stability, never accuracy.
"""
import sys, os, json, time
from pathlib import Path
import numpy as np
import cv2

CW, CH = 1640, 590
CULANE_VP_REL = 0.469441598461383     # CULane val GT vanishing row / 590 (results/adapter_v2/crop_calibration.json)
DEFAULT_VP_REL = 419.1917 / 720       # fallback when too few lanes are found to place the band (example-clip value)
NAMES = {'scnn': 'SCNN (VGG16)', 'ufldv2': 'UFLDv2 (ResNet34)', 'clrernet': 'CLRerNet (DLA34)',
         'clrernet_robust': 'CLRerNet robust fine-tune'}
COLORS = [(0, 255, 0), (255, 80, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0), (0, 255, 255)]


def write_json(p, obj):
    tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, indent=2)); os.replace(tmp, p)


def build(model, work_dir):
    """Build one model with the verified benchmark builders. Returns (pre, fwd, post, info).
    Note: build_ufld rewrites sys.argv (see run_dashcam_video.py, job 73901), so parse arguments before calling."""
    import torch
    torch.set_grad_enabled(False)   # inference only (run_dashcam_video.py uses torch.no_grad() the same way)
    from benchmark_speed import build_scnn, build_ufld, build_clr
    assert model in NAMES, model
    if model == 'scnn': pre, fwd, post, info, _ = build_scnn()
    elif model == 'ufldv2': pre, fwd, post, info, _ = build_ufld(Path(work_dir))
    else: pre, fwd, post, info, _ = build_clr(Path(work_dir))
    return pre, fwd, post, info


def process(model, built, src, kind, adapter, use_tracker, out, set_status=lambda **k: None):
    """Detect lanes in one image/video with an already-built model; writes overlay, lanes.jsonl and summary.json."""
    import torch
    from estimate_vp import vrows
    from temporal_ablation import Tracker, resample, draw, ROWS
    pre, fwd, post, info = built
    cuda = torch.cuda.is_available() and os.environ.get('LANE_DEVICE', 'cuda') == 'cuda'
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    use_tracker = bool(use_tracker) and kind == 'video'

    if kind == 'image':
        img = cv2.imread(str(src)); assert img is not None, f'cannot read image {Path(src).name}'
        H, W = img.shape[:2]; fps = None; total = 1
    else:
        cap = cv2.VideoCapture(str(src)); assert cap.isOpened(), f'cannot open video {Path(src).name}'
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0; total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    def infer(frame, top, bh):
        canvas = cv2.resize(frame[top:top + bh], (CW, CH), interpolation=cv2.INTER_LINEAR)
        sync(); t0 = time.perf_counter()
        lanes = post(fwd(pre(canvas)))
        sync()
        return [l for l in lanes if len(l) > 1], 1000 * (time.perf_counter() - t0)

    # ---------- adapter: squash the whole frame, or crop a CULane-shaped band around the vanishing row ----------
    bh_crop = int(round(W * CH / CW))
    adapter_info = dict(requested=adapter)
    if adapter == 'squash' or bh_crop >= H:
        top, bh = 0, H
        adapter_info.update(mode='squash', note='whole frame resized to 1640x590' + (
            '' if adapter == 'squash' else ' (frame is already at least as wide as CULane, so no crop is possible)'))
    else:
        set_status(stage='placing crop band')
        if kind == 'image':
            samples = [img]
        else:   # up to 40 evenly spaced frames
            n = total or 1000; idx = set(np.linspace(0, max(n - 1, 0), min(40, n)).astype(int).tolist()); samples = []
            for i in range(n):
                ok, f = cap.read()
                if not ok: break
                if i in idx: samples.append(f)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        sx0, sy0 = W / CW, H / CH
        lanes_src = [[[(x * sx0, y * sy0) for x, y in l] for l in infer(f, 0, H)[0] if len(l) >= 3] for f in samples]
        ys = vrows(lanes_src, H)
        need = 1 if kind == 'image' else 5
        if len(ys) >= need:
            vp, method = float(np.median(ys)), f'median of {len(ys)} lane-pair intersections from a squash pass on {len(samples)} frame(s)'
        else:
            vp, method = DEFAULT_VP_REL * H, f'fallback: only {len(ys)} lane-pair intersections found; default vanishing row {DEFAULT_VP_REL:.3f} x height'
        top = int(np.clip(round(vp - CULANE_VP_REL * bh_crop), 0, H - bh_crop)); bh = bh_crop
        adapter_info.update(mode='crop', vanishing_row=round(vp, 1), band=[top, top + bh], method=method)
    sx, sy = W / CW, bh / CH

    # ---------- main pass ----------
    set_status(stage='detecting lanes', total=total, frame=0)
    writer = None
    if kind == 'video':
        writer = cv2.VideoWriter(str(out / 'overlay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
        assert writer.isOpened(), 'VideoWriter failed'
    tr = Tracker() if use_tracker else None
    jl = open(out / 'lanes.jsonl', 'w')
    counts, lat, i, t_start = [], [], 0, time.perf_counter()
    while True:
        if kind == 'image':
            if i: break
            frame = img.copy()
        else:
            ok, frame = cap.read()
            if not ok: break
        lanes, ms = infer(frame, top, bh)
        if adapter_info['mode'] == 'crop':   # show the model's field of view
            for yy in (top, top + bh - 1): cv2.line(frame, (0, yy), (W - 1, yy), (160, 160, 160), 1)
        if tr:
            dets = [x for x in (resample(l) for l in lanes) if x is not None]
            disp = tr.step(dets)
            draw(frame, dict(disp=disp, rej=[]), sx, sy, f'{NAMES[model]} + tracker  lanes={len(disp)}', False, y0=top)
            n_l = len(disp)
            rec = [[[round(float(x) * sx, 1), round(float(y) * sy + top, 1)] for x, y in zip(d['x'], ROWS)
                    if np.isfinite(x)] for d in disp]
        else:
            rec = [[[round(float(x) * sx, 1), round(float(y) * sy + top, 1)] for x, y in l] for l in lanes]
            for j, l in enumerate(rec):
                cv2.polylines(frame, [np.round(np.array(l)).astype(np.int32)], False, COLORS[j % len(COLORS)], 3, cv2.LINE_AA)
            cv2.putText(frame, f'{NAMES[model]}  lanes={len(rec)}', (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            n_l = len(rec)
        if writer: writer.write(frame)
        else: cv2.imwrite(str(out / 'overlay.png'), frame)
        jl.write(json.dumps(dict(frame=i, n_lanes=n_l, latency_ms=round(ms, 2), lanes=rec)) + '\n')
        counts.append(n_l); lat.append(ms); i += 1
        if i % 25 == 0: set_status(frame=i)
    wall = time.perf_counter() - t_start
    jl.close()
    if writer: writer.release(); cap.release()
    c, l = np.array(counts), np.array(lat)
    assert c.size > 0, 'no frames processed'
    summary = dict(
        model=model, model_name=NAMES[model], kind=kind, device=torch.cuda.get_device_name(0) if cuda else 'CPU',
        model_info=info,
        checkpoint=os.environ.get('CLR_CKPT', 'released author checkpoint') if model.startswith('clrernet') else 'released author checkpoint',
        input_size=[W, H], fps=fps, frames=int(c.size), adapter=adapter_info, temporal_tracker=use_tracker,
        frames_with_lanes=float((c > 0).mean()), mean_lanes=float(c.mean()),
        lane_count_histogram={int(k): int(v) for k, v in zip(*np.unique(c, return_counts=True))},
        lane_count_changes=int((np.diff(c) != 0).sum()) if c.size > 1 else 0,
        median_latency_ms=float(np.median(l[1:] if l.size > 3 else l)),   # first frame includes warmup
        processing_fps=float(c.size / wall),
        note='Uploaded media have no labels: these numbers describe detections and stability, not accuracy.')
    summary['gpu'] = summary['device']   # backwards-compatible key for the cluster app
    write_json(out / 'summary.json', summary)
    set_status(stage='done', frame=int(c.size))
    return summary


def main():
    job, model = Path(sys.argv[1]).resolve(), sys.argv[2]
    assert model in NAMES, model
    req = json.loads((job / 'request.json').read_text())
    status = job / f'status_{model}.json'
    st = dict(model=model, stage='loading model', frame=0, total=None, t=time.time())
    write_json(status, st)

    def set_status(**k):
        st.update(k, t=time.time()); write_json(status, st)

    # Parse everything before building the model: build_ufld rewrites sys.argv.
    src = job / req['input']; out = job / 'out' / model
    built = build(model, out)
    summary = process(model, built, src, req['kind'], req.get('adapter', 'auto'), req.get('tracker'), out, set_status)
    print('APP_INFER DONE', model, json.dumps({k: summary[k] for k in ('frames', 'mean_lanes', 'median_latency_ms')}))


if __name__ == '__main__':
    main()
