#!/usr/bin/env python3
"""Baseline dashcam video inference (handbook Phase E), one model per call, NO temporal postprocessing.

Adapter (identical for all models): each video frame is resized to the CULane canvas 1640x590, passed through
the model's verified pre/forward/decode path (benchmark_speed.py builders, checked against vendor code), and lane
coordinates are scaled back to source-frame pixels. If the video aspect ratio differs from 1640/590 this resize
distorts geometry; it is recorded in report.json.
No labels exist for this video: the report contains availability/stability statistics, NOT accuracy.
Usage: run_dashcam_video.py <model> <video> <run_dir> [crop_calibration.json]
v2: with crop_calibration.json (estimate_vp.py) the adapter crops a full-width band of CULane aspect ratio around the
calibrated vanishing row instead of squashing the whole frame; lanes are only predicted inside that band.
"""
import sys, os, json, time, hashlib, socket, datetime
from pathlib import Path
import numpy as np
import cv2
import torch
from benchmark_speed import build_scnn, build_ufld, build_clr

CW, CH = 1640, 590
COLORS = [(0, 255, 0), (255, 80, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0), (0, 255, 255)]


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): h.update(c)
    return h.hexdigest()


def main():
    name, video, run = sys.argv[1], Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
    run.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available()
    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened(), f'cannot open {video}'
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'video {W}x{H} fps={fps:.3f} frames(meta)={n_meta}', flush=True)

    # v2: optional aspect-preserving crop adapter (4th arg = crop_calibration.json from estimate_vp.py).
    # Must be parsed BEFORE building the model: build_ufld replaces sys.argv for the author's config loader
    # (job 73901 bug: UFLDv2 silently ran without the crop).
    crop = json.loads(Path(sys.argv[4]).read_text()) if len(sys.argv) > 4 else None
    if name == 'scnn': pre, fwd, post, info, _ = build_scnn()
    elif name == 'ufldv2': pre, fwd, post, info, _ = build_ufld(run)
    else: pre, fwd, post, info, _ = build_clr(run)
    top, bh = (crop['band_top'], crop['band_height']) if crop else (0, H)
    sx, sy = W / CW, bh / CH

    out = cv2.VideoWriter(str(run / 'overlay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    assert out.isOpened(), 'VideoWriter failed'
    jl = open(run / 'frames.jsonl', 'w')
    lat, counts, bad = [], [], 0
    t_start = time.perf_counter()
    i = 0
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok: break
            if frame is None or frame.size == 0:
                bad += 1; i += 1; continue
            canvas = cv2.resize(frame[top:top + bh], (CW, CH), interpolation=cv2.INTER_LINEAR)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            lanes = post(fwd(pre(canvas)))
            torch.cuda.synchronize(); ms = 1000 * (time.perf_counter() - t0)
            src = [[(float(x) * sx, float(y) * sy + top) for x, y in l] for l in lanes if len(l) > 1]
            if crop:   # show the model's field of view; nothing is predicted outside the band
                cv2.line(frame, (0, top), (W - 1, top), (160, 160, 160), 1); cv2.line(frame, (0, top + bh - 1), (W - 1, top + bh - 1), (160, 160, 160), 1)
            for j, l in enumerate(src):
                cv2.polylines(frame, [np.round(np.array(l)).astype(np.int32)], False, COLORS[j % len(COLORS)], 3, cv2.LINE_AA)
            cv2.putText(frame, f'{name} baseline  frame {i}  lanes={len(src)}', (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            out.write(frame)
            jl.write(json.dumps(dict(frame=i, n_lanes=len(src), latency_ms=round(ms, 3),
                                     lanes=[[[round(x, 1), round(y, 1)] for x, y in l] for l in src])) + '\n')
            lat.append(ms); counts.append(len(src))
            if i % 200 == 0: print(f'{name} frame {i} lanes={len(src)} ms={ms:.1f}', flush=True)
            i += 1
    wall = time.perf_counter() - t_start
    cap.release(); out.release(); jl.close()
    c = np.array(counts); l = np.array(lat)
    report = {
        'model': name, 'run_id': run.name, 'utc': datetime.datetime.utcnow().isoformat() + 'Z', 'host': socket.gethostname(),
        'gpu': torch.cuda.get_device_name(0), 'variant': 'baseline (no temporal/geometry postprocessing)',
        'video_path': str(video), 'video_sha256': sha256(video),
        'video_provenance': ('UFLDv2 repo demo clip (vendors/ufldv2/example.mp4, added in upstream PR #115); original footage source undocumented'
                             if video.name == 'example.mp4' else os.environ.get('DASH_PROVENANCE', 'user-provided dashcam recording')),
        'video_width': W, 'video_height': H, 'video_fps': fps, 'video_frames_meta': n_meta,
        'video_frames_processed': int(c.size), 'unreadable_frames': bad,
        'adapter': (f'crop rows {top}-{top + bh} ({W}x{bh}) -> resize {CW}x{CH} (aspect-preserving), coords x*{sx:.4f}, y*{sy:.4f}+{top}'
                    if crop else f'resize {W}x{H} -> {CW}x{CH}, model pre/decode, scale coords by ({sx:.4f},{sy:.4f})'),
        'adapter_affine': dict(sx=sx, sy=sy, x0=0, y0=top),
        'crop_calibration': crop,
        'aspect_ratio_distortion': round((W / bh) / (CW / CH), 4),
        'model_info': info,
        'observed_detection_frame_fraction': float((c > 0).mean()),
        'zero_observed_detection_frame_fraction': float((c == 0).mean()),
        'estimated_or_held_lane_frame_fraction': 0.0,
        'lane_count_histogram': {int(k): int(v) for k, v in zip(*np.unique(c, return_counts=True))},
        'mean_lanes_per_frame': float(c.mean()),
        'lane_count_change_events': int((np.diff(c) != 0).sum()),
        'median_latency_ms': float(np.median(l)), 'p95_latency_ms': float(np.percentile(l, 95)),
        'latency_definition': 'resized canvas -> lanes (pre+forward+decode), CUDA-synchronized; first frames include warmup',
        'video_end_to_end_fps': float(c.size / wall),
        'end_to_end_definition': 'wall clock incl. video decode from disk, resize, inference, overlay drawing, mp4v encode, jsonl write',
        'warning': 'No labels: availability and visual quality are not accuracy/F1',
    }
    (run / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ('model', 'video_frames_processed', 'observed_detection_frame_fraction',
          'lane_count_change_events', 'median_latency_ms', 'video_end_to_end_fps')}, indent=2))
    assert c.size > 0 and c.size >= 0.98 * max(n_meta, 1), 'processed far fewer frames than video metadata'
    print('DASHCAM DONE', run)


if __name__ == '__main__':
    main()
