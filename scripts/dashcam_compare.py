#!/usr/bin/env python3
"""Three-model visual comparison from baseline overlay videos (same frames for all models).
Usage: dashcam_compare.py <out_dir> <scnn_run> <ufldv2_run> <clrernet_run>
Writes three_model_contact_sheet.jpg (6 evenly spaced frames x 3 models), side_by_side.mp4, summary.json."""
import sys, json
from pathlib import Path
import numpy as np
import cv2

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
runs = [Path(p) for p in sys.argv[2:5]]
names = ['scnn', 'ufldv2', 'clrernet']
caps = [cv2.VideoCapture(str(r / 'overlay.mp4')) for r in runs]
n = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps)
fps = caps[0].get(cv2.CAP_PROP_FPS) or 30.0
TW = 640

def small(f):
    h, w = f.shape[:2]
    return cv2.resize(f, (TW, int(round(h * TW / w))), interpolation=cv2.INTER_AREA)

picks = set(np.linspace(0, n - 1, 6).astype(int).tolist())
sheet_rows, vw = [], None
for i in range(n):
    frames = []
    for c in caps:
        ok, f = c.read()
        assert ok, f'read failed at frame {i}'
        frames.append(small(f))
    row = np.hstack(frames)
    if vw is None:
        vw = cv2.VideoWriter(str(out / 'side_by_side.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (row.shape[1], row.shape[0]))
    vw.write(row)
    if i in picks: sheet_rows.append(row)
vw.release()
header = np.full((40, sheet_rows[0].shape[1], 3), 255, np.uint8)
for k, nm in enumerate(names):
    cv2.putText(header, f'{nm} (baseline)', (k * TW + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
cv2.imwrite(str(out / 'three_model_contact_sheet.jpg'), np.vstack([header] + sheet_rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
summ = {nm: {k: json.load(open(r / 'report.json'))[k] for k in (
    'video_frames_processed', 'observed_detection_frame_fraction', 'mean_lanes_per_frame', 'lane_count_histogram',
    'lane_count_change_events', 'median_latency_ms', 'p95_latency_ms', 'video_end_to_end_fps', 'run_id')}
    for nm, r in zip(names, runs)}
summ['contact_sheet_frames'] = sorted(picks)
summ['warning'] = 'No labels: availability and visual quality are not accuracy/F1'
(out / 'summary.json').write_text(json.dumps(summ, indent=2))
print(json.dumps(summ, indent=2)); print('COMPARE DONE', out)
