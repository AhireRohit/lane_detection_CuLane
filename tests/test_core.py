"""Unit tests for the evaluation and post-processing code (no dataset, no GPU, no checkpoints needed).

Run: pytest -q tests
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

S = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(S))


def load(name):
    spec = importlib.util.spec_from_file_location(name, S / f'{name}.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


te = load('tusimple_eval')
ta = load('temporal_ablation')

YS = list(range(160, 720, 10))


def lane(x0, slope):
    return [x0 + slope * (y - 400) if y >= 260 else -2 for y in YS]


LABELS = [te.truncate_gt(dict(raw_file=f'clip{i}/20.jpg', h_samples=YS,
                              lanes=[lane(400 + 10 * i, -0.8), lane(640, 0.0), lane(880 - 10 * i, 0.8)]))
          for i in range(12)]


# ---------------- TuSimple metric ----------------
def test_tusimple_perfect_and_empty():
    perfect = {l['raw_file']: l['lanes'] for l in LABELS}
    r = te.score(perfect, LABELS)
    assert (r['accuracy'], r['fp_rate'], r['fn_rate'], r['f1']) == (1.0, 0.0, 0.0, 1.0)
    r = te.score({l['raw_file']: [] for l in LABELS}, LABELS)
    assert r['fn_rate'] == 1.0 and r['f1'] == 0.0


def test_tusimple_threshold_and_false_positive():
    near = {l['raw_file']: [[x + 15 if x >= 0 else x for x in ln] for ln in l['lanes']] for l in LABELS}
    assert te.score(near, LABELS)['f1'] == 1.0                     # inside the 20 px threshold
    far = {l['raw_file']: [[x + 60 if x >= 0 else x for x in ln] for ln in l['lanes']] for l in LABELS}
    assert te.score(far, LABELS)['fn_rate'] == 1.0                 # far outside it
    extra = {l['raw_file']: l['lanes'] + [lane(100, -1.5)] for l in LABELS}
    r = te.score(extra, LABELS)
    assert r['fn_rate'] == 0.0 and r['fp_rate'] == pytest.approx(0.25)


def test_tusimple_rows_above_cut_are_ignored():
    lab = te.truncate_gt(dict(raw_file='a', h_samples=YS, lanes=[[500.0] * len(YS)]))
    assert all(x == -2 for x, y in zip(lab['lanes'][0], YS) if y < te.Y_MIN)
    pts = [(500.0, float(y)) for y in range(200, 720, 5)]
    s = te.to_samples(pts, YS)
    assert all(x == -2 for x, y in zip(s, YS) if y < te.Y_MIN) and all(x == 500.0 for x, y in zip(s, YS) if y >= te.Y_MIN)


# ---------------- B2 temporal tracker ----------------
def det(x):
    return np.full(len(ta.ROWS), float(x))


def test_tracker_confirms_after_two_hits_and_smooths():
    tr = ta.Tracker()
    assert tr.step([det(800)]) == []                               # first sighting: not shown yet
    out = tr.step([det(810)])
    assert len(out) == 1 and not out[0]['held']
    assert np.allclose(out[0]['x'], 0.5 * 810 + 0.5 * 800)          # EMA, alpha 0.5


def test_tracker_holds_at_most_three_frames():
    tr = ta.Tracker(); tr.step([det(800)]); tr.step([det(800)])
    for _ in range(ta.P['max_hold']):
        out = tr.step([])
        assert len(out) == 1 and out[0]['held']
    assert tr.step([]) == []


def test_tracker_gate_starts_new_track():
    tr = ta.Tracker(); tr.step([det(800)]); tr.step([det(800)])
    out = tr.step([det(800 + ta.P['gate_px'] + 5)])                 # beyond the gate: old track held, new one unconfirmed
    assert len(out) == 1 and out[0]['held']


# ---------------- CPU port of CLRerNet's lane NMS ----------------
def proposal(x, start=0.2, length=40.0):
    p = np.zeros(77, np.float32); p[2] = start; p[3] = 0.5; p[4] = length; p[5:] = x
    return p


def test_nms_cpu_suppresses_duplicates_and_keeps_distinct():
    torch = pytest.importorskip('torch')
    nms_cpu = load('nms_cpu')
    boxes = torch.tensor(np.stack([proposal(300), proposal(302), proposal(600), proposal(900)]))
    scores = torch.tensor([0.9, 0.8, 0.7, 0.95])
    keep, n, _ = nms_cpu.nms(boxes, scores, overlap=50, top_k=4)
    assert sorted(keep[:int(n)].tolist()) == [0, 2, 3]              # box 1 duplicates box 0 (lower score)
    keep, n, _ = nms_cpu.nms(boxes, scores, overlap=50, top_k=2)
    assert keep[:int(n)].tolist() == [3, 0]                         # top_k, in score order


# ---------------- CULane lane files used by the labeled-video tracker test ----------------
def test_lane_file_round_trip(tmp_path):
    tc = load('tracker_culane')
    x = np.where(ta.ROWS >= 300, 700.0 + (ta.ROWS - 300) * 0.5, np.nan)
    p = tmp_path / 'a.lines.txt'; tc.write_lanes(p, [x])
    back = tc.read_lanes(p)
    assert len(back) == 1 and np.allclose(back[0][np.isfinite(x)], x[np.isfinite(x)], atol=0.01)
