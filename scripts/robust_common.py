"""Shared, torch-free definitions for the CULane corruption-robustness benchmark.

Subset: every STRIDE-th entry (offset 0) of each of the nine official test_split scenario lists, i.e. a
deterministic, scenario-proportional ~1/12 sample of the 34,680-image test set (crossroad included: FP only).
Corruptions are deterministic per (image index, condition, severity), so every model sees byte-identical inputs.
"""
from pathlib import Path
import numpy as np
import cv2

B = Path(__file__).resolve().parents[1]
D = B.parent / 'datasets' / 'CULane'
STRIDE = 12
SCEN = ['test0_normal', 'test1_crowd', 'test2_hlight', 'test3_shadow', 'test4_noline',
        'test5_arrow', 'test6_curve', 'test7_cross', 'test8_night']
CORR = ['dark', 'motion_blur', 'jpeg', 'shadow', 'rain']
CONDS = [('clean', 0)] + [(c, s) for c in CORR for s in (1, 2, 3)]
SEVERITY = dict(dark='brightness x0.6 / x0.4 / x0.25', motion_blur='linear kernel 7 / 15 / 25 px, angle U(-45,45) deg',
                jpeg='JPEG quality 30 / 15 / 5', shadow='1 / 2 / 3 soft polygon shadows on road, factor 0.6 / 0.5 / 0.4',
                rain='300 / 700 / 1500 streaks (15-35 px, common slant), image x0.85')


def subset():
    """[(scenario, list_entry_as_written, rel_path_without_leading_slash)]"""
    out = []
    for s in SCEN:
        lines = [l.split()[0] for l in open(D / 'list/test_split' / f'{s}.txt') if l.strip()]
        out += [(s, e, e.lstrip('/')) for e in lines[::STRIDE]]
    return out


def cond_name(c, s):
    return c if c == 'clean' else f'{c}_s{s}'


def corrupt(img, cond, sev, idx):
    if cond == 'clean':
        return img
    rng = np.random.default_rng(idx * 1000 + CORR.index(cond) * 10 + sev)
    H, W = img.shape[:2]
    if cond == 'dark':
        return np.clip(img.astype(np.float32) * [0.6, 0.4, 0.25][sev - 1], 0, 255).astype(np.uint8)
    if cond == 'motion_blur':
        L = [7, 15, 25][sev - 1]
        k = np.zeros((L, L), np.float32); k[L // 2, :] = 1
        M = cv2.getRotationMatrix2D(((L - 1) / 2, (L - 1) / 2), float(rng.uniform(-45, 45)), 1.0)
        k = cv2.warpAffine(k, M, (L, L)); k /= k.sum()
        return cv2.filter2D(img, -1, k)
    if cond == 'jpeg':
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, [30, 15, 5][sev - 1]])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if cond == 'shadow':
        n, a = [(1, 0.6), (2, 0.5), (3, 0.4)][sev - 1]
        m = np.ones((H, W), np.float32)
        for _ in range(n):
            cx, w = rng.uniform(0, W), rng.uniform(0.15, 0.35) * W
            top, skew = rng.uniform(0.45, 0.7) * H, rng.uniform(-0.3, 0.3) * W
            poly = np.array([[cx - w / 2 + skew, top], [cx + w / 2 + skew, top], [cx + w / 2, H], [cx - w / 2, H]], np.int32)
            cv2.fillPoly(m, [poly], a)
        m = cv2.GaussianBlur(m, (31, 31), 0)
        return np.clip(img.astype(np.float32) * m[..., None], 0, 255).astype(np.uint8)
    if cond == 'rain':
        n = [300, 700, 1500][sev - 1]
        layer = np.zeros((H, W), np.uint8); slant = float(rng.uniform(-8, 8))
        for x, y, l in zip(rng.integers(0, W, n), rng.integers(0, H, n), rng.integers(15, 35, n)):
            cv2.line(layer, (int(x), int(y)), (int(x + slant), int(y + l)), 200, 1)
        layer = cv2.blur(layer, (3, 3))
        return np.clip(img.astype(np.float32) * 0.85 + layer[..., None].astype(np.float32) * 0.6, 0, 255).astype(np.uint8)
    raise ValueError(cond)
