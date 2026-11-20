#!/usr/bin/env bash
# CPU-only job (python:3.10, which has git and curl like Streamlit Cloud): fresh venv from streamlit_cloud/requirements.txt, a clean copy of the repo layout,
# the real one-time setup (clone pinned model code, download + sha256-check every checkpoint), then demo_core.run_job on
# an image (all four models) and a short video (tracker on). Mirrors what Streamlit Community Cloud will do.
# Usage: test_streamlit_demo.sh <image> <video>
set -uo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)"; IMG="$1"; VID="$2"
T=$(mktemp -d /tmp/stdemo.XXXX); OUT=$B/results/verify_cpu/streamlit_demo_test; mkdir -p $OUT
python3 -m venv $T/venv && $T/venv/bin/pip install -q --upgrade pip && $T/venv/bin/pip install -q -r $B/streamlit_cloud/requirements.txt || exit 2
# No libGL in this container (Streamlit Cloud gets libgl1 from packages.txt): let the headless build provide cv2 here.
$T/venv/bin/pip install -q --no-deps --force-reinstall opencv-python-headless==4.10.0.84
$T/venv/bin/pip install -q imageio-ffmpeg && mkdir -p $T/bin && ln -s "$($T/venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" $T/bin/ffmpeg
$T/venv/bin/pip freeze > $OUT/pip_freeze.txt
export PATH=$T/venv/bin:$T/bin:$PATH
mkdir -p $T/repo && cp -r $B/scripts $B/config $B/streamlit_cloud $T/repo/
cd $T/repo/streamlit_cloud && python - "$IMG" "$VID" "$OUT" <<'PY'
import sys, json, time, shutil
img, vid, out = sys.argv[1:4]
import demo_core as core
t = time.time(); avail = core.setup()
print('setup', round(time.time() - t), 's; available:', avail); print((core.REPO / 'demo_setup.log').read_text()[-1500:])
assert avail, 'no models available'
for name, f, ms in (('image', img, avail), ('video', vid, [m for m in avail if m.startswith('clrernet')][:2])):
    t = time.time(); kind, res = core.run_job(open(f, 'rb').read(), f.split('/')[-1], ms, 'auto', True)
    for m, r in res.items():
        src = r.get('image') or r.get('video'); shutil.copy(src, f"{out}/{name}_{m}_{src.name}")
        s = r['summary']; print(name, m, s['frames'], round(s['mean_lanes'], 2), s['adapter'].get('band'), round(s['median_latency_ms']), 'ms')
    print(name, 'OK', round(time.time() - t), 's')
print('STREAMLIT DEMO TEST OK')
PY
