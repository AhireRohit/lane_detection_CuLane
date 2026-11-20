#!/usr/bin/env bash
# CPU-only job (python:3.10-slim): assemble the Space with hf_space/make_space.sh, install its environment the way
# Hugging Face does (gradio from sdk_version first, then requirements.txt), and call the app's generate() on an image
# (all four models) and a short video (tracker on). Vendors/checkpoints are linked from the cluster instead of downloaded.
# Usage: test_space_app.sh <image> <video>
set -uo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)"; IMG="$1"; VID="$2"
V=$B/.venvs/space_cpu; S=$(mktemp -d /tmp/space.XXXX); OUT=$B/results/verify_cpu/space_app_test; mkdir -p $OUT
GRADIO=$(sed -n 's/^sdk_version: *//p' $B/hf_space/SPACE_README.md)
$V/bin/pip install -q "gradio==$GRADIO" imageio-ffmpeg && $V/bin/pip install -q -r $B/hf_space/requirements.txt || exit 2
# This container has no libGL (the Space gets libgl1 from packages.txt): make the headless build of the same
# OpenCV version provide cv2 here.
$V/bin/pip install -q --no-deps --force-reinstall opencv-python-headless==4.10.0.84
$V/bin/pip freeze > $OUT/pip_freeze_with_gradio.txt
mkdir -p $S/bin && ln -s "$($V/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" $S/bin/ffmpeg   # test only (Space: packages.txt)
export PATH=$S/bin:$PATH
bash $B/hf_space/make_space.sh $S/app $B/results/export/clrernet_culane_dla34_robust_ft.pth || exit 3
ln -s $B/vendors $S/app/benchmark_culane/vendors; ln -s $B/checkpoints $S/app/benchmark_culane/checkpoints; touch $S/app/.setup_ok
cd $S/app && $V/bin/python - "$IMG" "$VID" "$OUT" <<'PY'
import sys, json, shutil, time
sys.argv = [sys.argv[0]] + sys.argv[1:]
img, vid, out = sys.argv[1:4]
import app
prog = lambda *a, **k: None
labels = [v[0] for v in app.MODELS.values()]
for name, f, labs in (('image', img, labels), ('video', vid, labels[:2])):
    t = time.time()
    res = app.generate(f, labs, 'Crop to CULane shape (recommended)', True, progress=prog)
    gallery, vids, table, msg = res[0], res[1:5], res[5], res[6]
    files = [g[0] for g in (gallery.get('value') or [])] + [v.get('value') for v in vids if v.get('value')]
    for p in files: shutil.copy(p, f"{out}/{name}_{len(list(__import__('pathlib').Path(out).iterdir()))}_{p.split('/')[-2]}_{p.split('/')[-1]}")
    print(name, 'OK', round(time.time() - t), 's', msg); print(json.dumps(table, default=str))
print('SPACE APP TEST OK')
PY
