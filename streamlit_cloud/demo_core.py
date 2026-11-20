"""Shared logic for the public demo (Streamlit Community Cloud): one-time setup, input preparation, running models.

Inference uses the same code as the benchmark (scripts/app_infer.py, LANE_DEVICE=cpu), one process per model because
the model repos have clashing module names. The CPU path was checked against the GPU path behind every reported number
(hf_space/VERIFICATION.md). Media are unlabeled: results are detections and stability, never accuracy.
"""
import json, os, shutil, subprocess, sys, time, uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                      # repository root: scripts/, config/, vendors/ (cloned), checkpoints/ (downloaded)
SCRIPTS = REPO / 'scripts'
JOBS = Path(os.environ.get('LANE_JOBS', '/tmp/lane_jobs'))
ENV_DIR = Path(os.environ.get('LANE_ENV', REPO / '.demo_env'))   # models' own Python 3.11 env (setup_demo.sh)
PY = ENV_DIR / 'bin' / 'python'
ROBUST_CKPT = REPO / 'checkpoints' / 'clrernet' / 'clrernet_culane_dla34_robust_ft.pth'
MAX_S, FPS_CAP = 10, 10
MODELS = {   # key: (label, note, checkpoint files it needs)
    'clrernet_robust': ('CLRerNet, robust fine-tune', 'Fine-tuned with rain/blur/JPEG/dark augmentation: mean F1 on corrupted '
                        'images 77.4 ± 0.1 vs 72.6, CULane test F1 81.23 ± 0.05 (3 seeds).', ['clrernet/clrernet_culane_dla34_robust_ft.pth']),
    'clrernet': ('CLRerNet (DLA34)', 'Most accurate: CULane test F1 81.55.', ['clrernet/clrernet_culane_dla34_ema.pth']),
    'ufldv2': ('UFLDv2 (ResNet34)', 'Ultra-Fast-Lane-Detection v2: CULane test F1 75.86 (no TTA).', ['ufldv2/culane_res34.pth']),
    'scnn': ('SCNN (VGG16)', 'Classic baseline: CULane test F1 70.07. Slowest on CPU.', ['scnn/vgg_SCNN_DULR_w9.pth']),
}
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXT = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm'}


def _uv():
    try:
        from uv import find_uv_bin   # the uv package from requirements.txt
        return find_uv_bin()
    except Exception:
        return shutil.which('uv') or 'uv'


def setup():
    """Build the models' Python 3.11 env, clone the model code at the pinned commits and fetch checkpoints
    (sha256-checked). Returns the usable models."""
    ok = REPO / '.demo_setup_ok'
    if not ok.exists():
        r = subprocess.run(['bash', str(HERE / 'setup_demo.sh')], cwd=REPO, capture_output=True, text=True,
                           env=dict(os.environ, LANE_ENV=str(ENV_DIR), UV_BIN=_uv()))
        (REPO / 'demo_setup.log').write_text(r.stdout + r.stderr)
        if r.returncode == 0: ok.touch()
    have = lambda f: (REPO / 'checkpoints' / f).is_file()
    return [k for k, v in MODELS.items() if all(have(f) for f in v[2]) and (REPO / 'vendors').is_dir() and PY.is_file()]


def setup_log(n=3000):
    try: return (REPO / 'demo_setup.log').read_text()[-n:]
    except OSError: return ''



def _ff():
    ff = shutil.which('ffmpeg')
    if not ff: raise RuntimeError('ffmpeg is not installed')
    return ff


def _run(cmd, timeout=900):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode: raise RuntimeError((r.stderr or r.stdout)[-600:])


def cleanup(max_age_s=3600):
    JOBS.mkdir(parents=True, exist_ok=True)
    for d in JOBS.iterdir():
        if time.time() - d.stat().st_mtime > max_age_s: shutil.rmtree(d, ignore_errors=True)


def prepare(data, name, job):
    """Write the upload as input.jpg / input.mp4. Video: first MAX_S s at FPS_CAP fps, <= 1280 px wide, no audio.
    Image: re-saved as JPEG, which also drops metadata such as GPS location."""
    ext = Path(name).suffix.lower(); raw = job / ('upload' + ext); raw.write_bytes(data)
    if ext in IMAGE_EXT:
        from PIL import Image, ImageOps
        ImageOps.exif_transpose(Image.open(raw)).convert('RGB').save(job / 'input.jpg', quality=95)
        out, kind = 'input.jpg', 'image'
    elif ext in VIDEO_EXT:
        _run([_ff(), '-loglevel', 'error', '-y', '-i', str(raw), '-t', str(MAX_S), '-vf', f"fps={FPS_CAP},scale='min(1280,iw)':-2",
              '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-an', '-map_metadata', '-1',
              str(job / 'input.mp4')])
        out, kind = 'input.mp4', 'video'
    else:
        raise RuntimeError(f'Unsupported file type {ext}')
    raw.unlink()
    return out, kind


def run_job(data, name, models, adapter='auto', tracker=True, progress=lambda frac, text: None):
    """Run the chosen models on one upload. Returns (kind, {model: dict(summary, image|video, lanes)})."""
    cleanup()
    job = JOBS / uuid.uuid4().hex[:10]; job.mkdir(parents=True)
    inp, kind = prepare(data, name, job)
    (job / 'request.json').write_text(json.dumps(dict(models=models, input=inp, kind=kind, adapter=adapter, tracker=tracker)))
    for mi, m in enumerate(models):
        env = dict(os.environ, LANE_DEVICE='cpu', PYTHONPATH=str(REPO / 'vendors' / 'clrernet'))
        if m == 'clrernet_robust': env['CLR_CKPT'] = str(ROBUST_CKPT)
        with open(job / f'log_{m}.txt', 'w') as log:
            p = subprocess.Popen([str(PY), str(SCRIPTS / 'app_infer.py'), str(job), m], cwd=SCRIPTS, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
            while p.poll() is None:
                time.sleep(1.0)
                try: s = json.loads((job / f'status_{m}.json').read_text())
                except (OSError, json.JSONDecodeError): s = {}
                tot = s.get('total') or 0; within = s.get('frame', 0) / tot if tot else 0.0
                progress(min((mi + within) / len(models), 1.0),
                         f"{MODELS[m][0]}: {s.get('stage', 'starting')}" + (f" {s.get('frame', 0)}/{tot}" if tot > 1 else ''))
        if p.returncode:
            raise RuntimeError(f'{MODELS[m][0]} failed:\n' + (job / f'log_{m}.txt').read_text()[-1200:])
    res = {}
    for m in models:
        d = job / 'out' / m; item = dict(summary=json.loads((d / 'summary.json').read_text()), lanes=d / 'lanes.jsonl')
        if kind == 'image':
            item['image'] = d / 'overlay.png'
        else:
            _run([_ff(), '-loglevel', 'error', '-y', '-i', str(d / 'overlay.mp4'), '-c:v', 'libx264', '-preset', 'veryfast',
                  '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(d / 'overlay_web.mp4')])
            item['video'] = d / 'overlay_web.mp4'
        res[m] = item
    return kind, res
