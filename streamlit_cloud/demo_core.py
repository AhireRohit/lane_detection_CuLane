"""Shared logic for the public demo (Streamlit Community Cloud): one-time setup, input preparation, running models.

Inference uses the same code as the benchmark (scripts/app_infer.py, LANE_DEVICE=cpu). Each model runs in its own
persistent worker process (the model repos have clashing module names), loaded once and reused, so repeat runs skip
model loading. At most MAX_WORKERS stay loaded (least recently used is stopped) to fit the free tier's memory.
The CPU path was checked against the GPU path behind every reported number (hf_space/VERIFICATION.md).
Media are unlabeled: results are detections and stability, never accuracy.
"""
import json, os, shutil, subprocess, threading, time, uuid
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
MAX_WORKERS = int(os.environ.get('LANE_MAX_WORKERS', 2))
SAMPLE_SRC = REPO / 'vendors' / 'ufldv2' / 'example.mp4'   # ships with the pinned UFLDv2 repo (MIT); fetched, not redistributed
SAMPLES = {'sample_clip.mp4': 'Sample video (highway, 10 s)', 'sample_frame.jpg': 'Sample image (one frame of it)'}
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



def sample(name):
    """Bytes of a built-in sample, cut from the example clip of the UFLDv2 repository at first use."""
    out = REPO / 'samples' / name
    if not out.is_file():
        out.parent.mkdir(exist_ok=True)
        if name == 'sample_clip.mp4':
            _run([_ff(), '-loglevel', 'error', '-y', '-ss', '20', '-i', str(SAMPLE_SRC), '-t', str(MAX_S), '-an',
                  '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', str(out)])
        else:
            _run([_ff(), '-loglevel', 'error', '-y', '-ss', '25', '-i', str(SAMPLE_SRC), '-frames:v', '1', '-q:v', '2', str(out)])
    return out.read_bytes()


class _Worker:
    """One persistent model process: scripts/app_infer.py --serve <model> (job directories on stdin)."""

    def __init__(self, model):
        self.model, self.p, self.used, self.log = model, None, 0.0, None

    def alive(self):
        return self.p is not None and self.p.poll() is None

    def start(self):
        env = dict(os.environ, LANE_DEVICE='cpu', PYTHONPATH=str(REPO / 'vendors' / 'clrernet'))
        if self.model == 'clrernet_robust': env['CLR_CKPT'] = str(ROBUST_CKPT)
        wd = JOBS.parent / 'lane_workers' / self.model; wd.mkdir(parents=True, exist_ok=True)
        self.log = wd / 'worker.log'
        self.p = subprocess.Popen([str(PY), str(SCRIPTS / 'app_infer.py'), '--serve', self.model, str(wd)], cwd=SCRIPTS,
                                  env=env, stdin=subprocess.PIPE, stdout=open(self.log, 'w'), stderr=subprocess.STDOUT,
                                  text=True)

    def stop(self):
        if self.alive():
            self.p.terminate()
            try: self.p.wait(10)
            except subprocess.TimeoutExpired: self.p.kill()
        self.p = None

    def submit(self, job):
        if not self.alive(): self.start()
        self.used = time.time(); self.p.stdin.write(str(job) + '\n'); self.p.stdin.flush()

    def log_tail(self, n=1200):
        try: return self.log.read_text()[-n:]
        except (OSError, AttributeError): return ''


_workers, _wlock = {}, threading.Lock()


def _worker(model):
    with _wlock:
        w = _workers.setdefault(model, _Worker(model))
        loaded = sorted((x for x in _workers.values() if x.alive() and x is not w), key=lambda x: x.used)
        for x in loaded[:max(0, len(loaded) - (MAX_WORKERS - 1))]: x.stop()   # make room, least recently used first
        return w


def loaded_models():
    return [m for m, w in _workers.items() if w.alive()]


def _grid(paths, out):
    """Side-by-side comparison image: 2 columns, each overlay scaled to 820 px wide."""
    from PIL import Image
    ims = [Image.open(p).convert('RGB') for p in paths]
    w = 820; ims = [im.resize((w, round(im.height * w / im.width))) for im in ims]
    h = max(im.height for im in ims); cols = 1 if len(ims) == 1 else 2; rows = (len(ims) + cols - 1) // cols
    g = Image.new('RGB', (cols * w, rows * h), (14, 17, 23))
    for i, im in enumerate(ims): g.paste(im, ((i % cols) * w, (i // cols) * h))
    g.save(out); return out


def _stack(paths, out):
    """Side-by-side comparison video: 2 columns of 640 px wide overlays (a dark tile fills an odd slot)."""
    n = len(paths); ins = sum((['-i', str(p)] for p in paths), [])
    f = ''.join(f'[{i}:v]scale=640:-2,setsar=1[v{i}];' for i in range(n))
    if n == 2:
        f += '[v0][v1]hstack=inputs=2[o]'
    else:
        if n == 3: f = f.replace('[v0];', '[v0a];[v0a]split[v0][v0b];[v0b]drawbox=c=black@1:t=fill[v3];', 1)
        f += '[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[o]'
    _run([_ff(), '-loglevel', 'error', '-y', *ins, '-filter_complex', f, '-map', '[o]', '-c:v', 'libx264', '-preset',
          'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-shortest', str(out)])
    return out


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
    """Run the chosen models on one upload. Returns (kind, {model: dict(summary, image|video, lanes)}, comparison path
    or None). Each model's persistent worker starts on first use and stays loaded for the next request."""
    cleanup()
    job = JOBS / uuid.uuid4().hex[:10]; job.mkdir(parents=True)
    inp, kind = prepare(data, name, job)
    (job / 'request.json').write_text(json.dumps(dict(models=models, input=inp, kind=kind, adapter=adapter, tracker=tracker)))
    for mi, m in enumerate(models):
        w = _worker(m)
        status, done = job / f'status_{m}.json', job / f'done_{m}.json'
        status.write_text(json.dumps(dict(stage='starting' if w.alive() else 'loading model (first use)', frame=0, total=None)))
        w.submit(job)
        while not done.exists():
            time.sleep(0.5)
            if not w.alive() and not done.exists():
                raise RuntimeError(f'{MODELS[m][0]} stopped unexpectedly:\n' + w.log_tail())
            try: s = json.loads(status.read_text())
            except (OSError, json.JSONDecodeError): s = {}
            tot = s.get('total') or 0; within = s.get('frame', 0) / tot if tot else 0.0
            progress(min((mi + within) / len(models), 1.0),
                     f"{MODELS[m][0]}: {s.get('stage', 'starting')}" + (f" {s.get('frame', 0)}/{tot}" if tot > 1 else ''))
        r = json.loads(done.read_text())
        if not r.get('ok'):
            raise RuntimeError(f'{MODELS[m][0]} failed:\n' + r.get('error', ''))
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
    if len(models) > 1:
        progress(1.0, 'Building the side-by-side view')
        comp = (_grid([res[m]['image'] for m in models], job / 'compare.png') if kind == 'image'
                else _stack([res[m]['video'] for m in models], job / 'compare.mp4'))
        return kind, res, comp
    return kind, res, None
