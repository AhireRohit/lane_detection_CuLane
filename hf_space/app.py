"""Public lane-detection demo (Hugging Face Space, Gradio SDK, CPU).

Pick models, upload a dashcam image or short video, click Generate. Each model runs in its own process with the same
code as the benchmark (benchmark_culane/scripts/app_infer.py, LANE_DEVICE=cpu); the CPU path was checked against the
GPU path behind the reported numbers (VERIFICATION.md). Gradio's queue runs one job at a time. On first start the
model code (pinned commits) and the authors' checkpoints are fetched and checked by sha256 (setup_space.sh).
"""
import json, os, shutil, subprocess, sys, threading, time, uuid
from pathlib import Path
import gradio as gr

APP = Path(__file__).resolve().parent
SCRIPTS = APP / 'benchmark_culane' / 'scripts'
JOBS = Path('/tmp/lane_jobs'); JOBS.mkdir(exist_ok=True)
ROBUST_CKPT = APP / 'weights' / 'clrernet_culane_dla34_robust_ft.pth'
FF = shutil.which('ffmpeg')
MAX_S, FPS_CAP = 10, 10
MODELS = {   # key: (label, note)
    'clrernet_robust': ('CLRerNet, robust fine-tune', 'fine-tuned with rain/blur/JPEG/dark augmentation; mean F1 on corrupted images 77.4 ± 0.1 vs 72.6'),
    'clrernet': ('CLRerNet (DLA34)', 'CULane test F1 81.55'),
    'ufldv2': ('UFLDv2 (ResNet34)', 'CULane test F1 75.86 (no TTA)'),
    'scnn': ('SCNN (VGG16)', 'CULane test F1 70.07; slowest on CPU (~2 s per frame)'),
}
LABEL2KEY = {v[0]: k for k, v in MODELS.items()}
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXT = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm'}

# ---------- one-time setup (model code + checkpoints), in the background so the page loads immediately ----------
SETUP = dict(done=False, error=None, log=APP / 'setup.log')


def _setup():
    if (APP / '.setup_ok').exists():
        SETUP['done'] = True; return
    with open(SETUP['log'], 'w') as log:
        r = subprocess.run(['bash', str(APP / 'setup_space.sh')], cwd=APP, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode == 0:
        (APP / '.setup_ok').touch(); SETUP['done'] = True
    else:
        SETUP['error'] = SETUP['log'].read_text()[-1500:]


threading.Thread(target=_setup, daemon=True).start()


def cleanup(max_age_s=3600):
    for d in JOBS.iterdir():
        if time.time() - d.stat().st_mtime > max_age_s: shutil.rmtree(d, ignore_errors=True)


def run(cmd, timeout=900):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode: raise gr.Error((r.stderr or r.stdout)[-500:])


def prepare(path, job):
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXT:
        from PIL import Image, ImageOps
        ImageOps.exif_transpose(Image.open(path)).convert('RGB').save(job / 'input.jpg', quality=95)   # drops metadata
        return 'input.jpg', 'image'
    if ext in VIDEO_EXT:
        run([FF, '-loglevel', 'error', '-y', '-i', str(path), '-t', str(MAX_S), '-vf', f"fps={FPS_CAP},scale='min(1280,iw)':-2",
             '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-an', '-map_metadata', '-1',
             str(job / 'input.mp4')])
        return 'input.mp4', 'video'
    raise gr.Error(f'Unsupported file type {ext}. Use an image (jpg, png, ...) or a video (mp4, mov, ...).')


def generate(file, labels, adapter, tracker, progress=gr.Progress()):
    if not file: raise gr.Error('Upload an image or a video first.')
    if not labels: raise gr.Error('Pick at least one model.')
    while not SETUP['done']:
        if SETUP['error']: raise gr.Error('Model setup failed:\n' + SETUP['error'])
        progress(0, desc='First start: downloading model code and checkpoints (a few minutes, once)…'); time.sleep(3)
    cleanup()
    chosen = [LABEL2KEY[l] for l in labels]
    job = JOBS / uuid.uuid4().hex[:10]; job.mkdir()
    inp, kind = prepare(file, job)
    (job / 'request.json').write_text(json.dumps(dict(models=chosen, input=inp, kind=kind,
                                                      adapter='auto' if adapter.startswith('Crop') else 'squash', tracker=tracker)))
    t0 = time.time()
    for mi, m in enumerate(chosen):
        env = dict(os.environ, LANE_DEVICE='cpu', PYTHONPATH=str(APP / 'benchmark_culane' / 'vendors' / 'clrernet'))
        if m == 'clrernet_robust': env['CLR_CKPT'] = str(ROBUST_CKPT)
        with open(job / f'log_{m}.txt', 'w') as log:
            p = subprocess.Popen([sys.executable, str(SCRIPTS / 'app_infer.py'), str(job), m], cwd=SCRIPTS, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
            while p.poll() is None:
                time.sleep(1.0)
                try: s = json.loads((job / f'status_{m}.json').read_text())
                except (OSError, json.JSONDecodeError): s = {}
                tot = s.get('total') or 0; within = s.get('frame', 0) / tot if tot else 0.0
                progress((mi + within) / len(chosen), desc=f"{MODELS[m][0]}: {s.get('stage', 'starting')}"
                         + (f" {s.get('frame', 0)}/{tot}" if tot > 1 else ''))
        if p.returncode: raise gr.Error(f'{MODELS[m][0]} failed: ' + (job / f'log_{m}.txt').read_text()[-600:])

    images, videos, rows = [], [], []
    for m in chosen:
        d = job / 'out' / m; sm = json.loads((d / 'summary.json').read_text())
        if kind == 'image':
            images.append((str(d / 'overlay.png'), MODELS[m][0]))
        else:
            run([FF, '-loglevel', 'error', '-y', '-i', str(d / 'overlay.mp4'), '-c:v', 'libx264', '-preset', 'veryfast',
                 '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(d / 'overlay_web.mp4')])
            videos.append(str(d / 'overlay_web.mp4'))
        a = sm['adapter']
        rows.append([MODELS[m][0], sm['frames'], f"{sm['frames_with_lanes']:.0%}", round(sm['mean_lanes'], 2),
                     sm['lane_count_changes'], round(sm['median_latency_ms']),
                     f"crop rows {a['band'][0]}–{a['band'][1]}" if a['mode'] == 'crop' else 'squash'])
    vids = [gr.update(value=videos[i] if i < len(videos) else None, visible=i < len(videos),
                      label=MODELS[chosen[i]][0] if i < len(videos) else '') for i in range(4)]
    msg = f'Done in {int(time.time() - t0)} s. No labels exist for your media: these are detections, not accuracy.'
    return [gr.update(value=images, visible=kind == 'image'), *vids, rows, msg]


with gr.Blocks(title='Lane Detection Demo', theme=gr.themes.Soft()) as demo:
    gr.Markdown('# Lane detection demo\nPick models, upload a dashcam **image** or a short **video**, and click **Generate**. '
                'Models trained on [CULane](https://xingangpan.github.io/projects/CULane.html). '
                '[Code, benchmark and report on GitHub](https://github.com/AhireRohit/lane_detection_CuLane).')
    with gr.Row():
        with gr.Column(scale=3):
            models = gr.CheckboxGroup([v[0] for v in MODELS.values()], value=[MODELS['clrernet_robust'][0]], label='Models',
                                      info=' · '.join(f'{v[0]}: {v[1]}' for v in MODELS.values()))
            file = gr.File(label='Dashcam image or video', type='filepath',
                           file_types=sorted(IMAGE_EXT | VIDEO_EXT))
        with gr.Column(scale=2):
            adapter = gr.Radio(['Crop to CULane shape (recommended)', 'Squash whole frame'], value='Crop to CULane shape (recommended)',
                               label='Frame adapter', info='The models expect 1640×590 road views; "crop" takes a full-width band '
                               'of that shape at the road\'s vanishing point, so the road is not distorted.')
            tracker = gr.Checkbox(value=True, label='Temporal lane tracker (videos)',
                                  info='Matches lanes across frames and smooths them; held dashed lanes are drawn dashed.')
            btn = gr.Button('Generate', variant='primary')
    gr.Markdown(f'Runs on free shared CPU: about 0.5 s per frame for CLRerNet, 2 s for SCNN. Videos are cut to {MAX_S} s at '
                f'{FPS_CAP} fps. Uploads are processed only inside this Space and deleted after an hour.')
    status = gr.Markdown()
    gallery = gr.Gallery(label='Detected lanes', columns=2, height='auto', visible=False)
    with gr.Row():
        v = [gr.Video(visible=False) for _ in range(2)]
    with gr.Row():
        v += [gr.Video(visible=False) for _ in range(2)]
    table = gr.Dataframe(headers=['Model', 'Frames', 'Frames with lanes', 'Mean lanes', 'Lane-count changes',
                                  'Median ms/frame (CPU)', 'Adapter'], label='Per-model statistics', interactive=False)
    btn.click(generate, [file, models, adapter, tracker], [gallery, *v, table, status], concurrency_limit=1)

if __name__ == '__main__':
    demo.queue(max_size=20).launch()
