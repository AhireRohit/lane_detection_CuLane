"""Public lane-detection demo (Hugging Face Space, CPU).

Pick models, upload a dashcam image or short video, click Generate. Each model runs in its own process with the same
code as the benchmark (benchmark_culane/scripts/app_infer.py, LANE_DEVICE=cpu); the CPU path was checked against the
GPU predictions behind the reported numbers (VERIFICATION.md). One job runs at a time; uploads are deleted after an hour.
"""
import json, os, shutil, subprocess, sys, threading, time, uuid
from pathlib import Path
import streamlit as st

APP = Path(__file__).resolve().parent
SCRIPTS = APP / 'benchmark_culane' / 'scripts'
JOBS = Path('/tmp/lane_jobs'); JOBS.mkdir(exist_ok=True)
ROBUST_CKPT = APP / 'weights' / 'clrernet_culane_dla34_robust_ft.pth'
FF = shutil.which('ffmpeg')
MAX_S, FPS_CAP = 10, 10
MODELS = {
    'clrernet': dict(label='CLRerNet (DLA34)', blurb='Most accurate: CULane test F1 81.55.'),
    'clrernet_robust': dict(label='CLRerNet, robust fine-tune', blurb='Fine-tuned with rain/blur/JPEG/dark augmentation: '
                            'mean F1 on corrupted images 77.4 ± 0.1 vs 72.6, test F1 81.23 ± 0.05 (3 seeds).'),
    'ufldv2': dict(label='UFLDv2 (ResNet34)', blurb='Ultra-Fast-Lane-Detection v2: test F1 75.86 (no TTA).'),
    'scnn': dict(label='SCNN (VGG16)', blurb='Classic baseline: test F1 70.07. Slowest on CPU.'),
}
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXT = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm'}


@st.cache_resource
def gpu_lock():   # one job at a time across all visitors (free CPU hardware)
    return threading.Lock()


def cleanup(max_age_s=3600):
    for d in JOBS.iterdir():
        if time.time() - d.stat().st_mtime > max_age_s: shutil.rmtree(d, ignore_errors=True)


def run(cmd, timeout=900):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode: raise RuntimeError((r.stderr or r.stdout)[-800:])


def prepare(data, name, job):
    ext = Path(name).suffix.lower(); raw = job / ('upload' + ext); raw.write_bytes(data)
    if ext in IMAGE_EXT:
        from PIL import Image, ImageOps
        ImageOps.exif_transpose(Image.open(raw)).convert('RGB').save(job / 'input.jpg', quality=95)   # drops metadata
        out, kind = 'input.jpg', 'image'
    elif ext in VIDEO_EXT:
        run([FF, '-loglevel', 'error', '-y', '-i', str(raw), '-t', str(MAX_S), '-vf', f"fps={FPS_CAP},scale='min(1280,iw)':-2",
             '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-an', '-map_metadata', '-1',
             str(job / 'input.mp4')])
        out, kind = 'input.mp4', 'video'
    else:
        raise RuntimeError(f'unsupported file type {ext}')
    raw.unlink()
    return out, kind


st.set_page_config(page_title='Lane Detection Demo', page_icon='🛣️', layout='wide')
st.session_state.setdefault('runs', [])
cleanup()

st.title('Lane detection demo')
st.write('Pick models, upload a dashcam **image** or a short **video**, and click **Generate**. '
         'Models trained on [CULane](https://xingangpan.github.io/projects/CULane.html); code and full benchmark on GitHub.')
st.caption(f'Runs on free shared CPU, so expect ~1 s per frame (SCNN is slower). Videos are cut to {MAX_S} s at {FPS_CAP} fps. '
           'Uploads are processed only inside this Space and deleted after an hour. No labels exist for your media, so the '
           'numbers shown are detections and stability, not accuracy.')

left, right = st.columns([3, 2])
with left:
    chosen = st.multiselect('Models', list(MODELS), default=['clrernet_robust'], format_func=lambda k: MODELS[k]['label'])
    for k in chosen: st.caption(f"**{MODELS[k]['label']}**: {MODELS[k]['blurb']}")
    up = st.file_uploader('Dashcam image or video', type=sorted(e.lstrip('.') for e in IMAGE_EXT | VIDEO_EXT))
with right:
    st.subheader('Options')
    adapter = st.radio('Frame adapter', ['auto', 'squash'], horizontal=True,
                       format_func=lambda a: {'auto': 'Crop to CULane shape (recommended)', 'squash': 'Squash whole frame'}[a],
                       help='The models expect 1640×590 road views. "Crop" takes a full-width band of that shape at the '
                            'road\'s vanishing point, so the road is not distorted.')
    tracker = st.checkbox('Temporal lane tracker (videos)', value=True,
                          help='Matches lanes across frames and smooths them; held dashed lanes are drawn dashed.')

if st.button('Generate', type='primary', disabled=not (up and chosen)):
    job = JOBS / uuid.uuid4().hex[:10]; job.mkdir()
    box = st.status('Preparing…', expanded=True)
    try:
        inp, kind = prepare(up.getvalue(), up.name, job)
        (job / 'request.json').write_text(json.dumps(dict(models=chosen, input=inp, kind=kind, adapter=adapter, tracker=tracker)))
        lock = gpu_lock()
        if lock.locked(): box.update(label='Another visitor\'s job is running; waiting for it to finish…')
        with lock:
            bar = st.progress(0.0); t0 = time.time()
            for mi, m in enumerate(chosen):
                env = dict(os.environ, LANE_DEVICE='cpu', PYTHONPATH=str(APP / 'benchmark_culane' / 'vendors' / 'clrernet'))
                if m == 'clrernet_robust': env['CLR_CKPT'] = str(ROBUST_CKPT)
                log = open(job / f'log_{m}.txt', 'w')
                p = subprocess.Popen([sys.executable, str(SCRIPTS / 'app_infer.py'), str(job), m], cwd=SCRIPTS, env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
                while p.poll() is None:
                    time.sleep(1.5)
                    try: s = json.loads((job / f'status_{m}.json').read_text())
                    except (OSError, json.JSONDecodeError): s = {}
                    tot = s.get('total') or 0; within = s.get('frame', 0) / tot if tot else 0.0
                    bar.progress(min((mi + within) / len(chosen), 1.0),
                                 text=f"{MODELS[m]['label']}: {s.get('stage', 'starting')}"
                                      + (f" {s.get('frame', 0)}/{tot}" if tot > 1 else '') + f" · {int(time.time() - t0)} s")
                log.close()
                if p.returncode: raise RuntimeError(f"{MODELS[m]['label']} failed:\n" + (job / f'log_{m}.txt').read_text()[-1500:])
        res = {}
        for m in chosen:
            d = job / 'out' / m; item = dict(summary=json.loads((d / 'summary.json').read_text()), lanes=d / 'lanes.jsonl')
            if (d / 'overlay.png').is_file(): item['image'] = d / 'overlay.png'
            else:
                run([FF, '-loglevel', 'error', '-y', '-i', str(d / 'overlay.mp4'), '-c:v', 'libx264', '-preset', 'veryfast',
                     '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(d / 'overlay_web.mp4')])
                item['video'] = d / 'overlay_web.mp4'
            res[m] = item
        st.session_state.runs.insert(0, dict(name=up.name, kind=kind, results=res, adapter=adapter, tracker=tracker))
        box.update(label=f'Done in {int(time.time() - t0)} s', state='complete', expanded=False)
    except Exception as e:   # show the reason instead of a blank failure
        box.update(label='Error', state='error'); st.error(str(e))

for ri, run_ in enumerate(st.session_state.runs):
    st.divider(); res = run_['results']
    st.subheader(run_['name'])
    st.caption(f"adapter: {run_['adapter']} · tracker: {'on' if run_['tracker'] and run_['kind'] == 'video' else 'off'}")
    if len(res) > 1:
        st.dataframe([{'Model': MODELS[m]['label'], 'Frames with lanes': f"{r['summary']['frames_with_lanes']:.0%}",
                       'Mean lanes': round(r['summary']['mean_lanes'], 2), 'Lane-count changes': r['summary']['lane_count_changes'],
                       'Median ms/frame (CPU)': round(r['summary']['median_latency_ms'])} for m, r in res.items()],
                     hide_index=True, use_container_width=True)
    for tab, (m, r) in zip(st.tabs([MODELS[m]['label'] for m in res]), res.items()):
        with tab:
            sm = r['summary']
            if 'image' in r:
                st.image(str(r['image']), use_container_width=True)
                st.download_button('Download image', Path(r['image']).read_bytes(), file_name=f'{m}_lanes.png', key=f'i{ri}{m}')
            else:
                st.video(str(r['video']))
                st.download_button('Download video', Path(r['video']).read_bytes(), file_name=f'{m}_lanes.mp4', key=f'v{ri}{m}')
            c1, c2, c3 = st.columns(3)
            c1.metric('Frames', sm['frames']); c2.metric('Frames with lanes', f"{sm['frames_with_lanes']:.0%}")
            c3.metric('Mean lanes', f"{sm['mean_lanes']:.2f}")
            a = sm['adapter']
            st.caption(f"Adapter: {a['mode']}" + (f", band rows {a['band'][0]}–{a['band'][1]}" if a['mode'] == 'crop' else ''))
