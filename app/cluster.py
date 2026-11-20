"""Backend for the Streamlit demo: upload media to the cluster, run the GPU job through HTCondor, fetch the results.

All model inference runs on a cluster GPU via jobs/app/infer.sub -> scripts/app_job.sh -> scripts/app_infer.py,
using the same verified model code as the benchmark. This machine only needs ssh/scp (key-based login) and,
optionally, ffmpeg (to trim and re-encode uploads and to make the result video playable in the browser).
"""
import json, os, re, shutil, subprocess, time, uuid
from pathlib import Path

APP = Path(__file__).resolve().parent
RUNS = APP / 'runs'
SSH_OPTS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20']
VIDEO_EXT = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm'}
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


class ClusterError(RuntimeError):
    pass


def load_settings():
    """host = user@login-node, root = remote .../Lane_Detection/benchmark_culane. Env vars override app/settings.json."""
    s = {}
    f = APP / 'settings.json'
    if f.is_file():
        s = json.loads(f.read_text())
    return dict(host=os.environ.get('LANE_APP_HOST', s.get('host', '')),
                root=os.environ.get('LANE_APP_ROOT', s.get('root', '')))


def _run(args, timeout=120):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ClusterError(f'timed out: {args[0]}')
    if r.returncode != 0:
        raise ClusterError((r.stderr or r.stdout).strip()[-600:] or f'{args[0]} failed')
    return r.stdout


def ssh(host, cmd, timeout=60):
    return _run(['ssh', *SSH_OPTS, host, cmd], timeout)


def scp_up(host, local, remote, timeout=600):
    _run(['scp', '-q', *SSH_OPTS, str(local), f'{host}:{remote}'], timeout)


def scp_down(host, remote, local, recursive=False, timeout=600):
    _run(['scp', '-q', *SSH_OPTS, *(['-r'] if recursive else []), f'{host}:{remote}', str(local)], timeout)


def check_connection(host, root):
    out = ssh(host, f'test -f {root}/scripts/app_job.sh && test -f {root}/jobs/app/infer.sub && echo ok; '
                    f'condor_q -af JobStatus 2>/dev/null | wc -l', timeout=40).split()
    if not out or out[0] != 'ok':
        raise ClusterError('Connected, but app_job.sh or jobs/app/infer.sub was not found under that root.')
    return f'Connected. App files found; you have {out[-1]} job(s) in the queue.'


def kind_of(name):
    ext = Path(name).suffix.lower()
    if ext in VIDEO_EXT: return 'video'
    if ext in IMAGE_EXT: return 'image'
    raise ClusterError(f'unsupported file type {ext}')


def prepare_input(data, name, max_seconds, workdir):
    """Write the upload to workdir as input.mp4 / input.jpg. Videos: trimmed to max_seconds, at most 1280 px wide,
    H.264, audio removed. Images: re-saved as JPEG, which also drops metadata such as GPS location."""
    workdir.mkdir(parents=True, exist_ok=True)
    kind = kind_of(name)
    raw = workdir / ('upload' + Path(name).suffix.lower())
    raw.write_bytes(data)
    if kind == 'image':
        from PIL import Image, ImageOps
        img = ImageOps.exif_transpose(Image.open(raw)).convert('RGB')
        out = workdir / 'input.jpg'; img.save(out, quality=95)
    else:
        out = workdir / 'input.mp4'
        ff = shutil.which('ffmpeg')
        if ff:
            _run([ff, '-loglevel', 'error', '-y', '-i', str(raw), '-t', str(max_seconds),
                  '-vf', "scale='min(1280,iw)':-2", '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                  '-pix_fmt', 'yuv420p', '-an', '-map_metadata', '-1', str(out)], timeout=900)
        else:   # no ffmpeg: upload as-is (the job processes the whole video)
            shutil.copy(raw, out)
    raw.unlink()
    return out, kind


def submit(host, root, local_input, kind, models, adapter, tracker):
    job_id = time.strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:6]
    remote = f'{root}/results/app/{job_id}'
    req = dict(models=models, input=local_input.name, kind=kind, adapter=adapter, tracker=tracker)
    rf = local_input.parent / 'request.json'; rf.write_text(json.dumps(req, indent=2))
    ssh(host, f'mkdir -p {remote}')
    scp_up(host, local_input, f'{remote}/{local_input.name}')
    scp_up(host, rf, f'{remote}/request.json')
    out = ssh(host, f'cd {root} && APP_JOB={remote} condor_submit jobs/app/infer.sub')
    m = re.search(r'submitted to cluster (\d+)', out)
    if not m:
        raise ClusterError('condor_submit did not report a job id: ' + out[-300:])
    return dict(job_id=job_id, remote=remote, cluster_id=m.group(1), request=req)


CONDOR_STATE = {'1': 'waiting for a GPU', '2': 'running', '3': 'removed', '4': 'finished', '5': 'held', '7': 'suspended'}


def poll(host, job):
    r = job['remote']
    out = ssh(host, f'cd {r} && for f in status_*.json; do [ -f "$f" ] && cat "$f" && echo @@; done; echo "##"; '
                    f'ls DONE FAILED 2>/dev/null; echo "##"; condor_q {job["cluster_id"]} -af JobStatus 2>/dev/null; '
                    f'echo "##"; tail -4 job.out 2>/dev/null; true', timeout=40)   # job.out appears only once the job starts
    parts = out.split('##')
    status = {}
    for chunk in parts[0].split('@@'):
        chunk = chunk.strip()
        if chunk:
            try:
                s = json.loads(chunk); status[s['model']] = s
            except (json.JSONDecodeError, KeyError):
                pass
    flags = parts[1].split() if len(parts) > 1 else []
    q = parts[2].strip() if len(parts) > 2 else ''
    return dict(status=status, done='DONE' in flags, failed='FAILED' in flags,
                condor=CONDOR_STATE.get(q, 'left the queue' if not q else q), tail=parts[3].strip() if len(parts) > 3 else '')


def fetch(host, job, local_dir):
    """Download outputs; make videos browser-playable (H.264). Returns {model: {...paths, summary}}."""
    local_dir.mkdir(parents=True, exist_ok=True)
    scp_down(host, f'{job["remote"]}/out', local_dir, recursive=True)
    ff = shutil.which('ffmpeg')
    res = {}
    for m in job['request']['models']:
        d = local_dir / 'out' / m
        if not (d / 'summary.json').is_file():
            continue
        item = dict(summary=json.loads((d / 'summary.json').read_text()), lanes=d / 'lanes.jsonl')
        if (d / 'overlay.png').is_file():
            item['image'] = d / 'overlay.png'
        elif (d / 'overlay.mp4').is_file():
            web = d / 'overlay_web.mp4'
            if ff and not web.is_file():
                _run([ff, '-loglevel', 'error', '-y', '-i', str(d / 'overlay.mp4'), '-c:v', 'libx264', '-preset', 'veryfast',
                      '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(web)], timeout=900)
            item['video'] = web if web.is_file() else d / 'overlay.mp4'
        res[m] = item
    return res


def failure_logs(host, job):
    try:
        return ssh(host, f'cd {job["remote"]} && tail -n 15 log_*.txt job.err 2>/dev/null', timeout=40)
    except ClusterError as e:
        return str(e)


def delete_remote(host, job):
    """Remove this run's upload and outputs from the cluster (only the app's own job directory)."""
    r = job['remote']
    assert '/results/app/' in r and r.rstrip('/').split('/')[-1] == job['job_id']
    ssh(host, f'rm -rf -- {r}')
