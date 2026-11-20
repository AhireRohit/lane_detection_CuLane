#!/usr/bin/env python3
"""Write reports/environment_manifest.json: container, per-model venv packages, checkpoints, vendor commits, GPU (from speed run)."""
import json, subprocess, hashlib, platform, glob, os
from pathlib import Path
B = Path(__file__).resolve().parents[1]
def sh(cmd): return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
def head(repo):
    g = repo / '.git'; h = (g / 'HEAD').read_text().strip()
    if not h.startswith('ref:'): return h
    ref = h.split()[1]
    if (g / ref).is_file(): return (g / ref).read_text().strip()
    for l in (g / 'packed-refs').read_text().splitlines():
        if l.endswith(' ' + ref): return l.split()[0]
SRC = json.loads((B / 'config/sources.json').read_text())
out = dict(created_utc=sh('date -u +%FT%TZ'), docker_image='pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel',
           container_python=platform.python_version(), host_kernel=platform.release(), models={})
spd = sorted(glob.glob(str(B / 'results/speed/run_*/scnn/speed_manifest.json')))
if spd:
    s = json.load(open(spd[-1])); out['gpu'] = dict(name=s['gpu'], nvidia_smi=s['nvidia_smi'], torch=s['torch'], cuda=s['cuda'], cudnn=s['cudnn'], source=spd[-1])
for m in ['scnn', 'ufldv2', 'clrernet']:
    py = B / '.venvs' / m / 'bin/python'
    ck = B / SRC['models'][m]['checkpoint_file']
    h = hashlib.sha256()
    with open(ck, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): h.update(c)
    out['models'][m] = dict(repo=SRC['models'][m]['repo'], vendor_commit=head(B / 'vendors' / m), checkpoint=str(ck.relative_to(B)),
        checkpoint_sha256=h.hexdigest(), checkpoint_source=SRC['models'][m]['checkpoint_source'],
        venv_python=sh(f'{py} -c "import sys;print(sys.version.split()[0])"'),
        key_versions=sh(f'{py} -c "import torch,numpy,cv2;print(torch.__version__,numpy.__version__,cv2.__version__)"'),
        pip_freeze=sh(f'{py} -m pip freeze --all').splitlines())
    if m == 'clrernet':
        out['models'][m]['mm_versions'] = sh(f'{py} -c "import mmdet,mmcv,mmengine;print(mmdet.__version__,mmcv.__version__,mmengine.__version__)"')
out['evaluator'] = dict(path='vendors/ufldv2/evaluation/culane/build_conda_gcc/culane_evaluator',
                        sha256=sh(f'sha256sum {B}/vendors/ufldv2/evaluation/culane/build_conda_gcc/culane_evaluator').split()[0])
(B / 'reports/environment_manifest.json').write_text(json.dumps(out, indent=2))
print(json.dumps({m: dict(commit=v['vendor_commit'][:8], sha=v['checkpoint_sha256'][:8], versions=v['key_versions'], n_pkgs=len(v['pip_freeze'])) for m, v in out['models'].items()}, indent=1))
print('evaluator', out['evaluator']['sha256'][:16], 'gpu', out.get('gpu', {}).get('name'))
