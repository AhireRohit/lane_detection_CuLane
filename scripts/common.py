"""Paths and basic dataset/checkpoint checks. Python standard library only."""
from pathlib import Path
import json

B = Path(__file__).resolve().parents[1]
R = B.parent
D = R / 'datasets' / 'CULane'
SOURCES = json.loads((B/'config'/'sources.json').read_text())

def test_images(limit=None):
    p = D/'list'/'test.txt'
    if not p.is_file():
        raise FileNotFoundError(f'Missing CULane test list: {p}')
    for line in p.open(encoding='utf8'):
        if not line.strip():
            continue
        rel = line.split()[0].lstrip('/')
        if not rel.lower().endswith(('.jpg', '.jpeg', '.png')):
            raise ValueError(f'Unexpected test list entry: {line[:120]}')
        yield rel

def checkpoint(name):
    return B/SOURCES['models'][name]['checkpoint_file']

def vendor(name):
    return B/'vendors'/name

def valid_checkpoint(name):
    cp = checkpoint(name)
    return cp.is_file() and cp.stat().st_size > 1024 * 1024

def sample_image():
    for rel in test_images():
        p = D/rel
        if p.is_file():
            return p
    raise FileNotFoundError('No image from list/test.txt exists at dataset root.')
