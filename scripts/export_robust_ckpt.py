#!/usr/bin/env python3
"""Export the corruption-fine-tuned CLRerNet (seed 0, report section 7.1) as a small inference checkpoint for the demo.

Keeps only 'state_dict' (what mmdet init_detector loads) and meta['dataset_meta']; optimizer, message-hub state and the
training config (which contains cluster paths) are dropped. Checks that every tensor is bit-identical to the training checkpoint's state_dict.
Usage: export_robust_ckpt.py <epoch_3.pth> <out.pth>
"""
import sys, hashlib
import torch

src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location='cpu', weights_only=False)
print('keys in training checkpoint:', list(ck))
meta = {k: v for k, v in (ck.get('meta') or {}).items() if k == 'dataset_meta'}
out = {'state_dict': ck['state_dict'], 'meta': meta}
print('meta kept:', list(meta), '| dropped:', [k for k in (ck.get('meta') or {}) if k not in meta])
torch.save(out, dst)
re = torch.load(dst, map_location='cpu', weights_only=False)['state_dict']
assert re.keys() == ck['state_dict'].keys()
assert all(torch.equal(re[k], ck['state_dict'][k]) for k in re), 'tensor mismatch'
h = hashlib.sha256(open(dst, 'rb').read()).hexdigest()
blob = open(dst, 'rb').read()
assert b'/home/' not in blob, 'path strings left in export'
print(f'EXPORT OK {len(re)} tensors identical; sha256 {h}  {dst}')
