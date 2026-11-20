#!/usr/bin/env python3
"""Export the corruption-fine-tuned CLRerNet (seed 0, report section 7.1) as a small inference checkpoint for the demo.

Keeps only 'state_dict' (+ 'meta'), which is exactly what mmdet init_detector loads; optimizer and message-hub
state are dropped. Checks that every tensor is bit-identical to the training checkpoint's state_dict.
Usage: export_robust_ckpt.py <epoch_3.pth> <out.pth>
"""
import sys, hashlib
import torch

src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location='cpu', weights_only=False)
print('keys in training checkpoint:', list(ck))
out = {'state_dict': ck['state_dict'], 'meta': ck.get('meta', {})}
torch.save(out, dst)
re = torch.load(dst, map_location='cpu', weights_only=False)['state_dict']
assert re.keys() == ck['state_dict'].keys()
assert all(torch.equal(re[k], ck['state_dict'][k]) for k in re), 'tensor mismatch'
h = hashlib.sha256(open(dst, 'rb').read()).hexdigest()
print(f'EXPORT OK {len(re)} tensors identical; sha256 {h}  {dst}')
