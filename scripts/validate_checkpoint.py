#!/usr/bin/env python3
import sys,hashlib,json
from common import checkpoint
if __name__=='__main__':
    name=sys.argv[1]
    cp=checkpoint(name)
    if not cp.exists() or cp.stat().st_size<1024*1024:raise SystemExit(f'MISSING CHECKPOINT {name}: {cp}')
    import torch
    try: state=torch.load(cp,map_location='cpu',weights_only=False)
    except Exception as e:raise SystemExit(f'Checkpoint cannot be loaded: {type(e).__name__} {e}')
    h=hashlib.sha256()
    with cp.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    print(json.dumps({'model':name,'size':cp.stat().st_size,'sha256':h.hexdigest(),
          'torch_keys':list(state)[:15] if isinstance(state,dict) else str(type(state))},indent=2))
