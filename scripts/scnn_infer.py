#!/usr/bin/env python3
"""SCNN PyTorch port inference to CULane .lines.txt. Uses vendor's original prob2lines_CULane."""
import argparse, os, sys, json, time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from common import B,D,test_images,checkpoint,vendor

def extract_state(raw):
    if isinstance(raw,dict):
        for key in ('net','state_dict','model','model_state_dict'):
            if key in raw and isinstance(raw[key],dict):
                raw=raw[key];break
    if not isinstance(raw,dict) or not raw: raise ValueError('No state dictionary in SCNN checkpoint')
    result={}
    for k,v in raw.items():
        k=k[7:] if k.startswith('module.') else k
        result[k]=v
    return result

def run(max_frames):
    src=vendor('scnn')
    if not (src/'model.py').exists(): raise SystemExit('SCNN vendor not installed: run bootstrap.sh')
    sys.path.insert(0,str(src))
    from model import SCNN
    from utils.prob2lines import getLane
    cp=checkpoint('scnn')
    if not cp.exists() or cp.stat().st_size<1000000:raise SystemExit('SCNN checkpoint missing/too small')
    if not torch.cuda.is_available(): raise SystemExit('GPU required for this job')
    model=SCNN((800,288),pretrained=False)
    # Never silently ignore missing or unexpected keys: a partial load is not a reproduction.
    raw=torch.load(str(cp),map_location='cpu',weights_only=False)
    state=extract_state(raw)
    model.load_state_dict(state,strict=True)
    model=model.cuda().eval()
    out=B/'results/scnn'/('smoke' if max_frames else 'full')
    pred_dir=out/'coord_output';pred_dir.mkdir(parents=True,exist_ok=True)
    times=[];drawn=0;det=0
    colors=[(0,255,0),(255,0,0),(0,180,255),(255,0,255)]
    for index,rel in enumerate(test_images()):
        if max_frames and index>=max_frames: break
        source=D/rel
        img=cv2.imread(str(source))
        if img is None: raise RuntimeError(f'Cannot read {source}')
        rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        resized=cv2.resize(rgb,(800,288),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255
        resized=(resized-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        tensor=torch.from_numpy(resized.transpose(2,0,1).copy()).unsqueeze(0).cuda()
        torch.cuda.synchronize();t0=time.perf_counter()
        with torch.no_grad():
            seg, exist = model(tensor)[:2]
            seg=F.softmax(seg,dim=1)[0].cpu().numpy()
            exist=exist[0].cpu().numpy()
        torch.cuda.synchronize();times.append(1000*(time.perf_counter()-t0))
        lanes=getLane.prob2lines_CULane(seg,[int(x>0.5) for x in exist],resize_shape=(590,1640),y_px_gap=20,pts=18)
        dest=pred_dir/Path(rel).with_suffix('.lines.txt');dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('w') as f:
            for lane in lanes:
                if len(lane)>1:
                    f.write(' '.join(f'{x:.2f} {y:.2f}' for x,y in lane)+'\n')
        if any(len(l)>1 for l in lanes): det+=1
        if drawn < 8:
            preview=img.copy()
            for j,lane in enumerate(lanes):
                pts=np.array(lane,dtype=np.int32)
                if len(pts)>1:cv2.polylines(preview,[pts],False,colors[j%len(colors)],3)
            target=out/'previews'/f'{index:04d}.jpg';target.parent.mkdir(exist_ok=True,parents=True)
            cv2.imwrite(str(target),preview);drawn+=1
        if index % 100 == 0: print(f'SCNN processed {index+1}',flush=True)
    results={'model':'SCNN','frames':len(times),'frames_with_lanes':det,'mean_inference_ms':float(np.mean(times)) if times else None,
             'median_inference_ms':float(np.median(times)) if times else None,'model_source':str(src),'checkpoint':str(cp),
             'metric_status':'NOT_SCORED; run official CULane evaluator on complete outputs only' }
    (out/'inference.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--max-frames',type=int,default=4,
        help='0 means full CULane test; smoke default 4');args=ap.parse_args();run(args.max_frames)
