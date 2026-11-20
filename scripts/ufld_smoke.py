#!/usr/bin/env python3
"""UFLDv2 single-image smoke test using author model and checkpoint; fail strictly."""
import sys, types, json, time
from pathlib import Path
from common import B,D,checkpoint,vendor,sample_image
import torch

def main():
    repo=vendor('ufldv2');sys.path.insert(0,str(repo))
    # UFLDv2 imports NVIDIA DALI for its TRAIN loader even when only running evaluation.
    # Stub that unused import only; it cannot be used for training and contains no data.
    if 'data.dali_data' not in sys.modules:
        stub=types.ModuleType('data.dali_data')
        class NoTraining:
            def __init__(self,*args,**kwargs): raise RuntimeError('DALI train loader intentionally unavailable')
        stub.TrainCollect=NoTraining
        sys.modules['data.dali_data']=stub
    from utils.common import merge_config,get_model
    cfgpath=B/'config/culane_res34_cluster.py'
    sys.argv=['ufld_smoke.py',str(cfgpath),'--test_model',str(checkpoint('ufldv2'))]
    args,cfg=merge_config()
    import cv2,numpy as np
    from PIL import Image
    import torchvision.transforms as T
    img=Image.open(sample_image()).convert('RGB')
    transforms=T.Compose([T.Resize((int(cfg.train_height/cfg.crop_ratio),cfg.train_width)),T.ToTensor(),
                         T.Normalize((.485,.456,.406),(.229,.224,.225))])
    img=transforms(img)[:,-cfg.train_height:,:].unsqueeze(0).cuda()
    model=get_model(cfg)
    raw=torch.load(str(checkpoint('ufldv2')),map_location='cpu',weights_only=False)
    if not isinstance(raw,dict) or 'model' not in raw: raise ValueError('Wrong UFLDv2 checkpoint format (expected model key)')
    weights={k.removeprefix('module.'):v for k,v in raw['model'].items()}
    model.load_state_dict(weights,strict=True)
    model.eval()
    torch.cuda.synchronize();t=time.perf_counter()
    with torch.no_grad(): pred=model(img)
    torch.cuda.synchronize()
    shapes={k:list(v.shape) for k,v in pred.items() if isinstance(v,torch.Tensor)}
    print(json.dumps({'smoke':'PASS','model':'ufldv2','inference_ms':(time.perf_counter()-t)*1000,'shapes':shapes},indent=2))

if __name__=='__main__':main()
