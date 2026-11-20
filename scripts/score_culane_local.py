#!/usr/bin/env python3
"""Independent approximate CULane lane-F1 scorer; NOT the official C++ benchmark.

Uses 30-pixel thick line masks (OpenCV) at original 1640x590 geometry, an
IoU>0.5 Hungarian assignment, and one-to-one matches. Official evaluator
implementation details can differ; label this as LOCAL PROXY only.
"""
import argparse,json,sys,time
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from common import D,B,test_images

H,W=590,1640

def parse_lanes(path):
    lanes=[]
    if not path.exists(): return None
    for line in path.read_text(errors='replace').splitlines():
        nums=line.split()
        if len(nums)<4 or len(nums)%2:continue
        try: p=np.array([float(z) for z in nums],dtype=np.float32).reshape(-1,2)
        except ValueError:continue
        p=p[np.isfinite(p).all(axis=1)]
        p=p[(p[:,0]>=0)&(p[:,0]<W)&(p[:,1]>=0)&(p[:,1]<H)]
        if len(p)>=2: lanes.append(np.rint(p).astype(np.int32))
    return lanes

def draw(lane,width=30):
    mask=np.zeros((H,W),dtype=np.uint8)
    cv2.polylines(mask,[lane],False,1,thickness=width,lineType=cv2.LINE_8)
    return mask

def match(pred,gt):
    if not pred:return 0,0,len(gt)
    if not gt:return 0,len(pred),0
    pm=[draw(z) for z in pred];gm=[draw(z) for z in gt]
    mat=np.zeros((len(pm),len(gm)),dtype=np.float32)
    for i,a in enumerate(pm):
        for j,b in enumerate(gm):
            inter=cv2.countNonZero(cv2.bitwise_and(a,b))
            union=cv2.countNonZero(cv2.bitwise_or(a,b))
            mat[i,j]=inter/union if union else 0
    rows,cols=linear_sum_assignment(1-mat)
    tp=sum(mat[r,c]>0.5 for r,c in zip(rows,cols))
    return tp,len(pred)-tp,len(gt)-tp

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pred',type=Path,required=True)
    ap.add_argument('--max-frames',type=int,default=100,help='0 full test; first 100 default')
    ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    t0=time.time();tp=fp=fn=missing_gt=missing_pred=n=seen=0
    for rel in test_images():
        if args.max_frames and seen>=args.max_frames:break
        seen += 1
        pred_path=args.pred/Path(rel).with_suffix('.lines.txt')
        gt_path=D/'annotations_new'/Path(rel).with_suffix('.lines.txt')
        if not gt_path.exists():gt_path=(D/Path(rel)).with_suffix('.lines.txt')
        gt=parse_lanes(gt_path)
        if gt is None:missing_gt+=1; continue
        pred=parse_lanes(pred_path)
        if pred is None:missing_pred+=1;continue
        a,b,c=match(pred,gt);tp+=a;fp+=b;fn+=c;n+=1
        if n%500==0:print('LOCAL scorer frames',n,flush=True)
    precision=tp/(tp+fp) if tp+fp else 0
    recall=tp/(tp+fn) if tp+fn else 0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0
    res={'WARNING':'LOCAL PROXY metric, NOT official CULane published F1',
        'frames_scored':n,'missing_gt_files':missing_gt,'missing_prediction_files':missing_pred,
        'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,'f1':f1,
        'mask_width_px':30,'iou_threshold':0.5,'seconds':round(time.time()-t0,1)}
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(res,indent=2)+'\n')
    print(json.dumps(res,indent=2))
    if missing_gt or missing_pred:
        raise SystemExit('Missing files: score is incomplete and cannot be compared to published scores.')
if __name__=='__main__':main()
