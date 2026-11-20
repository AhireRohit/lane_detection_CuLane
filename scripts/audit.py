#!/usr/bin/env python3
"""Conservative preflight: read lists once; no recursive walk of 500k files."""
import argparse, json, hashlib, os, sys
from pathlib import Path
from common import R,B,D,SOURCES,test_images

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--sample',type=int,default=100)
    args=ap.parse_args()
    checks={}
    checks['project_root']=str(R)
    checks['dataset_root']=str(D)
    checks['dataset_exists']=D.is_dir()
    checks['dataset_nested_error']=(D/'CULane'/'list'/'test.txt').exists()
    checks['dataset_archive_exists']=(D/'culane.zip').exists()
    for name in ['train.txt','val.txt','test.txt']:
        p=D/'list'/name
        checks['list_'+name]=p.exists()
        if p.is_file():
            with p.open(errors='replace') as f:
                checks['list_'+name+'_lines']=sum(bool(s.strip()) for s in f)
    test=list(test_images()) if (D/'list/test.txt').exists() else []
    checks['test_entries']=len(test)
    checks['test_unique_entries']=len(set(test))
    step=max(1, len(test)//max(1,args.sample))
    sampled=test[::step][:args.sample]
    checks['sampled_test_count']=len(sampled)
    checks['sampled_test_missing']=[s for s in sampled if not (D/s).is_file()]
    checks['sampled_test_gt_missing']=[s for s in sampled if not (D/s).with_suffix('.lines.txt').exists() and not (D/'annotations_new'/s).with_suffix('.lines.txt').exists()]
    checks['laneseg_label_w16_exists']=(D/'laneseg_label_w16').exists()
    checks['laneseg_label_w16_test_exists']=(D/'laneseg_label_w16_test').exists()
    checks['annotations_new_exists']=(D/'annotations_new').exists()
    test_splits=D/'list/test_split'
    checks['test_scenarios']={p.name: sum(bool(z.strip()) for z in p.open()) for p in sorted(test_splits.glob('test*.txt'))} if test_splits.is_dir() else {}
    for model,meta in SOURCES['models'].items():
        cp=B/meta['checkpoint_file']; repo=B/'vendors'/model
        checks[model+'_checkpoint_size']=cp.stat().st_size if cp.exists() else 0
        checks[model+'_vendor_git']=(repo/'.git').exists()
    out=B/'results/audit.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))
    if not checks['dataset_exists'] or not checks['list_test.txt'] or checks['sampled_test_missing'] or checks['dataset_nested_error']:
        sys.exit('Dataset audit FAILED: fix path/list issues before jobs.')
    print('DATASET BASIC PREFLIGHT PASS. This is not full integrity verification.')
if __name__=='__main__':main()
