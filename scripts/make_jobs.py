#!/usr/bin/env python3
"""Submit only verified-model branches. smoke -> full; no training or silent fallback."""
import argparse,datetime,os,sys,uuid
from pathlib import Path
from common import B,R,D,SOURCES,valid_checkpoint

def make(models,require_all=False):
    if not (D/'list/test.txt').is_file():raise RuntimeError('CULane list/test.txt missing')
    available=[];missing=[]
    for model in models:
        if valid_checkpoint(model) and (B/'vendors'/model/'.git').is_dir():
            available.append(model)
        else:missing.append(model)
    if missing:
        print('SKIPPED (missing checkpoint or un-cloned repository):',', '.join(missing))
    if require_all and missing:raise RuntimeError('Required models missing, refusing partial DAG')
    if not available:raise RuntimeError('No models ready. See scripts/download_weights.sh')
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path=B/'jobs'/f'pretrained_{stamp}_{uuid.uuid4().hex[:6]}'
    path.mkdir(parents=True,exist_ok=False)
    logs=B/'logs';logs.mkdir(exist_ok=True)
    dag=[]
    for model in available:
        for phase in ('smoke','full'):
            label=f'{model}_{phase}'
            sub=path/f'{label}.sub'
            sub.write_text('\n'.join([
                'universe = docker',
                'docker_image = pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel',
                'executable = /bin/bash',
                f'arguments = {B}/scripts/gpu_entry.sh {model} {phase}',
                f'initialdir = {B}',
                f'output = {logs}/{label}.$(ClusterId).$(ProcId).out',
                f'error = {logs}/{label}.$(ClusterId).$(ProcId).err',
                f'log = {logs}/{label}.$(ClusterId).log',
                'should_transfer_files = NO',
                'request_GPUs = 1',
                'request_CPUs = 4',
                'request_memory = 16G',
                'request_disk = 8G',
                'requirements = UidDomain == "cs.uni-saarland.de"',
                '+WantGPUHomeMounted = true',
                '+WantScratchMounted = true',
                'queue 1','']))
            dag.append(f'JOB {model.upper()}_{phase.upper()} {sub.name}')
        dag.append(f'PARENT {model.upper()}_SMOKE CHILD {model.upper()}_FULL')
    (path/'batch.dag').write_text('\n'.join(dag)+'\n')
    (B/'last_jobdir.txt').write_text(str(path)+'\n')
    print('JOB DIRECTORY:',path)
    print('READY MODELS:', ', '.join(available))
    print('MAX PARALLEL: recommend 1 GPU job at a time')
    print(f'SUBMIT: (cd {path} && condor_submit_dag -maxjobs 1 batch.dag)')
    return path
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--models',nargs='+',choices=['scnn','ufldv2','clrernet'],default=['scnn','ufldv2','clrernet'])
    ap.add_argument('--require-all',action='store_true')
    args=ap.parse_args()
    try:make(args.models,args.require_all)
    except Exception as e:sys.exit('ERROR: '+str(e))
