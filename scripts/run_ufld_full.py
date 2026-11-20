#!/usr/bin/env python3
"""Execute the actual upstream UFLDv2 evaluator, stubbing TRAIN-ONLY DALI import."""
import runpy,sys,types
from common import B,checkpoint,vendor
repo=vendor('ufldv2')
sys.path.insert(0,str(repo))
if 'data.dali_data' not in sys.modules:
    stub=types.ModuleType('data.dali_data')
    class NoTraining:
        def __init__(self,*args,**kwargs):raise RuntimeError('DALI train loader unavailable')
    stub.TrainCollect=NoTraining
    sys.modules['data.dali_data']=stub
sys.argv=['test.py',str(B/'config/culane_res34_cluster.py'),
          '--test_model',str(checkpoint('ufldv2')),
          '--test_work_dir',str(B/'results/ufldv2/full'),
          '--batch_size','1']
(B/'results/ufldv2/full').mkdir(parents=True,exist_ok=True)
runpy.run_path(str(repo/'test.py'),run_name='__main__')
