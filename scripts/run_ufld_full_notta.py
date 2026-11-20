#!/usr/bin/env python3
"""UFLDv2 full CULane test WITHOUT TTA (author eval_wrapper.run_test path), to match the setting used for
speed and dashcam. Identical to run_ufld_full.py except config tta=False and a separate work dir."""
import runpy, sys, types
from common import B, checkpoint, vendor
repo = vendor('ufldv2'); sys.path.insert(0, str(repo))
if 'data.dali_data' not in sys.modules:
    stub = types.ModuleType('data.dali_data')
    class NoTraining:
        def __init__(self, *a, **k): raise RuntimeError('DALI train loader unavailable')
    stub.TrainCollect = NoTraining; sys.modules['data.dali_data'] = stub
W = B / 'results/ufldv2/full_notta'; W.mkdir(parents=True, exist_ok=True)
sys.argv = ['test.py', str(B / 'config/culane_res34_cluster_notta.py'), '--test_model', str(checkpoint('ufldv2')),
            '--test_work_dir', str(W), '--batch_size', '1']
runpy.run_path(str(repo / 'test.py'), run_name='__main__')
