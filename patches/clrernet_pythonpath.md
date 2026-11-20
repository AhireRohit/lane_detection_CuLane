# clrernet_pythonpath (2026-11-19)
Symptom: smoke job 73838 -> ModuleNotFoundError: No module named 'libs' (demo/image_demo.py).
Cause: upstream docker/Dockerfile sets PYTHONPATH to repo root and installs nms via setup.py install; our env built nms in-place.
Fix: gpu_entry.sh exports PYTHONPATH=vendors/clrernet:vendors/clrernet/libs/models/layers/nms/src for clrernet only.
Backup: scripts/gpu_entry.sh.before_pythonpath.
Note: import check must import torch first (details.so links libc10.so); job 73839 failed on check order only.

## CUBLAS_WORKSPACE_CONFIG (2026-11-19)
Job 73863 (after the short-TMPDIR fix, workers delivered batches) failed at the first forward pass:
RuntimeError deterministic algorithms + cuBLAS needs CUBLAS_WORKSPACE_CONFIG. The CLRerNet config sets
randomness=dict(seed=0, deterministic=True) and the upstream docker/Dockerfile:56 sets
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8. Mirrored in gpu_entry.sh for MODEL=clrernet (backup .before_cublas).
