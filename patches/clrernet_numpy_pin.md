# clrernet_numpy_pin (2026-11-19)
Symptom: smoke job 73764 -> AttributeError: `np.sctypes` was removed in NumPy 2.0 (imgaug 0.4.0 via albumentations 0.4.6 via mmdet 3.3.0).
Cause: mim/pip install of mmdet stack pulled numpy 2.2.6 into .venvs/clrernet despite earlier 'numpy<2'.
Fix: scripts/gpu_entry.sh installs numpy==1.26.4 (--no-deps) into .venvs/clrernet only, once (.numpy1_ok sentinel).
Backup: scripts/gpu_entry.sh.before_numpy_pin. No vendor source, checkpoint or dataset changed.
