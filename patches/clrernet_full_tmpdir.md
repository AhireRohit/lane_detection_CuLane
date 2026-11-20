# clrernet_full_tmpdir (2026-11-19)
CULaneMetric uses tempfile.TemporaryDirectory().name; predictions land in the worker's temp dir and are lost when Condor cleans up.
Fix: gpu_entry.sh clrernet/full sets TMPDIR=results/clrernet/full/run_<ts>/tmp and work_dir per run; asserts 34680 .lines.txt.
Vendor code unchanged. Backup: scripts/gpu_entry.sh.before_clrernet_full.

## Patch 2 (2026-11-19): job 73848 hung
Setting TMPDIR to $RUN/tmp (long path) made torch DataLoader workers fail with
`OSError: AF_UNIX path too long` (multiprocessing resource_sharer socket under TMPDIR/pymp-*), so no batches
reached the main process: 80 min, 0 predictions. Job removed with condor_rm; its run dir/logs are kept.
Fix: TMPDIR=$(mktemp -d /tmp/clr.XXXX) (short, container-local); after tools/test.py, copy the CULaneMetric
result dir (tempfile.TemporaryDirectory().name, never cleaned by the vendor) to $RUN/pred and assert 34680 files.
