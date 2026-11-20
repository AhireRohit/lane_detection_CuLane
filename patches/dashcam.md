# dashcam phase (2026-11-19)
Added PHASE=dashcam to gpu_entry.sh (backup .before_dashcam), scripts/run_dashcam_video.py, dashcam_compare.py, dashcam_all.sh.
Baseline only (no temporal/geometry postprocessing). Same adapter for all models: frame resized to 1640x590 CULane canvas,
verified pre/forward/decode from benchmark_speed.py, coordinates scaled back to source pixels. Video: vendors/ufldv2/example.mp4
(UFLDv2 demo clip, upstream PR #115; footage origin undocumented). Unlabeled: report.json has availability/stability stats, no accuracy.
