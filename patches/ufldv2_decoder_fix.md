# UFLDv2 in-memory decoder fix (2026-11-19)
Bug: benchmark_speed.build_ufld.post (and smoke5.make_ufld) reproduced only generate_lines_local (row anchors, ego lanes 1,2).
The author's non-TTA CULane path (eval_wrapper.run_test) ALSO appends generate_lines_col_local (column anchors, lanes 0,3).
The v1 equivalence check compared against generate_lines_local alone, so it passed. Found via robustness clean F1 57.4 vs 75.86.
Fix: column branch added; verify now compares to row+col author output. verify_ufld_decode.py (job 73914): 50/50 images exact
(max 0.001 px) with the author's PIL loader; with cv2 loader median 0.08 px, 46/50 identical counts (JPEG decoder differences).
Affected + rerun: UFLDv2 speed, smoke5, dashcam (squash+crop), val predictions/B1 calibration, ablations, robustness.
Unaffected: all official F1 numbers (author test.py), SCNN, CLRerNet. Old outputs kept as *_INVALID_rowonly_decoder or with INVALID_UFLDV2_DECODER.txt.
- Speed rerun 73915 contaminated by concurrent jobs (loadavg 9-15); marked INVALID_CONTENDED. Lesson: speed benchmarks run alone.
