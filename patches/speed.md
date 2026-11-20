# speed phase (2026-11-19)
Added PHASE=speed to scripts/gpu_entry.sh (backup gpu_entry.sh.before_speed), scripts/benchmark_speed.py, scripts/speed_all.sh.
One Condor job runs scnn, ufldv2, clrernet sequentially on the same P100. Batch 1, FP32, 50 warmup + 200 measured
frames (first 250 of list/test.txt, preloaded; disk/JPEG decode excluded). Stage-synchronized timing: pre / forward / post.
UFLDv2 decode is an in-memory copy of generate_lines_local and CLRerNet split (extract_feat+head | get_lanes) mirrors
predict(); both verified against the vendor reference path on frame 0 (job fails otherwise). UFLDv2 timed WITHOUT TTA.
No vendor code modified.
