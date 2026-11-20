# smoke5 phase (2026-11-19)
Added PHASE=smoke5 to scripts/gpu_entry.sh (backup: gpu_entry.sh.before_smoke5) and scripts/smoke5.py.
Handbook Phase B 5-image GPU smoke, uniform across scnn/ufldv2/clrernet: first image of test0_normal,
test1_crowd, test2_hlight, test6_curve, test8_night. Each model uses its author/full-run decoder.
Outputs results/<model>/smoke5/run_<ts>_<host>/ (manifest, overlays, raw json, timings, logs, PASS.txt only if all checks pass).
The 5-image C++ evaluator run is a format/geometry sanity gate, not an accuracy result.
UFLDv2 smoke uses non-TTA decode (mode=normal); the scored full run used tta=True.
No vendor code modified.
- v2: pick first entry with non-empty GT (job 73858 failed: first hlight/night test frames have empty labels).
- v3: job 73859 ran all 3 pipelines end-to-end but failed the all-5-frames-have-lanes rule (night frame empty for all models; hlight empty for ufldv2/clrernet). The same frames are empty in the full-run predictions (scnn night=0; ufldv2 hlight=0, night=0), so these are model misses, not pipeline faults. Rule relaxed to >=3/5 frames with lanes; first (normal) frame must still have lanes; evaluator TP>0 still required. Failed runs kept.
