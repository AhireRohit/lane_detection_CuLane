# Corruption robustness benchmark (2026-11-19)
scripts/robust_common.py (subset + deterministic corruptions), robustness.py (GPU inference, verified builders, UFLDv2 non-TTA),
score_robustness.py (official C++ evaluator, identical-input md5 check). PHASE robust in gpu_entry.sh (backup .before_robust).
Subset: every 12th entry of each test_split list. 5 corruptions x 3 severities + clean. Design fixed before results.
- Job 73900.2 (clrernet) exit 2 AFTER 'ROBUST DONE': bash 'unexpected EOF' in gpu_entry.sh, which was edited (crop/split patches) while the job ran; bash reads scripts incrementally. All 16 conditions complete (2895 files each), manifest written. Lesson: never edit gpu_entry.sh while jobs use it.
