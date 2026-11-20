# CLRerNet corruption-robust fine-tuning (2026-11-19)
Goal: test whether short fine-tuning with corruption augmentation improves robustness, against a CONTROL with the same schedule.
- configs_ft/clrernet_ft_control.py: authors' DLA34-EMA recipe unchanged; load_from released EMA checkpoint; 3 epochs;
  AdamW lr 6e-5 cosine -> 6e-6; batch 24 (single GPU, as authors); seed 0, deterministic; diff filter (thr 15) from
  data_aux/train_diffs.npz (computed by scripts/frame_diffs*.py, same values as vendor tool; dataset folder untouched).
- configs_ft/clrernet_ft_corrupt.py: identical + ONE OneOf corruption step (p=0.5, full resolution, before the crop):
  RandomRain, MotionBlur(7-25), ImageCompression(q5-40), RandomBrightnessContrast(brightness -0.7..-0.2).
  albumentations 0.4.6 implementations, NOT the test corruption code; shadow held out (unseen).
- Decided in advance: last-epoch checkpoint (no selection on any set); val F1 per epoch logged for monitoring only.
  Test evaluated once: official C++ evaluator on full test (clean) + robustness subset (2,895 imgs x 16 conditions).
- Smoke (job 73942): 10 iterations each, 7.9 GB at batch 24, ~2.0 s/iter; val F1 after 10 iters 82.29 / 82.78.
- gpu_entry.sh PHASE clrernet/finetune (backup .before_finetune). Vendor code unchanged. Original checkpoint untouched.
- frame diffs: 8 parallel chunks (73990) + merge (73991): 88,880 frames, 55,698 kept at thr 15 (data_aux/train_diffs.npz). Original single-pass job 73941 still running as a cross-check.
- cross-check (74046): 8-chunk diffs == single-pass frame_diffs.py (73941) exactly (max abs diff 0.0, same 55,698 kept). 73941 rewrote data_aux/train_diffs.npz at 19:37 with identical values, after the fine-tunes (73992) had loaded it.
