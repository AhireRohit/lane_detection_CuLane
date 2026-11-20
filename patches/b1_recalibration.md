# B1 recalibration on model predictions (2026-11-19)
Reason: GT-calibrated B1 (job 73883) rejected real lanes on the video (ufldv2 curvature rule: jagged row-wise decode;
clrernet extent/slope: 16:9 squash). New: PHASE=valpred (gpu_entry backup .before_valpred) + scripts/val_predict.py writes
CULane VAL predictions with the same decode as the video; scripts/b1_calibrate_preds.py labels lanes TP/FP (width 30, IoU 0.5,
half-res polylines) and sets per-model thresholds from TP lanes (extent p0.5, curv/slope p99.5) - rule fixed before results.
temporal_ablation.py v2 (backup .v1_gt_calib) accepts per-model thresholds. Video already seen: development evidence only.
Output root: /path/to/Lane_Detection/benchmark_culane/results/b1_recal/valpred_20261119T145622Z
Result: filter rejected for this footage; see STATUS.md "Phase F v2". No further threshold changes made on the video.
