# CULane Lane-Detection Benchmark — Final Report

**Project:** `Lane_Detection/benchmark_culane` on the Saarland HPC cluster (HTCondor, Tesla P100-SXM2-16GB)
**Date:** 2026-11-19
**Scope:** three pretrained CULane models — SCNN (PyTorch port), Ultra-Fast-Lane-Detection-v2 ResNet34, CLRerNet DLA34 EMA. No training, no TuSimple, no YOLOPv2.

Every number below was measured in this project unless it is explicitly labelled *published*. Published values come from the authors' papers/READMEs and are never mixed into measured columns.

---

## 1. Summary

| | SCNN | UFLDv2 ResNet34 | CLRerNet DLA34 EMA |
|---|---|---|---|
| CULane test F1 — **reproduced** (official C++ evaluator) | **70.07** | **75.98** | **81.55** |
| CULane test F1 — *published* | *71.6* | *76.0* | *81.55* |
| Delta | −1.53 | −0.02 | ±0.00 |
| Full-prediction latency, P100, batch 1 (median) | 63.4 ms | 48.1 ms | 20.1 ms |
| Full-prediction throughput (from median) | 15.8 FPS | 20.8 FPS | 49.8 FPS |

- UFLDv2 and CLRerNet reproduce their published F1 within 0.02. SCNN is 1.53 below the paper, most likely because the PyTorch port decodes probability maps in Python rather than with the original MATLAB `prob2lines`.
- CLRerNet is the most accurate in every scenario and the fastest end-to-end on the P100.
- **Downstream improvement:** a causal temporal lane tracker (B2) roughly halves lane re-starts and cuts frame-to-frame jitter on the dashcam clip, at < 1 ms/frame. A geometry-validity filter (B1/B3) was tested with two calibrations and **rejected**, because on this video it mostly removed real lanes.

---

## 2. Setup and provenance

| Model | Code (pinned commit) | Checkpoint | SHA-256 |
|---|---|---|---|
| SCNN | `delldu/SCNN` PyTorch port of the Torch7 original, `a6fadfbd9e593a28684dcaf29ca8ad425c1c700f` | `vgg_SCNN_DULR_w9.pth` (converted official Torch7 weights) | `07c0c768dc435b8b162b58f42c7f686656133e0b13b8b4f110214245edb16671` |
| UFLDv2 | `cfzd/Ultra-Fast-Lane-Detection-v2`, `c903880678454dfd9b55a63022368db05c00bc6d` | `culane_res34.pth` | `1b4ca8385122f74c46a2e274950cdb5eb132ff97aa1ab1311550526d83caebf6` |
| CLRerNet | `hirotomusiker/CLRerNet`, `dae038f67da57e292e5293a68c9c1c2922de13c2` | `clrernet_culane_dla34_ema.pth` (release v0.1.0) | `cce8b552bc8301e31186da192fe4213043e9351b35651ef5d40014ce3040c787` |

- **Container:** `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel` (torch 2.3.1, CUDA 12.1), one isolated venv per model (`--system-site-packages`, container torch never replaced).
- **CLRerNet stack:** MMDetection 3.3.0, MMCV 2.1.0 built from source for P100 (sm_60), mmengine 0.10.4, numpy pinned to 1.26.4.
- **GPU:** Tesla P100-SXM2-16GB, driver 580.178.04. All GPU work ran through HTCondor; nothing heavy ran on the login node.
- **Checkpoint loading:** strict (`strict=True`) for SCNN and UFLDv2. CLRerNet's only missing keys are `bbox_head.seg_decoder.*` and `bbox_head.loss_seg.criterion.weight` — the training-only auxiliary segmentation branch, which `predict()` does not use.
- **Vendor code was never modified.** All compatibility fixes are environment-level and documented in `patches/` (see §9).

## 3. Dataset audit

- CULane extracted at `datasets/CULane`: train/val/test lists of 88,880 / 9,675 / 34,680 entries; test entries unique.
- Test scenarios (frames): normal 9,621 · crowd 8,113 · highlight 486 · shadow 930 · no-line 4,067 · arrow 890 · curve 422 · crossroad 3,122 · night 7,029.
- Lane-segmentation label folders present (needed for SCNN-style segmentation labels).
- Observations, nothing changed: some test frames have empty `.lines.txt` labels (e.g. the first highlight and night entries) — normal for CULane; `annotations_new` covers only drivers 161, 182 and 23 and was **not** compared against the `driver_*` labels; an apparently empty nested folder `datasets/CULane/datasets/CULane` exists; `culane.zip` (45.7 GB) is still present. CULane is non-commercial and was not copied or redistributed.

## 4. Smoke tests (handbook Phase B)

Uniform 5-image GPU smoke (`scripts/smoke5.py`, Condor 73860): first non-empty-label frame of normal, crowd, highlight, curve and night. Gate: first image has lanes, ≥ 3/5 images have lanes, all coordinates finite and in frame, 5 prediction files and overlays, official evaluator finds TP > 0. **All three PASS** (`results/<model>/smoke5/run_*73860/PASS.txt`). Earlier single-image smokes: SCNN 73763, UFLDv2 73749, CLRerNet 73840.

Two failed attempts are kept: 73858 (my input picker chose frames with empty labels) and 73859 (an "every frame must have lanes" rule; the same frames are empty in the full-test predictions too, i.e. genuine model misses).

## 5. CULane accuracy

**Protocol:** official CULane C++ evaluator (`evaluate.cpp`, UFLDv2 vendor copy, unmodified, binary sha256 `65b36a3d…`), lane width 30 px, IoU 0.5, 1640×590, full test set of 34,680 images. The scoring script runs a GT-vs-GT self-check (must give F1 = 1) and a path preflight before scoring (`scripts/score_culane_official.sh`). The same evaluator scored all three models' saved predictions.

### 5.1 Published vs reproduced (`reports/published_vs_reproduced.csv`)

| Model | Published F1 | Reproduced F1 | Precision | Recall | Inference / scoring jobs | Test-time aug. |
|---|---|---|---|---|---|---|
| SCNN | 71.6 | 70.07 | 70.82 | 69.33 | 73765 / 73835 | none |
| UFLDv2 | 76.0 | 75.98 | 77.27 | 74.74 | 73755 / 73836 | author default `tta=True` |
| CLRerNet | 81.55 | 81.55 | 87.79 | 76.14 | 73864 / 73878 | none |

- CLRerNet's own Python `CULaneMetric` on the same predictions also gives F1 81.55 at IoU 0.5 (88.13 at IoU 0.1, 65.02 at IoU 0.75). The two evaluators differ by 5 TP out of ~105,000 GT lanes.
- UFLDv2's 75.98 uses the author-default test-time augmentation (`configs/culane_res34.py`: `tta=True`). An earlier version of the CSV said "none"; this was corrected.

### 5.2 Per scenario, F1 (`reports/culane_by_scenario.csv`)

| Scenario | SCNN | UFLDv2 | CLRerNet |
|---|---|---|---|
| Normal | 90.01 | 92.59 | **94.38** |
| Crowd | 68.68 | 74.84 | **80.80** |
| Highlight | 56.86 | 65.62 | **75.30** |
| Shadow | 61.39 | 75.46 | **84.55** |
| No line | 42.69 | 49.01 | **56.77** |
| Arrow | 83.71 | 88.20 | **91.06** |
| Curve | 62.79 | 69.76 | **78.88** |
| Crossroad (false positives, lower is better) | 1,926 | 1,851 | **1,335** |
| Night | 61.65 | 70.90 | **76.87** |

"No line" (worn or missing markings) is the hardest scenario for every model: 43–57 F1, against 90–94 in normal scenes. Published per-scenario values are deliberately omitted because they were not verified against the papers in this project.

## 6. Same-hardware speed (handbook Phase D)

Condor 73862 — one job ran all three models sequentially on the same P100. Batch 1, FP32, cuDNN benchmark off. Input: the first 250 CULane test frames, preloaded into memory (disk read and JPEG decode excluded); 50 warmup + 200 measured. CUDA-synchronised timing at every stage boundary. Full table and per-frame samples: `reports/speed.csv`, `results/speed/run_20261119T135639Z_*-73862/<model>/timing_samples.csv`.

| Model | Network input | Forward only, median / p95 | Full prediction, median / p90 / p95 / mean | FPS (median) | Pre / post median | Peak GPU mem |
|---|---|---|---|---|---|---|
| SCNN | 800×288 | 51.0 / 55.8 ms | 63.4 / 65.0 / 68.7 / 64.3 ms | 15.8 | 5.7 / 6.6 ms | 208 MB |
| UFLDv2 (no TTA) | 1600×320 | 11.6 / 11.8 ms | 48.1 / 49.3 / 52.5 / 47.7 ms | 20.8 | 20.8 / 15.6 ms | 906 MB |
| CLRerNet | vendor val pipeline | 13.9 / 16.0 ms | 20.1 / 22.6 / 22.8 / 20.4 ms | 49.8 | 3.1 / 3.2 ms | 120 MB |

- "Full prediction" = preprocessing from an in-memory 1640×590 frame + forward + decode (CLRerNet: NMS + lane decode) to source-frame coordinates.
- UFLDv2's network alone runs at 86 FPS; its end-to-end time is dominated by the author's PIL resize to 1600×533 (~21 ms) and the per-point Python decode loop (~16 ms). This measures the implementation, not the architecture.
- UFLDv2 was timed **without** TTA (its standard speed setting); its scored F1 run used TTA.
- The stage-split decode was verified against each vendor's reference path on frame 0: UFLDv2 max difference 0.0009 px, CLRerNet 0.0 px; SCNN uses the identical code path as its scored run.
- No literature FPS values are reported as measurements.

## 7. Dashcam baseline (handbook Phase E)

**Video:** `vendors/ufldv2/example.mp4` — the UFLDv2 repo's demo clip (added in upstream PR #115; original footage source undocumented), 1280×720, 25 fps, 1,260 frames, sha256 `651949922042c223375d40bf8929def97dbb62c6ade651d1f5b08eea47127d5f`. **The video has no labels: nothing in this section is an accuracy measurement.**

**Adapter (identical for all models):** resize each frame to 1640×590, run the verified pre/forward/decode, scale lanes back to source pixels. Because the video is 16:9 and CULane is 2.78:1, frames are squashed vertically (aspect factor 0.64). This puts all models at the same disadvantage relative to CULane-shaped footage.

| Model | Frames with ≥ 1 lane | Mean lanes/frame | Lane-count changes | Median latency | Video end-to-end FPS |
|---|---|---|---|---|---|
| SCNN | 97.1% | 3.33 | 291 | 62.6 ms | 13.8 |
| UFLDv2 | 99.9% | 2.00 | 4 | 45.6 ms | 17.7 |
| CLRerNet | 100% | 3.79 | 137 | 23.9 ms | 26.3 |

End-to-end FPS includes reading the video from disk, resizing, inference, drawing and mp4 encoding (Condor 73882).

**Visual review** (six evenly spaced frames on the contact sheet only, not a systematic review):
- **SCNN:** zig-zag false geometry on frame 0; no lanes on a bright concrete bridge (frame 1007); a lane drawn along the median barrier in several frames.
- **UFLDv2:** always exactly the two ego lanes — its non-TTA CULane decode emits only row lanes 1 and 2 by design. Stable and on the markings. Its low lane-count-change number is partly structural.
- **CLRerNet:** four lanes, mostly on markings; a false curved lane crossing the road on frame 0, and the right lane drawn across a car body on frames 755 and 1007.

Outputs: `results/dashcam/<model>/run_20261119T144052Z_*-73882/{overlay.mp4, frames.jsonl, report.json}`; `results/comparison/run_20261119T144052Z_*-73882/{three_model_contact_sheet.jpg, side_by_side.mp4, summary.json}`.

## 8. Downstream improvement and ablation (handbook Phase F)

**Motivation from the baseline:** lane flicker (CLRerNet 137 and SCNN 291 lane-count changes on the clip) and short-lived false lanes.

**Design** (`scripts/temporal_ablation.py`): model-agnostic, strictly causal (frame *t* uses frames ≤ *t* only), applied offline to each model's saved baseline lanes, so checkpoint and score threshold are exactly the baseline's. Lanes are resampled at common image rows in CULane 1640×590 coordinates.

- **B0:** raw baseline.
- **B1:** geometric validity filter — minimum vertical extent, maximum curvature (second difference), maximum slope |dx/dy|, and removal of the shorter lane of any crossing pair.
- **B2:** temporal tracker — Hungarian matching on median lateral distance at common rows (gate 40 px), EMA smoothing (α = 0.5), a new lane is shown only after 2 consecutive hits, and a lost lane is held for at most 3 frames (120 ms), drawn dashed and counted separately from observed lanes.
- **B3:** B1 + B2.

All tracker parameters were fixed before any result was seen, with no parameter search. The clip is reported as two halves (frames 0–629 and 630–1259), and nothing was tuned on either half.

### 8.1 Accepted: B2 temporal tracking (CPU Condor 73883, `reports/ablation_temporal.csv`)

Values are first half / second half.

| Model | Variant | Observed availability | Held-lane share | Lane-count changes | Lane re-starts per 100 frames | Jitter at y = 500, median / p95 (px) | Cost |
|---|---|---|---|---|---|---|---|
| CLRerNet | B0 | 100% / 100% | — | 96 / 41 | 19.4 / 6.7 | 1.64 / 10.9 · 1.69 / 8.7 | — |
| CLRerNet | **B2** | 99.8% / 100% | 9.3% / 4.4% | 87 / **51 (worse)** | **9.1 / 4.1** | **1.10 / 6.8 · 1.18 / 5.3** | 0.65 ms/frame |
| SCNN | B0 | 99.5% / 94.6% | — | 120 / 171 | 17.0 / 32.0 | 2.92 / 51.5 · 2.97 / 22.6 | — |
| SCNN | **B2** | 99.0% / 94.1% | 9.6% / 13.2% | **84 / 77** | **8.1 / 8.3** | **1.84 / 25.6 · 1.64 / 12.7** | 0.5–0.6 ms/frame |
| UFLDv2 | B0 | 99.8% / 100% | — | 2 / 2 | 0.8 / 0.2 | 0.77 / 9.9 · 0.85 / 9.8 | — |
| UFLDv2 | B2 | 99.7% / 100% | 1.5% / 0.1% | 9 / 0 | 1.0 / 0.0 | 0.98 / 5.2 · 1.02 / 5.3 | 0.2 ms/frame |

Jitter is in CULane-canvas pixels (× 0.78 for source-video pixels). "Lane re-starts" counts displayed lanes that cannot be matched to a lane in the previous frame (fragmentation).

- B2 roughly halves lane re-starts (SCNN second half: about a quarter) and reduces p95 jitter by 38–39% for CLRerNet and 44–50% for SCNN. Smoothed lanes stay within ~1.5 px (median) of the current observation, so lag is small.
- It is not a uniform win: CLRerNet's lane-count changes rise in the second half (41 → 51) because held lanes appear and disappear, and UFLDv2 was already stable.
- **Hold risk observed:** in one reviewed frame (frame 167, B3 run with the v2 filter) a held lane continued a false centre-lane detection for up to 3 frames.

### 8.2 Rejected: B1 / B3 geometry filter (two calibrations)

| Calibration | Source | Cost on CULane val (real lanes removed) | False lanes removed on CULane val | Effect on the video |
|---|---|---|---|---|
| v1 (73883) | CULane **val** ground-truth lane shapes | 1.07% of GT lanes | not measurable | UFLDv2: 862 real ego lanes removed by the curvature rule (lane-count changes 2 → 241). CLRerNet: extent and slope rules removed mostly real far-right lanes; the false centre curve on frames 178–179 was not caught. |
| v2 (73888 + 73889) | each model's own **val predictions**, labelled TP/FP against val GT | 1.1–1.3% of TP lanes | CLRerNet 7.0% · SCNN 16.0% · UFLDv2 20.9% | UFLDv2 rejections fell to 251 but are still real ego lanes. CLRerNet rejected ~0.9 of ~3.8 lanes per frame; of 12 sampled rejection frames, 10 were real far-right dashed markings and 2 were the right marking drawn across an occluding car. |

With v2, CLRerNet B3 looks more stable than B2 (lane-count changes 60 / 37 vs 87 / 51), but only because a lane is deleted per frame. Under the handbook's own criterion, stability obtained by deleting real lanes is not an improvement.

**Root cause:** the 16:9 → 1640×590 squash makes far lanes flatter and shorter than any correctly detected CULane lane, so no CULane-calibrated slope/extent rule can separate them from false lanes. The fix belongs in the video adapter (aspect-preserving crop), not in thresholds. Thresholds were not retuned on the video.

**Decision:** B2 (temporal tracking and smoothing, no geometry filter) is the one accepted improvement. B1/B3 are rejected for this footage.

## 9. Problems found and fixed (reproducibility notes)

| Issue | Evidence | Fix (no vendor code changed) | Note |
|---|---|---|---|
| UFLDv2 first scoring run invalid | job 73767: tp = fp = fn = 0 | evaluator builds `anno_dir + list_entry`, so directory arguments need a trailing `/`; new script adds self-check and preflight | run marked "do not use" |
| CLRerNet `np.sctypes` error | job 73764: numpy 2.2.6 pulled in by deps | pin numpy 1.26.4 | `patches/clrernet_numpy_pin.md` |
| CLRerNet `No module named 'libs'` | job 73838 | `PYTHONPATH` mirrors upstream Dockerfile | `patches/clrernet_pythonpath.md` |
| CLRerNet full test hung (80 min, 0 outputs) | job 73848: `OSError: AF_UNIX path too long` | my earlier TMPDIR patch used a long path, breaking DataLoader worker sockets; switched to a short `/tmp` dir, then copy predictions out | `patches/clrernet_full_tmpdir.md` |
| CLRerNet deterministic cuBLAS error | job 73863 | `CUBLAS_WORKSPACE_CONFIG=:4096:8`, as in upstream `docker/Dockerfile:56` | `patches/clrernet_pythonpath.md` |
| UFLDv2 TTA mislabelled in report | config check | CSV corrected | logged in `STATUS.md` |

All failed runs, logs and earlier results are preserved; edited scripts keep `.before_*` backups.

## 10. Limitations and open items

- **SCNN gap (−1.53):** attributed to Python vs MATLAB probability-map decoding; not proven.
- **Published per-scenario values** were not verified against the papers and are therefore not reported.
- **Dashcam evidence is weak by construction:** a single unlabeled clip of unknown origin, analysed as two halves of the same recording rather than independent videos. Stability metrics are proxies, not accuracy. Visual reviews covered small samples (6 contact-sheet frames; in the ablation, 12 random second-half frames plus 12 rejection frames for each of the two filter calibrations, all CLRerNet). The clip has now been seen several times, so it should not be used as a final held-out test for further changes.
- **Aspect distortion (0.64)** in the video adapter affects all baseline video results and was the root cause of the B1 failure.
- **TP/FP lane labelling for B1 v2** used half-resolution polylines, approximating the official spline drawing; used only for calibration, not as a score.
- **Not done:** comparison of `annotations_new` against `driver_*` labels; a machine-readable environment manifest (`reports/environment_manifest.json`); a `literature` FPS table; video group-3 throughput on more than one clip.

**Suggested next steps:**
1. Aspect-preserving video adapter (crop a 2.78:1 band), re-run the baseline, then re-test B1 on a **new** video.
2. Record a second dashcam clip as an independent evaluation set for B2.
3. Verify published per-scenario numbers from the papers and add them as a separate *published* table.

## 11. How to reproduce

All commands are run from `Lane_Detection/benchmark_culane` on the cluster; jobs go through HTCondor.

```bash
condor_submit jobs/smoke5/smoke5.sub                                   # 5-image smoke, 3 models
condor_submit jobs/pretrained_20261119_025415_bff1e4/clrernet_full.sub  # CLRerNet full test (author tools/test.py)
condor_submit jobs/eval/{scnn,ufldv2,clrernet}_official_eval.sub        # official C++ scoring
condor_submit jobs/speed/speed_all.sub                                  # same-GPU latency, 3 models
condor_submit jobs/dashcam/dashcam_all.sub                              # dashcam baseline + comparison
condor_submit jobs/ablation/temporal_ablation.sub                       # B0-B3, GT-calibrated filter
condor_submit jobs/b1_recal/valpred.sub                                 # val predictions (3 GPU jobs)
condor_submit jobs/b1_recal/calib_ablate.sub                            # prediction-calibrated filter + ablation
```

SCNN and UFLDv2 full inference use `scripts/gpu_entry.sh <model> full` (jobs 73765, 73755). Every job writes its own `.out`, `.err` and `.log` under `logs/`, and its results to a timestamped run directory.

## 12. File index

| What | Where |
|---|---|
| Status log (chronological, with every job id) | `STATUS.md` |
| Accuracy | `reports/published_vs_reproduced.csv`, `reports/culane_by_scenario.csv`, `results/<model>/official_eval/*/official_eval.json` |
| Speed | `reports/speed.csv`, `results/speed/run_*-73862/` |
| Dashcam | `results/dashcam/`, `results/comparison/run_*-73882/` |
| Ablation | `reports/ablation_temporal.csv`, `reports/ablation_temporal_v2_predcalib.csv`, `reports/b1_calibration_pred_val.json`, `results/ablation/` |
| Smoke evidence | `results/<model>/smoke5/run_*73860/` |
| Patches and rationale | `patches/*.md` |
| Scripts | `scripts/` (`gpu_entry.sh`, `smoke5.py`, `score_culane_official.sh`, `benchmark_speed.py`, `run_dashcam_video.py`, `dashcam_compare.py`, `temporal_ablation.py`, `val_predict.py`, `b1_calibrate_preds.py`) |
