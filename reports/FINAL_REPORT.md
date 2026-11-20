# CULane Lane-Detection Benchmark — Final Report (v5)

**Project:** `Lane_Detection/benchmark_culane` on the Saarland HPC cluster (HTCondor, Tesla P100-SXM2-16GB)
**Date:** 2026-11-20 · **v5** adds a cross-dataset test on TuSimple (§7.3), the tracker's accuracy on labeled video (§9.3), ONNX FP32/INT8 CPU deployment of CLRerNet (§6.1), a public demo with persistent model workers, and CI. v4 is kept as `reports/FINAL_REPORT_v4.md`. · **v4** added three-seed results for the fine-tunes and a broad-augmentation arm (§7.2). v3 is kept as `reports/FINAL_REPORT_v3.md`. · **v3** added a corruption-augmented fine-tune of CLRerNet with a same-schedule control (§7.1). v2 is kept as `reports/FINAL_REPORT_v2.md`. · **v2** (2026-11-19) added robustness, verified published per-scenario values, UFLDv2 without TTA, an aspect-preserving video adapter, a failure gallery and the environment manifest. It also corrects a UFLDv2 decoder bug that affected v1's UFLDv2 speed, video and ablation numbers (see §11). v1 is kept as `reports/FINAL_REPORT_v1.md`.
**Scope:** three pretrained CULane models — SCNN (PyTorch port), Ultra-Fast-Lane-Detection-v2 ResNet34, CLRerNet DLA34 EMA. No training from scratch, no TuSimple training (TuSimple test clips are used only for evaluation, §7.3), no YOLOPv2. The only training is the 3-epoch CLRerNet fine-tune in §7.1, which starts from the released checkpoint.

Every number is measured in this project unless it is labelled *published*. Published values come from the papers or READMEs, were checked against the source text, and sit in their own columns.

---

## 1. Summary

| | SCNN | UFLDv2 ResNet34 | CLRerNet DLA34 EMA |
|---|---|---|---|
| CULane test F1, reproduced (official C++ evaluator) | **70.07** | **75.98** with TTA · **75.86** without | **81.55** |
| CULane test F1, *published* | *71.6* | *76.0* with TTA · *75.9* without | *81.55* (released checkpoint) |
| Full-prediction latency, P100, batch 1 (median) | 63.9 ms | 53.3 ms | 20.7 ms |
| Throughput from median | 15.7 FPS | 18.8 FPS | 48.3 FPS |
| Mean F1 over 15 image corruptions (clean = 100%) | 56.1 (80%) | 64.4 (85%) | 72.6 (89%) |

- UFLDv2 (both settings) and CLRerNet reproduce their published F1 within 0.05. SCNN is 1.53 below its paper, with most of the gap in the shadow (−5.5) and night (−4.5) scenarios.
- CLRerNet is the most accurate in every scenario, the fastest end to end, and the most robust to corruption. Rain is the worst corruption for all three models.
- **Corruption-augmented fine-tuning (§7.1–7.2, three seeds):** 3 epochs of CLRerNet training with rain / blur / JPEG / darkening augmentation raise mean corrupted F1 from 72.56 to 77.44 ± 0.11 (heavy rain 46.4 → 77.4). Clean test F1 falls from 81.55 to 81.23 ± 0.05; the same-schedule control (81.27 ± 0.10, 70.17 ± 0.15 corrupted) shows that this cost comes from fine-tuning itself, not from the corruptions. There is no gain on the held-out shadow corruption. A broad-augmentation arm with 7 other corruption types transfers partly: motion blur +4.3, JPEG +2.3, rain −1.2, against the control.
- Dashcam video: an aspect-preserving crop adapter, calibrated by vanishing-point matching, removes the 0.64 aspect distortion of the naive resize and reduces lane-count flicker for SCNN (291 → 213), UFLDv2 (23 → 12) and CLRerNet (137 → 118).
- **Cross-dataset (§7.3, TuSimple, 233-clip public subset, never trained on):** CLRerNet keeps the lead, with TuSimple F1 68.0 [95% CI 64.8–71.1] against UFLDv2's 65.2. SCNN finds lanes in only 4 of 233 frames, confirmed independently with the demo's own pipeline. The corruption fine-tune generalises **worse** than its same-schedule control: 58.2 ± 0.7 vs 62.5 ± 1.6 over 3 seeds, and both are below the released model. Robustness to synthetic corruptions did not transfer to a new dataset.
- **Tracker accuracy on labeled video (§9.3):** at real video rate (TuSimple, 20 fps), B2 raises CLRerNet's F1 by +3.4 [paired 95% CI 1.6–5.2]; for SCNN (+0.4) and UFLDv2 (+0.4) the change is not significant. At CULane's 1 annotated frame per second, the same tracker costs 16–20 F1 (9–12 without held lanes). Its smoothing and holds assume consecutive video frames.
- **CPU deployment (§6.1):** ONNX Runtime FP32 gives the same CULane test result as the GPU run, with identical TP/FP/FN, and is 1.4× faster than PyTorch on CPU. Post-training static INT8 (MinMax calibration) is not usable: F1 drops to 0 with every layer quantised and to 67.9 with the lane head in FP32. It is also no faster on the cluster's Broadwell Xeon, which has no VNNI instructions.
- **Downstream improvement:** a causal temporal lane tracker (B2) roughly halves lane re-starts for CLRerNet and SCNN and cuts frame-to-frame jitter for all three models, at < 1 ms/frame. For UFLDv2, which is already stable, it increases lane-count changes. The geometry-validity filter (B1/B3) was tested under three settings and is **not adopted**: it removed real lanes or had no measurable benefit.

---

## 2. Setup and provenance

| Model | Code (pinned commit) | Checkpoint | SHA-256 |
|---|---|---|---|
| SCNN | `delldu/SCNN` PyTorch port of the Torch7 original, `a6fadfbd9e593a28684dcaf29ca8ad425c1c700f` | `vgg_SCNN_DULR_w9.pth` (converted official weights) | `07c0c768dc435b8b162b58f42c7f686656133e0b13b8b4f110214245edb16671` |
| UFLDv2 | `cfzd/Ultra-Fast-Lane-Detection-v2`, `c903880678454dfd9b55a63022368db05c00bc6d` | `culane_res34.pth` | `1b4ca8385122f74c46a2e274950cdb5eb132ff97aa1ab1311550526d83caebf6` |
| CLRerNet | `hirotomusiker/CLRerNet`, `dae038f67da57e292e5293a68c9c1c2922de13c2` | `clrernet_culane_dla34_ema.pth` (release v0.1.0) | `cce8b552bc8301e31186da192fe4213043e9351b35651ef5d40014ce3040c787` |

- **Container:** `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel` (torch 2.3.1, CUDA 12.1), with one isolated venv per model. The container's torch is never replaced.
- **CLRerNet stack:** MMDetection 3.3.0, MMCV 2.1.0 built from source for sm_60, mmengine 0.10.4, numpy 1.26.4.
- **Environment manifest:** `reports/environment_manifest.json` holds the full `pip freeze` per venv, commits, checksums, the evaluator binary sha256 (`65b36a3d…`) and GPU details.
- **Checkpoint loading:** strict for SCNN and UFLDv2. CLRerNet's only missing keys are the training-only auxiliary segmentation branch (`bbox_head.seg_decoder.*`, `loss_seg.criterion.weight`), which `predict()` does not use.
- **Vendor code was never modified.** Every fix is environment-level and documented in `patches/`. All GPU work ran through HTCondor; nothing heavy ran on the login node.

## 3. Dataset audit

- **Splits:** CULane train/val/test lists have 88,880 / 9,675 / 34,680 entries; test entries are unique.
- **Test scenarios (frames):** normal 9,621 · crowd 8,113 · highlight 486 · shadow 930 · no-line 4,067 · arrow 890 · curve 422 · crossroad 3,122 · night 7,029.
- **Segmentation labels:** `laneseg_label_w16` and `laneseg_label_w16_test` are present.
- **Empty labels:** some test frames have empty `.lines.txt` files (no annotated lanes, e.g. the first highlight and night entries). This is normal for CULane.
- **`annotations_new`** (drivers 161, 182, 23; 98,555 files): **every file is byte-identical** to the dataset-root label at the same path (88,880 train + 9,675 val). The root already contains these annotations (`reports/annotations_new_comparison.json`).
- **Left untouched, not deleted:** an empty-looking nested folder `datasets/CULane/datasets/CULane`, and `culane.zip` (45.7 GB). CULane is non-commercial; nothing was copied or redistributed.

## 4. Smoke tests (Phase B)

**Test design.** A uniform 5-image GPU smoke (`scripts/smoke5.py`) runs each model on the first non-empty-label frame of normal, crowd, highlight, curve and night. A model passes when:
- the first image has lanes;
- at least 3 of 5 images have lanes;
- all coordinates are finite and inside the frame;
- 5 prediction files and 5 overlays are written;
- the official evaluator finds TP > 0.

**Results.** `PASS.txt` files are in `results/<model>/smoke5/`.
- **SCNN and CLRerNet:** both pass (job 73860).
- **UFLDv2:** passes in the corrected rerun (job 73914), with lanes per image of 3 / 3 / 0 / 4 / 0.

**Failed attempts, kept as evidence:**
- **73858:** frames with empty labels were picked.
- **73859:** the "every frame must have lanes" rule was too strict. The full-test predictions miss the same frames.

## 5. CULane accuracy

**Protocol.** Official CULane C++ evaluator (`evaluate.cpp` from the UFLDv2 vendor tree, unmodified): lane width 30 px, IoU 0.5, 1640×590, full 34,680-image test set. A GT-vs-GT self-check (must give F1 = 1) and a path preflight run first (`scripts/score_culane_official.sh`). All models' saved predictions are scored by this same evaluator.

### 5.1 Published vs reproduced (`reports/published_vs_reproduced.csv`)

| Model | Published F1 | Reproduced F1 | Δ | Precision | Recall | Jobs (inference / scoring) |
|---|---|---|---|---|---|---|
| SCNN | 71.6 | 70.07 | −1.53 | 70.82 | 69.33 | 73765 / 73835 |
| UFLDv2, author-default TTA | 76.0 | 75.98 | −0.02 | 77.27 | 74.74 | 73755 / 73836 |
| UFLDv2, no TTA | 75.9 | 75.86 | −0.04 | 77.13 | 74.63 | 73894 / 73905 |
| CLRerNet | 81.55 | 81.55 | ±0.00 | 87.79 | 76.14 | 73864 / 73878 |

- **UFLDv2:** the paper (arXiv 2206.07389, Table 7) lists ResNet-34 both without TTA (75.9) and with feature-level TTA (76.0). The repo's `configs/culane_res34.py` defaults to `tta=True`. TTA adds +0.12 F1 here and +0.1 in the paper.
- **CLRerNet, released checkpoint:** 81.55 matches the README. The paper's Table 2 reports a five-seed mean of 81.43 ± 0.14 for the same EMA setup.
- **CLRerNet, its own evaluator:** its Python `CULaneMetric` on the same predictions also gives 81.55 at IoU 0.5 (88.13 at 0.1, 65.02 at 0.75).

### 5.2 Per scenario: reproduced vs published (published values verified from the papers, `reports/published_by_scenario.csv`)

| Scenario | SCNN repr. | SCNN *pub.* | UFLDv2 repr. (TTA) | UFLDv2 *pub.* (TTA) | UFLDv2 repr. (no TTA) | UFLDv2 *pub.* (no TTA) | CLRerNet repr. | CLRerNet *pub.*† |
|---|---|---|---|---|---|---|---|---|
| Normal | 90.01 | *90.6* | 92.59 | *92.5* | 92.60 | *92.5* | **94.38** | *94.36* |
| Crowd | 68.68 | *69.7* | 74.84 | *74.8* | 74.88 | *74.9* | **80.80** | *80.62* |
| Highlight | 56.86 | *58.5* | 65.62 | *65.5* | 65.91 | *65.7* | **75.30** | *75.23* |
| Shadow | 61.39 | *66.9* | 75.46 | *75.5* | 75.16 | *75.3* | **84.55** | *84.35* |
| No line | 42.69 | *43.4* | 49.01 | *49.2* | 48.78 | *49.0* | **56.77** | *57.31* |
| Arrow | 83.71 | *84.1* | 88.20 | *88.8* | 88.12 | *88.5* | **91.06** | *91.17* |
| Curve | 62.79 | *64.4* | 69.76 | *70.1* | 69.29 | *70.2* | **78.88** | *79.11* |
| Crossroad (FP, lower is better) | 1,926 | *1,990* | 1,851 | *1,910* | 1,856 | *1,864* | **1,335** | *1,540* |
| Night | 61.65 | *66.1* | 70.90 | *70.8* | 70.42 | *70.6* | **76.87** | *76.92* |
| **Total** | **70.07** | *71.6* | **75.98** | *76.0* | **75.86** | *75.9* | **81.55** | *81.43*† |

Sources: SCNN, arXiv 1712.06080 Table 5; UFLDv2, arXiv 2206.07389 Table 7; CLRerNet, arXiv 2305.08366 Table 2. † CLRerNet's published per-scenario row is the **five-seed mean** of the EMA configuration, not the released checkpoint. Per-scenario values for the released checkpoint are not published, so its column is a reference, not a reproduction target.

### 5.3 Failure gallery (`results/failure_gallery/`)

**Method.** For each scenario, the gallery shows the 4 test images with the most errors summed over the three models. Each image appears as [SCNN | UFLDv2 | CLRerNet], with GT white, matched predictions green and unmatched predictions red. A fast half-resolution matcher ranks the images; it reads about 1 F1 high compared with the official evaluator (72.4 / 77.3 / 82.6 vs 70.1 / 76.0 / 81.6) and is used for ranking only.

**Main finding.** In the hardest no-line cases, CULane annotates lane positions where no paint is visible. All three models instead follow curbs and road edges, so they score zero TP. This is a large part of why no-line is the lowest-F1 scenario for every model.

## 6. Same-hardware speed (Phase D)

**Setup.** One P100, batch 1, FP32, cuDNN benchmark off. Each model gets the first 250 CULane test frames, preloaded so disk read and JPEG decode are excluded: 50 warmup frames, then 200 measured. CUDA is synchronised at every stage boundary. Per-frame samples are kept.

**Runs used.** SCNN and UFLDv2 come from quiet run 73930 (load average about 2). CLRerNet comes from the controlled run 73932 (see the note below the table). Full table: `reports/speed.csv`.

| Model | Network input | Forward only, median / p95 | Full prediction, median / p90 / p95 / mean | FPS (median) | Pre / post median | Peak GPU mem |
|---|---|---|---|---|---|---|
| SCNN | 800×288 | 51.0 / 54.0 ms | 63.9 / 66.5 / 69.8 / 64.7 ms | 15.7 | 5.9 / 6.8 ms | 208 MB |
| UFLDv2 (no TTA) | 1600×320 | 11.7 / 11.8 ms | 53.3 / 55.0 / 57.5 / 52.5 ms | 18.8 | 21.0 / 20.6 ms | 906 MB |
| CLRerNet | vendor val pipeline | 14.2 / 20.2 ms | 20.7 / 23.4 / 28.0 / 21.5 ms | 48.3 | 3.2 / 3.1 ms | 120 MB |

- **Full prediction:** preprocessing from an in-memory 1640×590 frame, then forward pass, then decode (for CLRerNet, NMS + lane decode) to source-frame coordinates.
- **UFLDv2's bottleneck is outside the network:** the network alone runs at 86 FPS. End-to-end time is dominated by the author's PIL resize (~21 ms) and the per-point Python decode of its row and column anchors (~21 ms). This measures the implementation, not the architecture.
- **CLRerNet and the cuBLAS setting:** `CUBLAS_WORKSPACE_CONFIG=:4096:8` is needed only for the deterministic `tools/test.py`; upstream's Dockerfile sets it. A controlled off/on/off A/B on one GPU (job 73932) shows it costs CLRerNet ~3.6 ms per forward pass and 24 MB: 20.7 ms full without it, 24.1 ms (41.5 FPS) with it. SCNN and UFLDv2 never had the setting, so the like-for-like figure above is without it.
- **Decode check:** every stage-split decode was verified against the vendor reference on frame 0 (max difference ≤ 0.001 px). The UFLDv2 reference is the author's complete row + column output.
- **Superseded runs:** 73862 measured UFLDv2 with the broken row-only decoder. 73915 was contended (load average 9–15). Both are kept and marked. No literature FPS is reported as a measurement.

### 6.1 CPU deployment: ONNX FP32 and INT8 (v5)

**Question.** How fast does CLRerNet run on a CPU, as in the public demo, and what does post-training quantisation cost?

**Design (fixed before any test-set scoring, `scripts/onnx_clrernet.py`).**
- The exported graph is exactly the network part timed in §6: backbone, neck and head up to the final refinement stage's raw lane tensors (opset 17). Preprocessing and decoding stay in Python and are shared by every variant: the vendor pipeline, then NMS through the exact CPU port `nms_cpu.py`, then the vendor CULane string writer.
- **Variants:**
  - `onnx_fp32`: ONNX Runtime 1.19.2, CPU.
  - `int8_all`: static INT8, QDQ, per-channel S8 weights, S8 activations, MinMax calibration on 300 CULane **train** images.
  - `int8_bbneck`: the same, but with the lane head left in FP32.
- **Checks:**
  - every variant against PyTorch on 50 validation images;
  - this writer path against the original GPU prediction files on 300 test images: 300 of 300 match, max 0.002 px.
- **Speed:** 200 validation images on one Xeon E5-2690 v4 (Broadwell), median.

| Variant | Lanes vs PyTorch (50 val images) | Forward, 1 thread | Forward, 4 threads | Model size | CULane test F1 |
|---|---|---|---|---|---|
| PyTorch FP32 (CPU) | reference | 772 ms | 224 ms | 63 MB | 81.55 (GPU run, §5) |
| ONNX FP32 | identical, max 0.003 px | **548 ms** | **156 ms** | 60.5 MB | **81.55**, TP/FP/FN identical to the GPU run |
| INT8, all layers | differ on 42 of 50 images | 605 ms | 251 ms | 15.6 MB | 0 (no true positive) |
| INT8, lane head in FP32 | differ on 43 of 50 images | 632 ms | 223 ms | 16.7 MB | 67.92 |

**Findings.**
- **ONNX FP32 is a free speed-up:** 1.41× on 1 thread and 1.44× on 4 threads.
  - The full test set gives the same 79,864 / 11,110 / 25,022 TP/FP/FN as the original GPU run.
  - Speeds are from the final export run (job 74381). The first run (74374) measured 807 / 562 ms and 234 / 152 ms, the same ratio.
- **Post-training INT8 is not usable here.**
  - **Speed:** Broadwell has no VNNI int8 instructions, so the QDQ graphs run no faster than FP32, though they are 3.6–3.9× smaller.
  - **Accuracy:** with every layer quantised, no predicted lane is correct (F1 0). With the lane head kept in FP32, F1 is still 67.92 (−13.6). So most of the damage happens in the quantised backbone and neck.
  - Better calibration (entropy or percentile ranges) or quantisation-aware training might recover accuracy. These were not tried: choosing among them would need a validation-based selection step that was not declared in advance.
- **Bug found and fixed:** the first INT8 run (`results/onnx/export_v1.json`) quantised after graph optimisation, which renamed every node. The lane-head exclusion therefore matched nothing, and the two INT8 variants were identical. The fix keeps the node names; it is a bug fix to the declared variant, not a new setting.

## 7. Corruption robustness (CULane test subset)

**Setup.** Every 12th image of each scenario list: 2,895 test images in the full test set's scenario proportions, crossroad included. Five corruptions × three severities:
- darkening ×0.6 / 0.4 / 0.25;
- motion blur 7 / 15 / 25 px;
- JPEG quality 30 / 15 / 5;
- 1 / 2 / 3 soft cast shadows on the road;
- 300 / 700 / 1,500 rain streaks.

**Controls.** Each corruption is seeded per image and condition. Byte-identical inputs across models were verified by checksum. The design was fixed before any result was seen.

**Scoring.** Same official evaluator; UFLDv2 without TTA.

**Sanity check.** Clean F1 on the subset is close to the full-test F1: 69.77 vs 70.07, 76.00 vs 75.86, and 81.74 vs 81.55.

| Condition | SCNN | UFLDv2 | CLRerNet |
|---|---|---|---|
| clean | 69.77 | 76.00 | 81.74 |
| dark ×0.6 / ×0.4 / ×0.25 | 69.45 / 68.15 / 64.05 | 75.34 / 73.49 / 70.45 | 81.63 / 81.35 / 80.01 |
| motion blur 7 / 15 / 25 px | 69.14 / 58.61 / 34.32 | 73.53 / 62.59 / 42.98 | 81.07 / 74.30 / 55.92 |
| JPEG q30 / q15 / q5 | 68.69 / 64.99 / 42.90 | 75.24 / 72.18 / **51.01** | 80.83 / 77.68 / 47.83 |
| shadow 1 / 2 / 3 | 69.08 / 68.46 / 67.08 | 75.59 / 75.35 / 73.91 | 81.43 / 81.32 / 81.00 |
| rain 300 / 700 / 1500 | 51.17 / 31.19 / 14.11 | 63.35 / 48.38 / 33.04 | 75.02 / 62.67 / 46.41 |
| **mean over 15 corruptions** | **56.09** | **64.43** | **72.56** |
| **retained share of clean F1** | **80.4%** | **84.8%** | **88.8%** |

- **Darkening and shadows:** almost harmless for CLRerNet (≤ 1.7 F1 lost).
- **Rain and strong motion blur:** the most damaging for every model. SCNN keeps only 14.1 F1 under heavy rain.
- **Heavy JPEG (q5):** the only condition where CLRerNet does not lead: UFLDv2 51.0 vs CLRerNet 47.8.
- Outputs: `reports/robustness.csv`, `reports/robustness_summary.json`, `results/robustness/full_20261119T154132Z/` (corruption examples in `*/corruption_examples*.jpg`).

### 7.1 Corruption-augmented fine-tuning of CLRerNet (v3)

**Question.** Does a short fine-tune with corruption augmentation make CLRerNet more robust, and what does it cost on clean images?

**Design (fixed before training, `patches/clrernet_finetune.md`).**
- Two runs from the released EMA checkpoint, identical except for one augmentation step:
  - **control:** the authors' DLA34-EMA training recipe, unchanged;
  - **corruption:** the same recipe plus one `OneOf` step (p = 0.5, applied before the crop): rain, motion blur 7–25 px, JPEG quality 5–40, or darkening (brightness −0.2 to −0.7).
- 3 epochs over the CULane train set (55,698 of 88,880 frames after the authors' frame-difference filter, threshold 15), batch 24, AdamW lr 6e-5 cosine to 6e-6, seed 0, deterministic. One P100 per run, about 3.6 h each (job 73992).
- The training corruptions are albumentations 0.4.6 implementations, **not** the code that generates the test corruptions. The test shadow corruption is held out entirely.
- **No checkpoint selection:** the epoch-3 checkpoint is used for both runs. Validation F1 was logged for monitoring only (control 82.57 / 82.38 / 82.47, corruption 82.29 / 82.22 / 81.83 at epochs 1–3). The test set was scored once (DAG 74045).

**Full CULane test (34,680 images, official evaluator):**

| | Original CLRerNet | Fine-tuned control | Fine-tuned with corruptions |
|---|---|---|---|
| Test F1 | 81.55 | 81.30 | 81.18 |
| Shadow scenario (real shadows) | 84.55 | 83.72 | 83.19 |
| Night scenario | 76.87 | 76.37 | 76.54 |

**Robustness subset (§7 setup, same 2,895 images and byte-identical corrupted inputs, verified by checksum):**

| Condition | Original | Control | Corruption |
|---|---|---|---|
| clean | 81.74 | 81.15 | 81.05 |
| dark ×0.6 / ×0.4 / ×0.25 | 81.63 / 81.35 / 80.01 | 81.42 / 80.79 / 79.32 | 81.22 / 80.88 / 79.99 |
| motion blur 7 / 15 / 25 px | 81.07 / 74.30 / 55.92 | 80.34 / 72.80 / 52.15 | 80.51 / 75.36 / **60.15** |
| JPEG q30 / q15 / q5 | 80.83 / 77.68 / 47.83 | 80.16 / 75.82 / 39.90 | 80.47 / 78.35 / **64.24** |
| shadow 1 / 2 / 3 *(held out)* | 81.43 / 81.32 / 81.00 | 80.93 / 80.86 / 80.52 | 80.94 / 80.61 / 80.64 |
| rain 300 / 700 / 1500 | 75.02 / 62.67 / 46.41 | 72.54 / 56.49 / 39.05 | **80.20 / 79.39 / 77.06** |
| **mean over 15 corruptions** | **72.56** | **70.21** | **77.33** |
| **retained share of clean F1** | **88.8%** | **86.5%** | **95.4%** |

**Findings.**
- **Robustness gain:** mean corrupted F1 rises from 72.56 to 77.33 (+4.8) against the original, and +7.1 against the control.
  - The largest gains are on the hardest conditions: heavy rain 46.4 → 77.1, JPEG q5 47.8 → 64.2, blur 25 px 55.9 → 60.2.
  - CLRerNet now leads UFLDv2 at JPEG q5 (64.2 vs 51.0), the one condition it lost in §7.
- **Clean cost:** test F1 falls by 0.37 against the original. Only 0.12 of that is attributable to the corruptions; the other 0.25 also appears in the control, so it comes from the extra training itself.
- **No transfer to the unseen corruption:** synthetic shadow (80.73 vs control 80.77) and darkening (already near-harmless) show no gain. So the evidence supports robustness to the **corruption families used in training**, implemented differently from the test code. It does not show generalisation to new kinds of corruption.
- **The control lost some robustness** (72.56 → 70.21) while keeping clean accuracy. A plausible reason is that the released checkpoint is an EMA of a longer run and 3 epochs of further training move it slightly; this was not investigated.
- **Seeds:** this section's runs are seed 0. §7.2 adds seeds 1 and 2 for both arms; the gains hold, and the clean-cost split above is revised there.
- Outputs: `reports/finetune_comparison.csv`, `reports/robustness.csv` (5 models), `results/finetune/clrernet_ft_{control,corrupt}_*-73992/`, `results/clrernet_ft_{control,corrupt}/official_eval/`. The original three-model robustness files are kept as `*_3models.*`.

### 7.2 Overnight round: three seeds, and broad augmentation (v4)

**Design (fixed before training, `patches/clrernet_finetune.md`).**
- **Seeds:** control and corruption were each trained again with seeds 1 and 2, with identical configs (job 74187). Together with §7.1's seed-0 runs, each arm now has 3 runs.
- **Broad arm (one seed):** the control recipe plus one `OneOf` step (p = 0.5) drawn from 7 corruption types, none of them among the 5 test corruptions: fog, snow, sun flare, Gaussian noise, ISO sensor noise, contrast reduction and colour shift.
  - The question: does diverse augmentation transfer to corruption types never added in training?
  - Caveat recorded in advance: the authors' recipe, used by every arm including the original, already has mild motion/median blur (≤ 5 px at 800×320, p = 0.2) and small brightness/contrast/HSV jitter.
- **Same protocol as §7.1:** epoch-3 checkpoint, no selection, test scored once (DAG 74188). All 10 models received byte-identical corrupted inputs, verified by checksum.

**Results: mean ± standard deviation over seeds 0, 1 and 2 (`reports/finetune_seeds.csv`).**

| | Original | Control (3 seeds) | Corruption (3 seeds) | Broad (1 seed) |
|---|---|---|---|---|
| CULane test F1 (34,680 images) | 81.55 | 81.27 ± 0.10 | 81.23 ± 0.05 | 81.32 |
| Robustness subset, clean | 81.74 | 81.22 ± 0.09 | 81.09 ± 0.03 | 81.28 |
| **Mean over 15 corruptions** | 72.56 | 70.17 ± 0.15 | **77.44 ± 0.11** | 71.32 |
| Share of clean F1 kept | 88.8% | 86.4% ± 0.2 | **95.5% ± 0.1** | 87.8% |
| Rain, mean of 3 severities | 61.37 | 55.93 ± 0.23 | **79.08 ± 0.19** | 54.74 |
| JPEG, mean of 3 | 68.78 | 65.08 ± 0.33 | **74.43 ± 0.10** | 67.34 |
| Motion blur, mean of 3 | 70.43 | 68.53 ± 0.11 | 72.18 ± 0.20 | **72.87** |
| Darkening, mean of 3 | 81.00 | 80.51 ± 0.11 | 80.77 ± 0.08 | 80.76 |
| Shadow, mean of 3 | 81.25 | 80.78 ± 0.06 | 80.73 ± 0.04 | 80.88 |
| Heavy rain (1,500 streaks) | 46.41 | 38.69 ± 0.33 | **77.38 ± 0.32** | 37.44 |
| JPEG q5 | 47.83 | 39.18 ± 0.78 | **64.15 ± 0.17** | 45.37 |
| Motion blur 25 px | 55.92 | 52.18 ± 0.08 | 60.55 ± 0.59 | **62.93** |

**Findings.**
- **The corruption-training gain is consistent across seeds.**
  - Mean corrupted F1 is 77.44 ± 0.11, against 70.17 ± 0.15 for the control (+7.3) and 72.56 for the original (+4.9).
  - The seed spread is at most 0.8 F1 on any metric, so §7.1's single-seed conclusion stands.
- **The clean cost comes from fine-tuning, not from the corruptions.**
  - Test F1: corruption 81.23 ± 0.05 vs control 81.27 ± 0.10, a difference of −0.04, well within the spread.
  - Both are about 0.3 below the original. §7.1's estimate that the corruptions themselves cost 0.12 was single-seed noise.
- **Fine-tuning without corruptions consistently costs robustness:** the control loses 2.4 ± 0.15 mean corrupted F1 against the original in all three seeds. This is now a reproducible effect. Its cause was not investigated.
- **Broad augmentation transfers only partly to unseen corruption types** (single seed; differences of 1 F1 or more are well beyond the 0.1–0.3 seed spread above):
  - **Motion blur:** +4.3 mean against the control; 62.9 at 25 px, the best of all models, even above the arm trained on motion blur (60.6).
  - **JPEG:** +2.3 mean against the control.
  - **Darkening and shadow:** no meaningful change.
  - **Rain:** −1.2 mean against the control.
  - Overall, mean corrupted F1 is 71.32: +1.2 over the control but −1.2 under the original, because the fine-tuning loss above outweighs the transfer.
  - **Conclusion:** targeted augmentation works strongly for the corruption families it contains. Broad augmentation with different families gives some transfer, to blur and compression, but not a general robustness gain.
- Outputs: `reports/finetune_seeds.csv`, `reports/robustness.csv` (10 models), `results/finetune/*-74187/`, `results/clrernet_ft_*_s{1,2}/` and `results/clrernet_ft_broad/official_eval/`. The 5-model robustness files are kept as `*_5models.*`.

### 7.3 Cross-dataset test: TuSimple (v5)

**Question.** How do the CULane models, and the fine-tunes from §7.1–7.2, do on a dataset none of them was trained on?

**Data.** The TuSimple lane test set via the ungated Hugging Face mirror `dotblue/TuSimple` (Apache-2.0, pinned commit). The mirror holds a **10% sample**: 233 of the 2,782 test clips are complete (all 20 frames), and only those are used. Downloaded to the cluster only and never redistributed. Training on TuSimple is out of scope; nothing was trained or tuned on it.

**Protocol, fixed before any TuSimple result was seen (`scripts/tusimple_eval.py`).**
- **Adapter:** the whole 1280×720 frame is resized to the 1640×590 CULane canvas, as in the demo's "squash" option. Then each model's verified inference path runs (the `benchmark_speed.py` builders).
- **Scored rows:** only label rows at y ≥ 330, for ground truth and predictions alike.
  - y = 330 is CLRerNet's crop line (cut_height 270 on the CULane canvas) mapped back, i.e. the region every model sees.
  - Ground-truth lanes left with fewer than 2 points are dropped.
- **Metrics:**
  - the official TuSimple metric (`LaneEval`, re-implemented line by line, with unit tests): accuracy, FP rate and FN rate;
  - the TuSimple F1 used by CLRNet/CLRerNet, 2(1−FP)(1−FN)/((1−FP)+(1−FN)), with 95% clip-level bootstrap CIs;
  - for reference, the official CULane C++ evaluator on the same lanes at 1280×720.

| Model | Accuracy | FP rate | FN rate | **TuSimple F1** [95% CI] | CULane-style F1 |
|---|---|---|---|---|---|
| SCNN | 0.9% | 1.7% | 100% | 0.0 | — |
| UFLDv2 (no TTA) | 80.3% | 29.8% | 39.2% | 65.2 [61.3–69.1] | 58.8 |
| **CLRerNet, released** | 75.6% | 19.6% | 41.1% | **68.0 [64.8–71.1]** | **60.1** |
| CLRerNet, control fine-tune (3 seeds) | 67.0 ± 1.7% | 16.9 ± 2.1% | 49.9 ± 1.8% | 62.5 ± 1.6 | 56.2 ± 1.5 |
| CLRerNet, corruption fine-tune (3 seeds) | 64.3 ± 1.8% | 19.9 ± 0.9% | 54.3 ± 1.1% | 58.2 ± 0.7 | 52.3 ± 0.9 |
| CLRerNet, broad fine-tune (1 seed) | 72.3% | 19.5% | 45.9% | 64.7 [61.2–68.0] | 56.9 |

**Findings.**
- **Every model loses a lot on unseen data, mostly through missed lanes.** CLRerNet scores 68.0 here against 81.55 on CULane, with an FN rate of 41%. CULane-trained detectors are known to transfer imperfectly; these numbers are not comparable to TuSimple-trained models (~97 F1).
- **SCNN fails almost completely:** it outputs 8 lanes in 233 frames. To rule out a scoring bug, the same clips were run through the demo's separate pipeline, which also shows 0 SCNN lanes where CLRerNet finds them (`assets/tusimple_crossdataset.gif`). SCNN loads and works on CULane and on the demo's sample clip, so this is a generalisation failure, not a loading error.
- **Fine-tuning on CULane costs cross-dataset accuracy.** The control loses 5.5 F1 against the released model, which is the EMA of a longer run. This mirrors the control's loss of corruption robustness in §7.2.
- **The corruption fine-tune generalises worse than its control:** −4.25 F1, in all three seed pairs (58.7 / 57.4 / 58.6 vs 64.0 / 60.8 / 62.6).
  - So the large §7.2 gains on synthetic rain, blur and JPEG do **not** carry over to a different camera, road and annotation style.
  - The broad arm (64.7, one seed) sits between control and released.
- Outputs: `reports/tusimple_cross_dataset.csv`, `results/tusimple/<model>/`.

## 8. Dashcam baseline (Phase E)

**Video:** `vendors/ufldv2/example.mp4`, the UFLDv2 repo's demo clip (upstream PR #115; original footage source undocumented). It is 1280×720, 25 fps, 1,260 frames, sha256 `65194992…7d5f`. **No labels: nothing here is an accuracy measurement.**

**Two adapters, each identical for all models:**
1. **Squash** (v1): resize the whole 16:9 frame to 1640×590. Aspect distortion factor 0.64.
2. **Crop** (v2): a full-width 1280×460 band (rows 203–663) resized to 1640×590. Distortion factor 1.001. The band was placed by matching vanishing points, not by looking at results:
   - CULane validation GT lanes meet at row 277/590 (0.47 of the height);
   - the video's baseline predicted lanes meet at row 419 (IQR 415–423);
   - so the band puts the video's vanishing row at the same relative height (`results/adapter_v2/crop_calibration.json`).

   Nothing is predicted in the bottom 57 rows (mostly hood) or the sky above row 203.

| Model | Adapter | Frames with ≥ 1 lane | Mean lanes/frame | Lane-count changes |
|---|---|---|---|---|
| SCNN | squash / crop | 97.1% / 97.4% | 3.33 / 3.48 | 291 / **213** |
| UFLDv2 | squash / crop | 99.9% / 100% | 3.95 / 3.99 | 23 / **12** |
| CLRerNet | squash / crop | 100% / 100% | 3.79 / 3.87 | 137 / **118** |

**Video end-to-end throughput** (decode from disk + inference + drawing + mp4 encode), clean squash run 73882: SCNN 13.8 FPS, CLRerNet 26.3 FPS. The UFLDv2 and crop video runs shared the node with other jobs, so their timings are not reported.

**Visual review** (6 contact-sheet frames per adapter, not systematic):
- **SCNN:**
  - The crop adapter removes a zig-zag on frame 0 and finds a lane on the bright concrete bridge (frame 1007).
  - It adds a short false centre curve.
- **UFLDv2:**
  - Ego lanes are clean.
  - The outer left lane, from the column branch, is visibly jagged.
  - The far-right lane is drawn across occluding cars (755, 1007).
- **CLRerNet:**
  - Four lanes, mostly on markings.
  - A false curved lane on frame 0.
  - The far-right lane drawn across a car (755, 1007).

Outputs: `results/dashcam/<model>/<run>/{overlay.mp4, frames.jsonl, report.json}`; `results/comparison/run_20261119T144052Z_*-73882_v3` (squash) and `run_20261119T154149Z_*-73901_crop_v3` (crop), each with a contact sheet, side-by-side video and summary.

## 9. Downstream improvement and ablation (Phase F)

**Design** (`scripts/temporal_ablation.py`). The postprocessing is model-agnostic and strictly causal. It runs offline on each model's saved baseline lanes, so the checkpoint and threshold are the baseline's. All tracker parameters were fixed in advance, with no search. The clip is reported as two halves (frames 0–629 and 630–1259); nothing was tuned on either.

**Variants:**
- **B0:** raw baseline.
- **B1:** geometric validity filter (minimum extent, maximum curvature, maximum slope, crossing pairs).
- **B2:** temporal tracker:
  - Hungarian matching on median lateral distance at common rows, gate 40 px;
  - EMA smoothing, α = 0.5;
  - a new lane is shown after 2 hits;
  - a lost lane is held for ≤ 3 frames (120 ms), drawn dashed and counted separately.
- **B3:** B1 + B2.

### 9.1 Accepted: B2 temporal tracking — crop-adapter baseline (`reports/ablation_temporal_crop_adapter.csv`)

Values are first half / second half; jitter is in CULane-canvas px (× 0.78 for source px).

| Model | Variant | Lane re-starts per 100 fr | Lane-count changes | Jitter y = 500 median | Jitter y = 500 p95 | Held-lane share | Cost |
|---|---|---|---|---|---|---|---|
| CLRerNet | B0 | 20.4 / 8.7 | 70 / 48 | 1.67 / 1.97 | 10.5 / 9.8 | — | — |
| CLRerNet | **B2** | **8.9 / 3.2** | 89 / **39** | **1.09 / 1.24** | **6.7 / 5.9** | 9% / 4% | 0.8 ms |
| SCNN | B0 | 10.5 / 18.9 | 65 / 148 | 2.19 / 2.79 | 18.9 / 14.6 | — | — |
| SCNN | **B2** | **5.7 / 6.4** | **55 / 68** | **1.57 / 1.58** | **15.3 / 7.8** | 6% / 9% | 0.6 ms |
| UFLDv2 | B0 | 4.8 / 4.0 | 6 / 6 | 1.28 / 1.41 | 20.6 / 17.8 | — | — |
| UFLDv2 | B2 | **4.3 / 2.4** | 45 / 30 (worse) | **1.18 / 1.18** | **14.0 / 9.5** | 4% / 3% | 0.7 ms |

- **Effect:** B2 halves lane re-starts for CLRerNet and SCNN and reduces jitter for all three models. Smoothed lanes stay ~1.2–1.6 px (median) from the current observation, so lag is small. Observed availability stays within 1 point of B0 (≥ 94% for every model and half).
- **Not a uniform win:**
  - Lane-count changes rise for UFLDv2, which was already very stable, and for CLRerNet's first half. The confirm and hold logic adds toggling.
  - In one reviewed frame, a held lane carried a false centre-lane detection for up to 3 frames.
- **Squash baseline (`reports/ablation_temporal.csv`):** B2 shows the same pattern, e.g. CLRerNet re-starts 19.4 → 9.1 / 6.7 → 4.1 and SCNN 17.0 → 8.1 / 32.0 → 8.3.

### 9.3 Tracker accuracy on labeled video (v5)

**Question.** §9.1 showed that B2 makes lanes steadier on unlabeled video. Does it make them more *accurate*?

**Design.** No parameter changed (`P` in `temporal_ablation.py`). Two labeled sources:
- **CULane test:** 419 clips with frames 1 s apart, all 34,680 frames scored by the official evaluator (`scripts/tracker_culane.py`).
- **TuSimple subset:** 233 clips at 20 fps, run in order over frames 1–20 and scored on the labeled frame 20 (§7.3 protocol).

**Variants, all declared before scoring:**
- **B0r:** raw predictions converted to the tracker's lane format (rows 250–580 every 10 px on the CULane canvas), with no tracking. This is the fair control, since B2 outputs that format.
- **B2:** the tracker's output including held lanes, as the demo draws them.
- **B2obs:** the tracker's output without held lanes.

| Data | Model | B0r F1 | B2 F1 | B2obs F1 |
|---|---|---|---|---|
| CULane test (1 frame/s) | CLRerNet | 81.48 | 61.63 | 69.00 |
| | SCNN | 69.76 | 53.50 | 61.16 |
| | UFLDv2 (no TTA) | 75.13 | 57.31 | 65.56 |
| TuSimple (20 fps) | CLRerNet | 67.98 | **71.40** | 67.80 |
| | SCNN | 0.00 | 0.43 | 0.00 |
| | UFLDv2 (no TTA) | 65.40 | 65.82 | 66.32 |

Paired clip bootstrap, B2 − B0r on TuSimple (2,000 resamples, `reports/v5_comparisons.csv`):
- CLRerNet **+3.42 [1.60, 5.23]**;
- SCNN +0.43 [0.00, 1.07];
- UFLDv2 +0.42 [−1.32, 2.20].

**Findings.**
- **At real video rate, B2 improves CLRerNet's accuracy significantly.** The gain is in recall: the FN rate drops from 41.1% to 36.7% while FP also drops. That is the held lanes bridging short missed detections. For UFLDv2 and SCNN the change is not significant.
- **At 1 frame per second, B2 is harmful** (−16 to −20 F1; −9 to −12 for B2obs). Two causes:
  - its EMA averages lanes 1 s apart;
  - "confirm after 2 hits" drops every lane in the first frame of each clip, and a held lane persists up to 3 s.

  This is a design boundary, not a bug: B2 assumes consecutive video frames, and the demo uses it only on video, at ≤ 10 fps after resampling.
- The format conversion alone (raw → B0r) costs 0.07 (CLRerNet) to 0.73 (UFLDv2) CULane F1. That is why B0r, not the raw score, is the control.
- This partly addresses §11's "no second video" limitation. It is labeled, but TuSimple is a different dataset, and the result rests on 233 clips.

### 9.2 Not adopted: B1 / B3 geometry filter (three settings)

**Calibration** (prediction-calibrated, CULane val; `reports/b1_calibration_pred_val.json`):
- Each model's thresholds come from its own validation-set lanes that match the GT (IoU ≥ 0.5): minimum extent at the 0.5th percentile, maximum curvature and slope at the 99.5th percentile.
- **Cost on real lanes:** 1.06–1.15% of true lanes rejected (by construction).
- **Benefit on false lanes:** caught 6.97% for CLRerNet, 15.6% for UFLDv2 and 16.0% for SCNN.

**Lanes rejected on the whole clip:**

| Setting | SCNN | UFLDv2 | CLRerNet |
|---|---|---|---|
| Squash adapter, thresholds from val GT shapes (filter rejects 1.07% of GT lanes) | 1,626 | 3,271 | 779 |
| Squash adapter, thresholds from each model's val predictions | 1,259 | 1,640 | 1,143 |
| **Crop adapter**, thresholds from each model's val predictions | 425 | 107 | **16** |

- **Squash adapter:** the filter mostly removed real lanes. In a visual review of 12 sampled CLRerNet rejection frames, 10 were real far-right markings and 2 were the right marking across an occluding car. For UFLDv2, lanes per frame fell from 3.97 to 1.3–2.5.
  - **Root cause:** the 0.64 squash makes far lanes flatter and shorter than any correct CULane lane.
- **Crop adapter:** with the distortion removed, CLRerNet rejections fall to 16 in total. Those are an ambiguous line along the barrier base, with no demonstrable benefit.
  - For SCNN and UFLDv2 the remaining rejections still raise flicker: SCNN lane-count changes 65 → 181 and UFLDv2 6 → 97 in the first half. Their jagged decodes trip the curvature rule.
- **Decision:** B2 (tracking + smoothing, no geometry filter) is the accepted improvement. Thresholds were never tuned on the video.

## 10. Problems found and fixed (reproducibility notes)

| Issue | Evidence | Fix | Note |
|---|---|---|---|
| UFLDv2 first scoring run invalid | 73767: tp = fp = fn = 0 | evaluator path concatenation needs a trailing `/`; self-check and preflight added | marked "do not use" |
| CLRerNet `np.sctypes` | 73764 | pin numpy 1.26.4 | `patches/clrernet_numpy_pin.md` |
| CLRerNet `No module named 'libs'` | 73838 | PYTHONPATH as in upstream Dockerfile | `patches/clrernet_pythonpath.md` |
| CLRerNet full test hung, 0 outputs | 73848: `AF_UNIX path too long` | my long TMPDIR broke DataLoader sockets → short `/tmp` dir, then copy | `patches/clrernet_full_tmpdir.md` |
| CLRerNet deterministic cuBLAS error | 73863 | `CUBLAS_WORKSPACE_CONFIG` as upstream (its speed cost quantified in §6) | `patches/clrernet_pythonpath.md` |
| UFLDv2 TTA mislabelled in report | config check | CSV corrected; no-TTA run added | `STATUS.md` |
| **My UFLDv2 in-memory decoder** omitted the column-anchor lanes 0, 3 (author `generate_lines_col_local`) | robustness clean F1 57.4 vs 75.86 | column branch added; verified 50/50 images exact (≤ 0.001 px) against the author's complete output (73914); all affected UFLDv2 speed, video, calibration, ablation and robustness results rerun | `patches/ufldv2_decoder_fix.md`; old outputs `*_INVALID_rowonly_decoder` |
| Crop adapter ignored for UFLDv2 | 73901: report showed squash | crop argument now read before `build_ufld` rewrites `sys.argv` | `*_INVALID_no_crop_applied` |
| Speed rerun contended | 73915: load average 9–15 | quiet rerun 73930; duplicate 73931 removed | `INVALID_CONTENDED.txt` |
| Robustness split jobs did nothing | 73903: `;` vs `,` delimiter | parser fixed; unknown condition names now abort | removed within a minute |
| int8_bbneck identical to int8_all | 74374: 0 head nodes excluded | quantise from a graph that keeps exported node names (`skip_optimization`) | v1 kept as `results/onnx/export_v1.json` |
| CPU NMS port crashed on INT8 outputs | 74379: empty slice for a negative lane start | index the flat buffer like the kernel's pointer; unit test shows identical results for valid outputs | `tests/test_core.py` |
| TuSimple CULane-style F1 counted phantom lanes | lane files with no lanes held one newline | empty files, as the official writer; rescored from saved predictions (74380); TuSimple metric unchanged (asserted) | v1 scores kept as `culane_eval_v1_*.txt` |
| TuSimple mirror incomplete | 404 for most clips | use the 233 complete test clips; documented in §7.3 | `download_tusimple.sh` |
| Public demo had no ffmpeg | live check: videos failed | Streamlit Cloud reads `packages.txt` only at the repo root; root copy + `imageio-ffmpeg` fallback | caught by testing the live site |
| CLRerNet robustness exit 2 after finishing | 73900.2 | `gpu_entry.sh` edited while running (bash reads scripts incrementally); results complete | lesson: never edit a running entry script |
| UFLDv2 vendor self-eval after inference | 73894: `./evaluation/culane/evaluate` not found | not needed; scored with the shared evaluator | predictions complete |

All failed runs, logs and earlier results are preserved. Edited scripts keep `.before_*` / `.v*` backups.

## 11. Limitations and open items

- **SCNN gap (−1.53):** concentrated in shadow (−5.5) and night (−4.5). It is attributed to Python vs MATLAB probability-map decoding in the port; not proven.
- **Dashcam evidence is weak by construction:**
  - one unlabeled clip of unknown origin, analysed as two halves of one recording;
  - stability metrics are proxies, not accuracy;
  - visual reviews covered small samples;
  - the clip has now been seen many times, so it must not serve as a final held-out test for further changes.
- **No second dashcam video.** B2's accuracy is now measured on labeled video (§9.3, TuSimple 20 fps, 233 clips), but not on a second recording from the dashcam setting.
- **Robustness:**
  - the corruptions are synthetic;
  - images are decoded with OpenCV for all models; for UFLDv2 this differs from the author's PIL loader by a median 0.08 px, but occasionally flips an anchor.
- **Fine-tune (§7.1–7.2):**
  - three seeds for control and corruption, but one seed for the broad arm;
  - the training and test corruptions share families (different implementations); held-out corruptions show only partial transfer;
  - the fine-tuned model was not re-timed or run on the dashcam clip.
- **Unverified video detail:** the crop band position comes from predicted-lane vanishing points of the baseline. It is label-free but not verified against camera calibration.
- **Cross-dataset (§7.3):** only a 10% public subset of TuSimple (233 clips) was available without an account, so CIs are wide (±3 F1). A fixed squash adapter and the y ≥ 330 row cut are this project's protocol, not the official TuSimple setting.
- **INT8 (§6.1):** measured on one CPU generation without VNNI. A modern CPU may show a speed-up; this was not tested.
- **Not done:** a separate `literature` FPS table, and video throughput on more than one clip.

**Suggested next steps:**
1. Record or upload a second dashcam clip and rerun the crop baseline plus B2 on it as an independent test.
2. Re-test B1 there with the crop adapter. Only CLRerNet is a plausible candidate.

## 12. How to reproduce (from `Lane_Detection/benchmark_culane`, all through HTCondor)

```bash
condor_submit jobs/smoke5/smoke5.sub                                   # 5-image smoke, 3 models
condor_submit jobs/pretrained_20261119_025415_bff1e4/clrernet_full.sub  # CLRerNet full test (author tools/test.py)
condor_submit jobs/extras/ufldv2_full_notta.sub                         # UFLDv2 full test without TTA
condor_submit jobs/eval/{scnn,ufldv2,ufldv2_notta,clrernet}_official_eval.sub   # official C++ scoring
condor_submit jobs/speed/speed_all.sub                                  # same-GPU latency (submit when nothing else runs)
condor_submit jobs/speed/cublas_ab.sub                                  # CLRerNet cuBLAS A/B
condor_submit jobs/robust/robust_full.sub                               # robustness inference (or robust_split.sub)
condor_submit jobs/robust/score_full.sub                                # robustness scoring
condor_submit jobs/extras/estimate_vp.sub                               # crop-band calibration
condor_submit jobs/dashcam/dashcam_all.sub                              # dashcam, squash adapter
condor_submit jobs/dashcam/dashcam_all_crop.sub                         # dashcam, crop adapter
condor_submit jobs/b1_recal/valpred.sub                                 # val predictions for filter calibration
condor_submit jobs/extras/followup_ufld.sub                             # calibration + comparisons + ablations + rescoring
condor_submit jobs/extras/gallery.sub                                   # failure gallery
condor_submit jobs/extras/cpu_housekeeping.sub                          # environment manifest + annotations_new check
condor_submit jobs/finetune/frame_diffs_chunks.sub                      # train-set frame differences, 8 chunks (then frame_diffs_merge.sub)
condor_submit jobs/finetune/ft_full.sub                                 # CLRerNet fine-tunes: control + corruption (job 73992)
condor_submit_dag jobs/ft_dag/ft_eval.dag                               # fine-tune test + robustness inference, then scoring
condor_submit jobs/finetune/ft_overnight.sub                            # seeds 1-2 for both arms + broad arm (job 74187)
bash scripts/make_ft2_dag.sh <ClusterId> && condor_submit_dag jobs/ft_dag2/ft2_eval.dag   # their evaluation + parallel scoring
condor_submit jobs/tracker_culane/tracker_culane.sub                    # v5: tracker accuracy on CULane test clips
condor_submit jobs/tusimple/download.sub && condor_submit jobs/tusimple/eval.sub   # v5: TuSimple subset, 10 models
condor_submit jobs/onnx/onnx_export.sub && condor_submit jobs/onnx/onnx_infer.sub  # v5: ONNX export, checks, speed, full test
python3 scripts/summarize_v5.py --print                                 # v5 tables (login node, standard library)
```

SCNN and UFLDv2 full inference use `scripts/gpu_entry.sh <model> full` (jobs 73765, 73755). Every job writes `.out`, `.err` and `.log` under `logs/` and its results to a timestamped run directory.

## 13. File index

| What | Where |
|---|---|
| Chronological status log with every job id | `STATUS.md` |
| Accuracy | `reports/published_vs_reproduced.csv`, `reports/culane_by_scenario.csv`, `reports/published_by_scenario.csv`, `results/<model>/official_eval/*/official_eval.json` |
| Failure gallery | `results/failure_gallery/`, `reports/failure_gallery_summary.json` |
| Speed | `reports/speed.csv`, `results/speed/run_*-73930/`, `results/speed/cublas_ab_73932/` |
| Robustness | `reports/robustness.csv`, `reports/robustness_summary.json`, `results/robustness/full_20261119T154132Z/` |
| Fine-tuning | `reports/finetune_seeds.csv`, `reports/finetune_comparison.csv`, `configs_ft/`, `results/finetune/`, `results/clrernet_ft_{control,corrupt}/official_eval/`, `patches/clrernet_finetune.md` |
| Dashcam | `results/dashcam/`, `results/comparison/*_v3/`, `results/adapter_v2/crop_calibration.json` |
| Ablation | `reports/ablation_temporal_crop_adapter.csv`, `reports/ablation_temporal.csv`, `reports/ablation_temporal_v2_predcalib.csv`, `reports/b1_calibration_pred_val.json`, `results/ablation/fix_*_73921/` |
| Environment and dataset | `reports/environment_manifest.json`, `reports/annotations_new_comparison.json` |
| Patches and rationale | `patches/*.md` |
| v5: cross-dataset (TuSimple) | `reports/tusimple_cross_dataset.csv`, `results/tusimple/`, `assets/tusimple_crossdataset.gif` |
| v5: tracker on labeled video | `reports/tracker_labeled.csv`, `reports/v5_comparisons.csv`, `results/tracker_culane/` |
| v5: ONNX / INT8 | `reports/onnx_clrernet.csv`, `results/onnx/export.json`, `results/onnx_eval/` |
| v5: demo and CI | `streamlit_cloud/`, `.github/workflows/ci.yml`, `tests/test_core.py` |
