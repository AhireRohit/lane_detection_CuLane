# Reproducing every result

This guide takes a fresh clone to every number in [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md). Run the steps in order. Each step lists the command, roughly how long it takes on one Tesla P100, and what you should see.

## 0. What you need

- **An HTCondor cluster with the Docker universe and NVIDIA GPUs**, plus a filesystem shared between the submit node and the GPU nodes. Everything here was run on P100-16GB GPUs with the image `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel`.
- **Site-specific lines.** Every `.sub` file contains three lines specific to the Saarland cluster; change them for your site:
  - `requirements = UidDomain == "cs.uni-saarland.de"`
  - `+WantGPUHomeMounted = true`
  - `+WantScratchMounted = true`
- **Internet access from the GPU jobs** for the first run of each model. The jobs `pip install` each model's dependencies into its own venv. CLRerNet compiles MMCV 2.1.0 and its own NMS extension for your GPU, which takes about 30–60 minutes once. UFLDv2's unmodified code also downloads ImageNet ResNet-34 weights when it builds the model; the CULane checkpoint then replaces them.
- **The CULane dataset** from its [official page](https://xingangpan.github.io/projects/CULane.html). Its licence is non-commercial, so it is not included here.
- **Folder layout.** This repository must live inside a `Lane_Detection/` folder:
  ```
  Lane_Detection/
  ├── benchmark_culane/        ← this repository
  └── datasets/CULane/         ← list/, driver_*/, laneseg_label_w16/ ...
  ```

## 1. Setup (login node, light work only)

```bash
git clone <this repo> Lane_Detection/benchmark_culane
cd Lane_Detection/benchmark_culane
./setup_paths.sh /absolute/path/to/Lane_Detection   # fills /path/to/Lane_Detection placeholders in configs, scripts and jobs
bash scripts/bootstrap.sh                          # model repos at the pinned commits, dataset link, dataset audit
bash scripts/download_weights.sh                   # author checkpoints, then a sha256 check against config/checkpoints.sha256
```

**Expected output:**
- `bootstrap.sh` ends with `BOOTSTRAP OK`, after printing `OK scnn a6fadfb…`, `OK ufldv2 c903880…` and `OK clrernet dae038f…`. It stops if any model repo is at a different commit or has local changes. The vendor code is never modified.
- `download_weights.sh` prints `OK` for all three checkpoints. Any mismatch means you don't have the checkpoints used in the report.
- The dataset audit (`results/audit.json`) should show 34,680 test, 88,880 train and 9,675 val entries.

## 2. Reproduce the accuracy of the three pretrained models

| Step | Command | Time | Expected |
|---|---|---|---|
| Smoke test | `condor_submit jobs/smoke5/smoke5.sub` | ~1 h the first time (builds the venvs) | `PASS` for all three models on 5 test images |
| Full test inference | `condor_submit jobs/full/full_all.sub` | 1–3 h per model, run in parallel | predictions in `results/<model>/full/` |
| UFLDv2 without TTA | `condor_submit jobs/extras/ufldv2_full_notta.sub` | ~2 h | `results/ufldv2/full_notta/` |
| Official scoring | `condor_submit jobs/eval/scnn_official_eval.sub` (and the `ufldv2_`, `ufldv2_notta_`, `clrernet_` versions) | ~20 min each, on CPU | `results/<model>/official_eval/*/all_test.txt` |

**Expected F1** (official C++ evaluator, IoU 0.5): SCNN **70.07**, UFLDv2 **75.98** with TTA and **75.86** without, CLRerNet **81.55**. Per-scenario values are in `reports/culane_by_scenario.csv`. Before scoring, a ground-truth-against-itself check must give F1 = 1.

## 3. Speed (P100, batch 1)

Submit these when nothing else is running on the node, because contention distorts the timings (see `patches/speed.md`).

```bash
condor_submit jobs/speed/speed_all.sub     # all three models, same GPU, 50 warmup + 200 timed frames
condor_submit jobs/speed/cublas_ab.sub     # CLRerNet with the deterministic-cuBLAS setting off/on/off
```

**Expected full-prediction medians:** about 64 / 53 / 21 ms. Absolute values depend on the GPU and on node load.

## 4. Corruption robustness

```bash
condor_submit jobs/robust/robust_full.sub   # 3 GPU jobs; 2,895 test images x 16 conditions, identical inputs for all models
condor_submit jobs/robust/score_full.sub    # official evaluator per condition, plus the identical-input checksum check
```

**Expected:** clean / mean corrupted F1 of 69.77 / 56.09 (SCNN), 76.00 / 64.43 (UFLDv2) and 81.74 / 72.56 (CLRerNet). Results are written to `results/robustness/full_20261119T154132Z/`, a fixed name that the fine-tune scripts below also use.

## 5. Corruption-augmented fine-tuning (report §7.1–7.2)

```bash
condor_submit jobs/finetune/frame_diffs_chunks.sub      # train-set frame differences (8 CPU jobs), then:
condor_submit jobs/finetune/frame_diffs_merge.sub       # -> data_aux/train_diffs.npz (55,698 of 88,880 frames kept)
condor_submit jobs/finetune/ft_smoke.sub                # 10 iterations per arm
condor_submit jobs/finetune/ft_full.sub                 # seed 0: control + corruption, ~3.6 h each in parallel
bash scripts/make_ft2_dag.sh <ClusterId of ft_full>     # writes jobs/ft_dag2/ft2_eval.dag for those runs
condor_submit_dag jobs/ft_dag2/ft2_eval.dag             # waits for each epoch_3.pth, evaluates, scores in parallel
condor_submit jobs/finetune/ft_overnight.sub            # seeds 1-2 for both arms + the broad arm (5 GPUs)
bash scripts/make_ft2_dag.sh <ClusterId of ft_overnight> && condor_submit_dag jobs/ft_dag2/ft2_eval.dag
```

**Expected:**
- Mean ± standard deviation over 3 seeds: test F1 81.27 ± 0.10 (control) and 81.23 ± 0.05 (corruption); mean corrupted F1 70.17 ± 0.15 and 77.44 ± 0.11.
- Broad arm: 81.32 test F1 and 71.32 mean corrupted F1.
- Training is seeded and deterministic, but GPU kernels and data-loading order can still differ slightly between machines. Expect values within about the seed spread, not identical to the last digit.
- The tables: `reports/finetune_seeds.csv` and `reports/robustness.csv`.

## 6. Dashcam video, crop adapter and temporal tracker (report §8–9)

```bash
condor_submit jobs/b1_recal/valpred.sub                                    # val predictions for the geometry-filter calibration
VIDEO=$PWD/vendors/ufldv2/example.mp4 TAG=example condor_submit jobs/dashcam/new_video.sub
```

`new_video.sub` runs the whole video pipeline in one job: squash baseline → vanishing-point crop calibration → crop baseline → comparisons → B0–B3 ablation. Set `VIDEO` to any dashcam clip to test it independently.

**Expected:**
- Crop band rows 203–663 for `example.mp4`.
- Lane-count changes with the crop adapter: SCNN 213, UFLDv2 12 and CLRerNet 118.
- The ablation CSV appears under `results/ablation/new_example_*/`.
- The clip is unlabeled, so these are stability numbers, not accuracy.

## 7. Failure gallery and environment manifest

```bash
condor_submit jobs/extras/gallery.sub          # 4 worst test images per scenario and model (kept local: CULane images)
condor_submit jobs/extras/cpu_housekeeping.sub  # reports/environment_manifest.json (pip freeze per venv, checksums)
```

## 8. Demo app

See the README's "Demo app" section. It submits `jobs/app/infer.sub` for each Generate.

## Provenance files

- **`jobs/pretrained_*/`, `jobs/ft_dag/` and a few wrappers in `scripts/`** (`followup_ufld_fix.sh`, `redo_crop_ufld.sh`, `rerun_ufld_dashcam.sh`, `run_ablation_crop.sh`, `b1_recal_then_ablate.sh`, `run_verify_ufld.sh`, `run_estimate_vp.sh`): the exact files used in the original session. They contain the original run directory names, with the account name replaced by `<cluster-user>`. They are kept for provenance; the steps above are the generic route.
- **`patches/*.md`**: every environment fix and correction, with the job ids and evidence.
- **`reports/FINAL_REPORT_v1.md`–`v3.md`**: earlier report versions. v1 contains UFLDv2 numbers later found to be wrong (decoder bug, see §10 of the report).
