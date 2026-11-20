<!-- DRAFT: rewrite this README in your own words before publishing. Every number below comes from reports/. -->

# CULane Lane-Detection Benchmark & Robustness Study

A reproducible benchmark of three **pretrained** lane detectors on the full CULane test set, run on an HTCondor GPU cluster. It covers:
- reproducing the published accuracy;
- a single, shared evaluation protocol;
- same-hardware speed;
- a corruption-robustness study;
- dashcam video tests;
- a temporal lane tracker with a controlled ablation;
- corruption-augmented fine-tuning of CLRerNet, compared against a same-schedule control.

| Model | Code | Checkpoint |
|---|---|---|
| SCNN (VGG16) | [delldu/SCNN](https://github.com/delldu/SCNN) @ `a6fadfb`, a PyTorch port of the Torch7 original | converted official weights |
| Ultra-Fast-Lane-Detection-v2 (ResNet34) | [cfzd/Ultra-Fast-Lane-Detection-v2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2) @ `c903880` | `culane_res34.pth` |
| CLRerNet (DLA34, EMA) | [hirotomusiker/CLRerNet](https://github.com/hirotomusiker/CLRerNet) @ `dae038f` | `clrernet_culane_dla34_ema.pth` (v0.1.0) |

The pretrained weights are the authors'. This project's work is everything built around them: evaluation, benchmarking, diagnosis, robustness testing, video adaptation, post-processing and fine-tuning.

## Results

### Accuracy (CULane test, 34,680 images, official C++ evaluator, IoU 0.5)

| Model | Published F1 | Reproduced F1 |
|---|---|---|
| SCNN | 71.6 | 70.07 |
| UFLDv2, with TTA (author default) | 76.0 | 75.98 |
| UFLDv2, without TTA | 75.9 | 75.86 |
| CLRerNet | 81.55 | 81.55 |

Per-scenario results with the published values (checked against the papers' tables) are in `reports/culane_by_scenario.csv` and `reports/published_by_scenario.csv`. SCNN's 1.5-point gap is concentrated in the shadow and night scenarios.

### Speed (one Tesla P100, batch 1, FP32, 200 timed frames after 50 warmup frames)

| Model | Forward pass, median | Full prediction, median | FPS |
|---|---|---|---|
| SCNN | 51.0 ms | 63.9 ms | 15.7 |
| UFLDv2 | 11.7 ms | 53.3 ms | 18.8 |
| CLRerNet | 14.2 ms | 20.7 ms | 48.3 |

UFLDv2's network is fast, but preprocessing and the per-point Python decode dominate its end-to-end time. CLRerNet's deterministic-cuBLAS setting (`CUBLAS_WORKSPACE_CONFIG`) costs about 3.5 ms per frame, measured with an off/on/off A/B test.

### Robustness (2,895 test images × 5 corruptions × 3 severities; identical inputs for all models)

| | SCNN | UFLDv2 | CLRerNet |
|---|---|---|---|
| Clean F1 | 69.8 | 76.0 | 81.7 |
| Mean F1 over 15 corrupted conditions | 56.1 | 64.4 | 72.6 |
| Share of clean F1 kept | 80% | 85% | 89% |

Rain and heavy motion blur are the most damaging corruptions. Heavy JPEG compression (quality 5) is the only condition where CLRerNet does not lead.

### Corruption-augmented fine-tuning (CLRerNet, 3 epochs from the released checkpoint)

Runs with identical settings except for augmentation:
- the **control** uses the authors' training recipe;
- the **corruption** run adds one random corruption (rain, motion blur, JPEG or darkening) to half of the training images;
- the **broad** run instead adds one of 7 corruption types that never appear in the test set.

The shadow corruption was never trained on. The last-epoch checkpoint was used, with no selection, and the test set was scored once.

Control and corruption runs: mean ± standard deviation over 3 seeds.

| | Original | Fine-tuned control | Fine-tuned with corruptions | Broad augmentation (1 seed) |
|---|---|---|---|---|
| CULane test F1 (34,680 images) | 81.55 | 81.27 ± 0.10 | 81.23 ± 0.05 | 81.32 |
| Mean F1 over 15 corrupted conditions | 72.56 | 70.17 ± 0.15 | **77.44 ± 0.11** | 71.32 |
| Share of clean F1 kept | 88.8% | 86.4% | **95.5%** | 87.8% |
| Heavy rain (1,500 streaks) | 46.41 | 38.69 ± 0.33 | **77.38 ± 0.32** | 37.44 |
| JPEG quality 5 | 47.83 | 39.18 ± 0.78 | **64.15 ± 0.17** | 45.37 |
| Motion blur 25 px | 55.92 | 52.18 ± 0.08 | 60.55 ± 0.59 | **62.93** |
| Shadow, mean of 3 severities (never trained on) | 81.25 | 80.78 ± 0.06 | 80.73 ± 0.04 | 80.88 |

- **Corruption training gives a large, reproducible gain on the corruption families it trains on:** +4.9 mean F1 over the original, and heavy rain +31.
- **Clean accuracy:** it costs nothing measurable against the control (−0.04 test F1). The 0.3 drop from the original comes from fine-tuning itself.
- **Broad augmentation** with 7 different corruption types (fog, snow, glare, noise, low contrast, colour cast) transfers only partly to unseen ones: motion blur +4.3 and JPEG +2.3, but rain −1.2 and shadow unchanged.

Details: §7.1–7.2 of the report and `reports/finetune_seeds.csv`.

### Dashcam video and temporal tracking

- **Video adapter:** squashing 16:9 video into CULane's 2.78:1 shape distorts geometry. An aspect-preserving crop adapter, placed by matching vanishing points, reduces lane-count flicker for all three models.
- **Temporal tracker:** a causal tracker with matching, EMA smoothing and short dashed holds halves lane re-starts for SCNN and CLRerNet and reduces jitter for all three models, at under 1 ms per frame.
- **Geometry filter:** tested with three calibrations and not adopted, because it removed real lanes.

Full write-up: [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md).

## Demo app

`app/streamlit_app.py` is a small web app:
1. Pick one or more models: CLRerNet, the robust fine-tuned CLRerNet, UFLDv2 or SCNN.
2. Upload a dashcam image or video.
3. Click **Generate**.

The app shows the detected lanes drawn on the image or video, with download buttons and per-model statistics. Inference runs on a cluster GPU through HTCondor (`jobs/app/infer.sub` → `scripts/app_job.sh` → `scripts/app_infer.py`), using the same model code as the benchmark, the vanishing-point crop adapter and, for videos, the temporal tracker. Your machine only needs SSH key login to the cluster and, ideally, `ffmpeg`.

```bash
pip install -r app/requirements.txt
cp app/settings.example.json app/settings.json   # fill in your login and the remote benchmark_culane path
streamlit run app/streamlit_app.py
```

Uploaded media have no labels, so the app shows detections and stability statistics, not accuracy.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/` | evaluation, speed, robustness, dashcam, ablation, calibration and helper scripts |
| `config/`, `configs_ft/` | UFLDv2 test configs and CLRerNet fine-tuning configs |
| `jobs/` | HTCondor submit-file templates and the fine-tune evaluation DAG |
| `reports/` | result tables (CSV/JSON) and the full report |
| `patches/` | notes on every environment fix and correction, with evidence |
| `cluster_runs/` | one-off rerun wrappers kept for provenance |
| `app/` | Streamlit demo app and its cluster backend |

## Reproducing

1. Clone the three model repos at the commits above into `benchmark_culane/vendors/{scnn,ufldv2,clrernet}`.
2. Download the checkpoints into `benchmark_culane/checkpoints/`. Links are in `config/sources.json`; checksums are in `reports/environment_manifest.json`.
3. Obtain CULane from its official source; its licence is non-commercial. Place it at `datasets/CULane/`.
4. Set the paths: `./setup_paths.sh /absolute/path/to/Lane_Detection`.
5. Submit jobs from `jobs/`. For example, `condor_submit jobs/smoke5/smoke5.sub` runs a 5-image smoke test for all three models. The fine-tune runs through `jobs/finetune/` (frame differences, then `ft_full.sub`) and `jobs/ft_dag/ft_eval.dag` (evaluation and scoring). Each fine-tune takes about 3.6 h on a P100. The run directory names in `ft_eval.dag` are from the original run, so edit them to point at yours.

Environment: `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel`, one venv per model. CLRerNet uses MMDetection 3.3.0 and MMCV 2.1.0, built for sm_60.

## Not included

- **CULane data and images:** the licence forbids redistribution.
- **Model weights:** download them from the authors. The fine-tuned checkpoints are not included either; rerun the fine-tune to recreate them.
- **Vendor code:** it's pinned by commit and was never modified.

## Licence

The code and documentation in this repository are released under the [MIT License](LICENSE). The dataset, the model weights and the referenced model repositories keep their own licences.

## Acknowledgements

SCNN (Pan et al., AAAI 2018), Ultra-Fast-Lane-Detection-v2 (Qin et al., TPAMI 2022), CLRerNet (Honda & Uchida, WACV 2024), and the CULane dataset.
