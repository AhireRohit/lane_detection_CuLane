# CPU demo vs GPU benchmark: verification

The public demo runs on CPU (PyTorch 2.1.2+cpu, MMCV 2.1.0 CPU build) instead of the cluster's GPUs (PyTorch 2.3.1, CUDA 12.1). These checks show that it computes the same lanes as the GPU code behind every number in the report. Every check ran on the Saarland HPC cluster on 2026-11-20; the scripts are in the GitHub repository.

## 1. Same code, same images: GPU vs CPU (`scripts/verify_cpu.py dump/cmp`)

**Setup**
- 200 CULane test images: every 173rd entry of `list/test.txt`.
- The same model builders as the demo (`benchmark_speed.py` → `app_infer.build`) and the same OpenCV image loading.
- GPU side: the cluster venvs (jobs 74334). CPU side: the Space's own `requirements.txt` in a `python:3.10-slim` container with no GPU (job 74335).

**Metric:** lanes are matched one-to-one, and the worst distance from any point to the matched lane's curve is reported, in pixels at 1640×590.

| Model | Same number of lanes | Median worst distance | Largest over 200 images |
|---|---|---|---|
| SCNN (VGG16) | 200 / 200 | 0.0 px | 0.0 px |
| UFLDv2 (ResNet34, no TTA) | 200 / 200 | 0.0002 px | 0.0004 px |
| CLRerNet (DLA34, released EMA weights) | 200 / 200 | 0.0002 px | 0.0007 px |
| CLRerNet, robust fine-tune (seed 0) | 200 / 200 | 0.0001 px | 0.0015 px |

The differences are floating-point rounding, about a thousandth of a pixel.

## 2. CLRerNet's lane NMS: CPU port vs the CUDA op (`scripts/nms_cpu.py`)

- CLRerNet's non-maximum suppression is a CUDA-only compiled extension. `nms_cpu.py` translates its kernel line by line, including float32 accumulation and the C integer rounding.
- In a GPU job (74328), both ran on the identical proposals for 2,000 test images: **1,751 NMS calls, 0 differences in the kept lanes**.

## 3. Robust fine-tune checkpoint shipped in the demo

- `weights/clrernet_culane_dla34_robust_ft.pth` is the seed-0 corruption fine-tune (report §7.1), reduced to the `state_dict` that `mmdet` loads plus `dataset_meta`. The training config is dropped because it contains cluster paths.
- All 305 tensors are bit-identical to the training checkpoint (jobs 74331 and 74341). sha256 `a90b78d306a510ee4cd775239c6a68e0eb01a0aa163a849daf03e480abcfa174`.

## 4. Demo code vs the cluster app

`app_infer.py` was split into `build()` / `process()` so the demo can reuse it. The cluster app (GPU) gives identical lanes and crop band for all four models before and after the change (jobs 74171 vs 74330).

## Context: CPU demo vs the scored prediction files

Comparing the CPU demo with the prediction files that the authors' test scripts wrote (job 74332) shows larger gaps for CLRerNet and UFLDv2:
- **SCNN:** 0.0 px, because the same writer was used.
- **UFLDv2:** median 1.5 px. The scored run decoded JPEGs with PIL, while the demo uses OpenCV; this occasionally moves an anchor.
- **CLRerNet:** median 3.8 px. `tools/test.py` resamples lanes on a different set of rows than the inference API.

Check 1 shows these come from the output writers and image loaders, not from running on CPU. The lane counts still agree on 99–100% of images.

## 5. The assembled Space app, end to end (`scripts/test_space_app.sh`, job 74339)

**Setup**
- The exact folder that is pushed to Hugging Face (`hf_space/make_space.sh`), installed the way the Gradio SDK installs it.
- Run in a CPU-only container, with the app's own `generate()` called on:
  - a dashcam frame from the UFLDv2 demo clip, all four models;
  - an 8-second clip at 10 fps, CLRerNet and the robust fine-tune, tracker on.

**Results:** both runs completed, in 38 s and 60 s, with correct overlays, videos and statistics.

**One small difference.** On this non-CULane frame, the crop band that SCNN's lanes place came out 2 rows lower than in the GPU cluster app (rows 209–669 vs 211–671). The other three models placed the same band as on GPU. On CULane test images, check 1 found SCNN's CPU and GPU lanes identical.
