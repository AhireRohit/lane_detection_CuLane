---
title: Lane Detection Demo
emoji: 🛣️
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 4.44.1
python_version: "3.10"
app_file: app.py
pinned: false
license: other
short_description: Lane detection: SCNN, UFLDv2, CLRerNet + robust fine-tune
---

# Lane detection demo

Upload a dashcam image or short video and compare four CULane lane detectors:
- **SCNN** (VGG16);
- **Ultra-Fast-Lane-Detection-v2** (ResNet34);
- **CLRerNet** (DLA34);
- **CLRerNet with our corruption-augmented fine-tune**, which is more robust to rain, blur, JPEG compression and darkness: mean corrupted-image F1 77.4 ± 0.1 against 72.6, over 3 seeds.

**How it works**
- Inference runs on CPU with the same code as the GPU benchmark. On first start the Space downloads the model code (pinned commits) and the authors' checkpoints and checks their sha256.
- [VERIFICATION.md](VERIFICATION.md) shows that the CPU path reproduces the GPU predictions behind the reported numbers.
- The full benchmark, report and code are in the GitHub repository.

**Licences**
- The app code is MIT.
- The pretrained weights belong to their authors and keep their licences: SCNN port by delldu, UFLDv2 by cfzd, CLRerNet by hirotomusiker.
- All models, including our fine-tune, were trained on CULane, whose licence is non-commercial. This demo is for research and education only.

**Privacy:** uploads are processed only inside this Space and deleted after an hour.
