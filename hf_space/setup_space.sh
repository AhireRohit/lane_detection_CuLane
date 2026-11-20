#!/usr/bin/env bash
# Runs during the Space's Docker build: model repos at the pinned commits (unmodified) and the authors' checkpoints,
# each verified by sha256. The robust fine-tuned CLRerNet is shipped in weights/ (verified too).
set -euo pipefail
B="$PWD/benchmark_culane"
mkdir -p "$B/vendors" "$B/checkpoints/scnn" "$B/checkpoints/ufldv2" "$B/checkpoints/clrernet"
clone() {   # name url sha
  git clone --quiet "$2" "$B/vendors/$1"
  git -C "$B/vendors/$1" checkout --quiet "$3"
  [[ "$(git -C "$B/vendors/$1" rev-parse HEAD)" == "$3" ]] || { echo "pin mismatch $1"; exit 4; }
  rm -rf "$B/vendors/$1/.git"
}
clone scnn     https://github.com/delldu/SCNN.git                       a6fadfbd9e593a28684dcaf29ca8ad425c1c700f
clone ufldv2   https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2.git c903880678454dfd9b55a63022368db05c00bc6d
clone clrernet https://github.com/hirotomusiker/CLRerNet.git            dae038f67da57e292e5293a68c9c1c2922de13c2
# Authors' checkpoints (links as in config/sources.json). Google Drive can rate-limit; retry a few times.
for try in 1 2 3; do
  [[ -s "$B/checkpoints/scnn/vgg_SCNN_DULR_w9.pth" ]] || gdown --quiet 1bBdN3yhoOQBC9pRtBUxzeRrKJdF7uVTJ -O "$B/checkpoints/scnn/vgg_SCNN_DULR_w9.pth" || true
  [[ -s "$B/checkpoints/ufldv2/culane_res34.pth" ]] || gdown --quiet 1AjnvAD3qmqt_dGPveZJsLZ1bOyWv62Yj -O "$B/checkpoints/ufldv2/culane_res34.pth" || true
  [[ -s "$B/checkpoints/scnn/vgg_SCNN_DULR_w9.pth" && -s "$B/checkpoints/ufldv2/culane_res34.pth" ]] && break
  sleep 20
done
curl -sSfL -o "$B/checkpoints/clrernet/clrernet_culane_dla34_ema.pth" \
  https://github.com/hirotomusiker/CLRerNet/releases/download/v0.1.0/clrernet_culane_dla34_ema.pth
(cd "$B" && sha256sum -c config/checkpoints.sha256)
(sha256sum -c weights/SHA256SUMS)
echo "SPACE SETUP OK"
