#!/usr/bin/env bash
# One-time setup for the public demo, run from the repository root by demo_core.setup():
# the models' own Python 3.11 environment (model_env.txt, built with uv whatever Python the host runs), model repos at
# the pinned commits (unmodified) and checkpoints, each verified by sha256. A model whose checkpoint cannot be fetched
# (e.g. Google Drive rate limits) is skipped; the app then offers only the models that are ready.
# Env: LANE_ENV (environment dir, default .demo_env), UV_BIN (uv executable, default uv on PATH).
set -uo pipefail
B="$PWD"; S="$B/streamlit_cloud"
ENV_DIR="${LANE_ENV:-$B/.demo_env}"; PY="$ENV_DIR/bin/python"; UV="${UV_BIN:-uv}"
want="$(sha256sum "$S/model_env.txt" "$S/build_constraints.txt" | sha256sum | cut -c1-16)"
if [[ "$(cat "$ENV_DIR/.built" 2>/dev/null)" != "$want" ]]; then
  echo "building model environment in $ENV_DIR"; rm -rf "$ENV_DIR"
  "$UV" venv --quiet --python 3.11 "$ENV_DIR" || exit 3
  "$UV" pip install --quiet --python "$PY" --build-constraints "$S/build_constraints.txt" -r "$S/model_env.txt" || exit 3
  "$PY" -c 'import torch, torchvision, mmcv, mmdet, cv2, albumentations; print("torch", torch.__version__, "mmcv", mmcv.__version__)' || exit 3
  echo "$want" > "$ENV_DIR/.built"
fi
mkdir -p "$B/vendors" "$B/checkpoints/scnn" "$B/checkpoints/ufldv2" "$B/checkpoints/clrernet"
clone() {   # name url sha
  [[ "$(git -C "$B/vendors/$1" rev-parse HEAD 2>/dev/null)" == "$3" ]] && return 0
  rm -rf "$B/vendors/$1"; git clone --quiet "$2" "$B/vendors/$1" && git -C "$B/vendors/$1" checkout --quiet "$3" || return 1
  [[ "$(git -C "$B/vendors/$1" rev-parse HEAD)" == "$3" ]]
}
clone scnn     https://github.com/delldu/SCNN.git                       a6fadfbd9e593a28684dcaf29ca8ad425c1c700f || exit 4
clone ufldv2   https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2.git c903880678454dfd9b55a63022368db05c00bc6d || exit 4
clone clrernet https://github.com/hirotomusiker/CLRerNet.git            dae038f67da57e292e5293a68c9c1c2922de13c2 || exit 4
fetch() {   # file sha256 command...
  local f="$1" sha="$2"; shift 2
  if [[ -s "$f" ]] && echo "$sha  $f" | sha256sum -c --status; then return 0; fi
  for try in 1 2 3; do "$@" && echo "$sha  $f" | sha256sum -c --status && return 0; rm -f "$f"; sleep 10; done
  echo "SKIPPED $f (download or checksum failed)"; return 0
}
C=$B/checkpoints
fetch $C/clrernet/clrernet_culane_dla34_ema.pth cce8b552bc8301e31186da192fe4213043e9351b35651ef5d40014ce3040c787 \
  curl -sSfL -o $C/clrernet/clrernet_culane_dla34_ema.pth https://github.com/hirotomusiker/CLRerNet/releases/download/v0.1.0/clrernet_culane_dla34_ema.pth
# Our corruption-fine-tuned CLRerNet (report section 7.1), hosted in the project's Hugging Face Space repository.
fetch $C/clrernet/clrernet_culane_dla34_robust_ft.pth a90b78d306a510ee4cd775239c6a68e0eb01a0aa163a849daf03e480abcfa174 \
  curl -sSfL -o $C/clrernet/clrernet_culane_dla34_robust_ft.pth https://huggingface.co/spaces/roh24/culane_lane_detection/resolve/main/weights/clrernet_culane_dla34_robust_ft.pth
fetch $C/ufldv2/culane_res34.pth 1b4ca8385122f74c46a2e274950cdb5eb132ff97aa1ab1311550526d83caebf6 \
  "$PY" -m gdown --quiet 1AjnvAD3qmqt_dGPveZJsLZ1bOyWv62Yj -O $C/ufldv2/culane_res34.pth
fetch $C/scnn/vgg_SCNN_DULR_w9.pth 07c0c768dc435b8b162b58f42c7f686656133e0b13b8b4f110214245edb16671 \
  "$PY" -m gdown --quiet 1bBdN3yhoOQBC9pRtBUxzeRrKJdF7uVTJ -O $C/scnn/vgg_SCNN_DULR_w9.pth
ls -la $C/*/
echo "DEMO SETUP DONE"
