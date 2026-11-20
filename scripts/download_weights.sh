#!/usr/bin/env bash
# Run on an authorized data-transfer node or login node if site policy permits.
# Weights are not bundled because they are large and third-party owned.
set -uo pipefail
B="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$(dirname "$B")"
mkdir -p "$B/checkpoints/"{scnn,ufldv2,clrernet}
GDOWN="$R/.venv_download/bin/gdown"
if [[ ! -x "$GDOWN" ]]; then GDOWN="$(command -v gdown || true)"; fi
if [[ -z "$GDOWN" ]]; then echo 'Missing gdown. Install inside .venv_download first.' >&2; exit 2; fi
maybe_gdown(){
  local model="$1" id="$2" out="$3"
  if [[ -s "$out" && $(stat -c%s "$out") -gt 1048576 ]]; then echo "KEEP $out"; return 0; fi
  echo "Downloading $model checkpoint (Google Drive may throttle) ..."
  "$GDOWN" --continue --retries 2 "$id" -O "$out" || { echo "BLOCKED $model: download manually from author link" >&2; rm -f "$out"; return 1; }
  [[ $(stat -c%s "$out") -gt 1048576 ]] || { echo "Invalid tiny checkpoint $out" >&2; return 1; }
}
maybe_gdown scnn 1bBdN3yhoOQBC9pRtBUxzeRrKJdF7uVTJ "$B/checkpoints/scnn/vgg_SCNN_DULR_w9.pth" || true
maybe_gdown ufldv2 1AjnvAD3qmqt_dGPveZJsLZ1bOyWv62Yj "$B/checkpoints/ufldv2/culane_res34.pth" || true
CLR="$B/checkpoints/clrernet/clrernet_culane_dla34_ema.pth"
if [[ -s "$CLR" && $(stat -c%s "$CLR") -gt 1048576 ]]; then echo "KEEP $CLR";
else
 curl --fail --location --retry 3 --retry-delay 3 \
  'https://github.com/hirotomusiker/CLRerNet/releases/download/v0.1.0/clrernet_culane_dla34_ema.pth' \
  --output "$CLR.part" && mv "$CLR.part" "$CLR" || { echo 'BLOCKED clrernet GitHub release' >&2; rm -f "$CLR.part"; }
fi
for f in "$B"/checkpoints/*/*; do [[ -f "$f" ]] && sha256sum "$f"; done | tee "$B/results/checkpoint_sha256.txt"
echo 'If Google Drive says quota exceeded, do NOT train from random weights. Put author checkpoints at exact paths above.'
