#!/usr/bin/env bash
# Runs on assigned HTCondor GPU in the specified Docker container.
set -euo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:?scnn|ufldv2|clrernet}"; PHASE="${2:?smoke|full}"
if [[ "$MODEL" != scnn && "$MODEL" != ufldv2 && "$MODEL" != clrernet ]]; then exit 5; fi
if [[ "$PHASE" != smoke && "$PHASE" != smoke5 && "$PHASE" != speed && "$PHASE" != dashcam && "$PHASE" != valpred && "$PHASE" != full_notta && "$PHASE" != robust && "$PHASE" != finetune && "$PHASE" != full ]]; then exit 6; fi
cd "$B"
mkdir -p "$B/results/$MODEL" "$B/logs"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PIP_DISABLE_PIP_VERSION_CHECK=1
export TORCH_HOME="$B/.cache/torch";mkdir -p "$TORCH_HOME"
printf 'MODEL=%s PHASE=%s HOST=%s DATE=%s\n' "$MODEL" "$PHASE" "$(hostname)" "$(date -Is)"
nvidia-smi || { echo 'No allocated GPU visible'; exit 7; }
ENV="$B/.venvs/$MODEL"
if [[ ! -e "$ENV/bin/python" ]];then
  python -m venv --system-site-packages "$ENV"
fi
source "$ENV/bin/activate"
if [[ ! -e "$ENV/.deps_ok" ]];then
  # NEVER overwrite the container's torch/torchvision; use --system-site-packages.
  pip install 'numpy<2' 'opencv-python-headless==4.10.0.84' 'scipy<1.14' \
    'Pillow>=9,<11' 'tqdm<5'
  if [[ "$MODEL" == ufldv2 ]]; then
    pip install 'addict' 'pathspec' 'imagesize' 'ujson' 'tensorboard' 'scikit-learn'
  fi
  if [[ "$MODEL" == clrernet ]]; then
    pip install 'albumentations<2' 'kornia' 'prettytable'
    echo 'Installing the upstream MMDetection stack for CLRerNet. A compatible prebuilt MMCV wheel is required.'
    pip install 'openmim==0.3.9'
    mim install 'mmcv==2.1.0' 'mmengine==0.10.4' 'mmdet==3.3.0'
    pip install -r "$B/vendors/clrernet/requirements.txt"
    # Compile only within the allocated GPU container; no installation on login node.
    (cd "$B/vendors/clrernet/libs/models/layers/nms" && python setup.py build_ext --inplace)
  fi
  touch "$ENV/.deps_ok"
fi
# CLRerNet depends on GUI OpenCV via an upstream requirement, but the GPU Docker image
# lacks libxcb.so.1. Install the headless wheel into the dedicated model venv.
if [[ "$MODEL" == clrernet && ! -f "$ENV/.headless_ok" ]]; then
  pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless || true
  pip install --no-deps --force-reinstall 'opencv-python-headless==4.10.0.84'
  python -c 'import cv2; print("CLRerNet headless OpenCV ready:", cv2.__version__)'
  touch "$ENV/.headless_ok"
fi
# PATCH 2026-11-19 (see patches/clrernet_numpy_pin.md): vendor requirements pulled numpy 2.2.6,
# which breaks imgaug 0.4.0 (np.sctypes removed in NumPy 2). Pin to the container's numpy 1.26.4.
if [[ "$MODEL" == clrernet && ! -f "$ENV/.numpy1_ok" ]]; then
  pip install --no-deps --force-reinstall 'numpy==1.26.4'
  python -c 'import numpy, imgaug, mmdet.apis; print("CLRerNet numpy pin ok:", numpy.__version__)'
  pip freeze > "$B/results/clrernet/pip_freeze_after_numpy_pin.txt"
  touch "$ENV/.numpy1_ok"
fi
python - <<'PY'
import torch,cv2,numpy
print('torch=',torch.__version__,'cuda=',torch.version.cuda,'cuda_ready=',torch.cuda.is_available(),
'opencv=',cv2.__version__,'numpy=',numpy.__version__,flush=True)
assert torch.cuda.is_available(), 'GPU unavailable'
PY
# Strict checkpoint gate: never evaluate partially loaded or random models.
python "$B/scripts/validate_checkpoint.py" "$MODEL"
# PATCH 2026-11-19 (patches/clrernet_pythonpath.md): mirror upstream docker/Dockerfile
# (ENV PYTHONPATH=/work; nms installed as package) without modifying vendor code.
if [[ "$MODEL" == clrernet ]]; then
  # PATCH 2026-11-19 (patches/clrernet_pythonpath.md): mirror upstream docker/Dockerfile:56; config sets
  # randomness.deterministic=True, which requires this for cuBLAS matmul (job 73863 RuntimeError).
  # CLR_NO_CUBLAS_CFG=1 leaves it unset (diagnostic: speed of plain inference, which needs no deterministic cuBLAS)
  [[ "${CLR_NO_CUBLAS_CFG:-0}" == 1 ]] || export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export PYTHONPATH="$B/vendors/clrernet:$B/vendors/clrernet/libs/models/layers/nms/src:${PYTHONPATH:-}"
  python -c 'import torch, libs; from nms import nms as _n; print("CLRerNet libs+nms import ok")'
fi
case "$MODEL/$PHASE" in
  clrernet/finetune)
    # PATCH 2026-11-19 (patches/clrernet_finetune.md): fine-tuning with configs_ft/*.py; vendor code unchanged.
    : "${FT_CFG:?}" "${FT_NAME:?}"
    RUN="$B/results/finetune/${FT_NAME}_$(date -u +%Y%m%dT%H%M%SZ)_${HOSTNAME%%.*}"; mkdir -p "$RUN"
    SHORT_TMP="$(mktemp -d /tmp/clrft.XXXX)"; export TMPDIR="$SHORT_TMP"
    echo "FT_RUN_DIR=$RUN"; cp "$FT_CFG" "$RUN/"
    cd "$B/vendors/clrernet"
    python tools/train.py "$FT_CFG" --work-dir "$RUN" ${FT_OPTS:+--cfg-options $FT_OPTS};;
  */robust)
    # PATCH 2026-11-19 (patches/robustness.md): corruption robustness on fixed CULane test subset.
    RUN="${ROBUST_ROOT:?}/$MODEL"; mkdir -p "$RUN"; echo "ROBUST_RUN_DIR=$RUN"
    python "$B/scripts/robustness.py" "$MODEL" "$RUN" ${ROBUST_LIMIT:+--limit $ROBUST_LIMIT} ${ROBUST_CONDS:+--conds $ROBUST_CONDS --tag ${ROBUST_TAG:-chunk}} > >(tee "$RUN/stdout.log") 2> >(tee "$RUN/stderr.log" >&2);;
  ufldv2/full_notta)
    # PATCH 2026-11-19 (patches/ufldv2_notta.md): author run_test path, tta=False, separate work dir.
    python "$B/scripts/run_ufld_full_notta.py"
    n=$(find "$B/results/ufldv2/full_notta/culane_eval_tmp" -name '*.lines.txt' | wc -l); echo "UFLD_NOTTA_PRED_FILES=$n"
    [[ "$n" -eq 34680 ]] || { echo "FAIL: expected 34680, got $n" >&2; exit 13; };;
  */valpred)
    # PATCH 2026-11-19 (patches/b1_recalibration.md): CULane VAL predictions for B1 filter calibration.
    RUN="${VALPRED_ROOT:?}/$MODEL"; mkdir -p "$RUN"; echo "VALPRED_RUN_DIR=$RUN"
    python "$B/scripts/val_predict.py" "$MODEL" "$RUN" > >(tee "$RUN/stdout.log") 2> >(tee "$RUN/stderr.log" >&2);;
  */dashcam)
    # PATCH 2026-11-19 (patches/dashcam.md): baseline dashcam overlay, handbook Phase E.
    RUN="$B/results/dashcam/$MODEL/${DASH_BATCH:-run_$(date -u +%Y%m%dT%H%M%SZ)_${HOSTNAME%%.*}}"; mkdir -p "$RUN"
    echo "DASHCAM_RUN_DIR=$RUN"
    python "$B/scripts/run_dashcam_video.py" "$MODEL" "${DASH_VIDEO:?}" "$RUN" ${DASH_CROP:+"$DASH_CROP"} > >(tee "$RUN/stdout.log") 2> >(tee "$RUN/stderr.log" >&2);;
  */speed)
    # PATCH 2026-11-19 (patches/speed.md): same-hardware latency benchmark, handbook Phase D.
    RUN="$B/results/speed/${SPEED_BATCH:-run_$(date -u +%Y%m%dT%H%M%SZ)_${HOSTNAME%%.*}}/$MODEL"; mkdir -p "$RUN"
    echo "SPEED_RUN_DIR=$RUN"
    python "$B/scripts/benchmark_speed.py" "$MODEL" "$RUN" > >(tee "$RUN/speed_stdout.log") 2> >(tee "$RUN/speed_stderr.log" >&2);;
  */smoke5)
    # PATCH 2026-11-19 (patches/smoke5.md): uniform handbook Phase B 5-image smoke, new run dir per job.
    RUN="$B/results/$MODEL/smoke5/run_$(date -u +%Y%m%dT%H%M%SZ)_${HOSTNAME%%.*}"; mkdir -p "$RUN"
    echo "SMOKE5_RUN_DIR=$RUN"
    python "$B/scripts/smoke5.py" "$MODEL" "$RUN" > >(tee "$RUN/smoke_stdout.log") 2> >(tee "$RUN/smoke_stderr.log" >&2);;
  scnn/smoke) python "$B/scripts/scnn_infer.py" --max-frames 4;;
  scnn/full)
    python "$B/scripts/scnn_infer.py" --max-frames 0
    if [[ -x "$B/vendors/scnn/utils/lane_evaluation/CULane/evaluate" ]]; then
      bash "$B/scripts/eval_official_scnn.sh"
    else
      echo 'SCNN FULL INFERENCE COMPLETE; OFFICIAL CULANE EVALUATION NOT RUN: C++ evaluator not yet compiled.'
    fi;;
  ufldv2/smoke) python "$B/scripts/ufld_smoke.py";;
  ufldv2/full)
    if [[ ! -x "$B/vendors/ufldv2/evaluation/culane/evaluate" ]]; then
      echo 'UFLDv2 official CULane evaluator executable missing. Refusing to repeat 34,680-frame inference. Use culane_repair/ufld_eval_only.py after compiling evaluator.' >&2
      exit 12
    fi
    python "$B/scripts/run_ufld_full.py";;
  clrernet/smoke)
    cd "$B/vendors/clrernet"
    IMAGE="$($ENV/bin/python "$B/scripts/first_image.py")"
    python demo/image_demo.py "$IMAGE" configs/clrernet/culane/clrernet_culane_dla34_ema.py \
      "$B/checkpoints/clrernet/clrernet_culane_dla34_ema.pth" \
      --out-file="$B/results/clrernet/smoke.jpg";;
  clrernet/full)
    # PATCH 2026-11-19 (patches/clrernet_full_tmpdir.md): CULaneMetric writes predictions under
    # tempfile.gettempdir(); point TMPDIR at a persistent per-run dir so .lines.txt survive the job.
    RUN="$B/results/clrernet/full/run_$(date -u +%Y%m%dT%H%M%SZ)_${HOSTNAME%%.*}"
    mkdir -p "$RUN"
    # PATCH 2 (job 73848 hung): a long TMPDIR broke DataLoader worker sockets (AF_UNIX path too long).
    # Use a short container-local TMPDIR, then copy predictions into $RUN/pred after the run.
    SHORT_TMP="$(mktemp -d /tmp/clr.XXXX)"; export TMPDIR="$SHORT_TMP"
    echo "CLRERNET_RUN_DIR=$RUN"
    cd "$B/vendors/clrernet"
    python tools/test.py configs/clrernet/culane/clrernet_culane_dla34_ema.py        "$B/checkpoints/clrernet/clrernet_culane_dla34_ema.pth"        --work-dir "$RUN/work_dir"
    PRED_SRC=""; for d in "$SHORT_TMP"/tmp*; do [[ -d "$d/driver_100_30frame" ]] && PRED_SRC="$d"; done
    [[ -n "$PRED_SRC" ]] || { echo "FAIL: no CULaneMetric result dir under $SHORT_TMP" >&2; ls -la "$SHORT_TMP" >&2; exit 14; }
    echo "CLRERNET_METRIC_RESULT_DIR=$PRED_SRC"; cp -r "$PRED_SRC" "$RUN/pred"
    n=$(find "$RUN/pred" -name '*.lines.txt' | wc -l); echo "CLRERNET_PRED_FILES=$n PRED_DIR=$RUN/pred"
    [[ "$n" -eq 34680 ]] || { echo "FAIL: expected 34680 prediction files, got $n" >&2; exit 13; };;
esac
