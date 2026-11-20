# Hugging Face Space (Docker SDK, free CPU hardware): lane-detection demo running the benchmark's verified model code
# on CPU. Built from the folder that hf_space/make_space.sh assembles.
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg curl ca-certificates libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 LANE_DEVICE=cpu OMP_NUM_THREADS=2
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
# imgaug pulls the GUI opencv-python (needs libGL); keep only the headless build, exactly as on the cluster.
RUN pip install --no-cache-dir --user -r requirements.txt \
    && pip uninstall -y opencv-python opencv-contrib-python \
    && pip install --no-cache-dir --user --no-deps --force-reinstall opencv-python-headless==4.10.0.84 \
    && python -c "import cv2, torch, mmcv, mmdet; print('env ok', cv2.__version__, torch.__version__, mmcv.__version__, mmdet.__version__)"

COPY --chown=user . .
# Model code at the pinned commits + the authors' checkpoints, verified by sha256 (fails the build on any mismatch).
RUN bash setup_space.sh

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false", "--server.maxUploadSize=100"]
