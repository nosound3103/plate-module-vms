# syntax=docker/dockerfile:1
ARG CUDA_BASE_IMAGE=multi-camera-ai:cu128

FROM python:3.12-slim AS sync

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY requirements.lock ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --require-hashes -r requirements.lock \
    && useradd --uid 10001 --create-home plate

COPY --chown=plate:plate plate_ai ./plate_ai
USER plate
CMD ["python", "-m", "plate_ai"]

FROM python:3.12-slim AS cpu
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PLATE_DEVICE=cpu \
    PLATE_WEIGHTS_DIR=/models \
    OPENCV_LOG_LEVEL=SILENT \
    OPENCV_FFMPEG_LOGLEVEL=-8 \
    TORCH_HOME=/tmp/torch \
    XDG_CACHE_HOME=/tmp/cache
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.10.0+cpu torchvision==0.25.0+cpu --index-url https://download.pytorch.org/whl/cpu
COPY requirements-inference.lock ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps --require-hashes -r requirements-inference.lock \
    && pip check \
    && useradd --uid 10001 --create-home plate
COPY --chown=plate:plate plate_ai ./plate_ai
USER plate
CMD ["python", "-m", "plate_ai", "--mode", "detection"]

FROM ${CUDA_BASE_IMAGE} AS gpu
USER root
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/plate \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PLATE_DEVICE=cuda:0 \
    PLATE_WEIGHTS_DIR=/models \
    OPENCV_LOG_LEVEL=SILENT \
    OPENCV_FFMPEG_LOGLEVEL=-8 \
    TORCH_HOME=/tmp/torch \
    XDG_CACHE_HOME=/tmp/cache
WORKDIR /opt/plate
RUN /app/.venv/bin/python -c "import torch, torchvision; assert torch.__version__ == '2.10.0+cu128'; assert torchvision.__version__ == '0.25.0+cu128'" \
    && apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-inference.lock requirements-cuda-compat.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip uninstall --python /app/.venv/bin/python ultralytics ultralytics-opencv-headless opencv-python opencv-python-headless \
    && uv pip install --python /app/.venv/bin/python --no-deps --require-hashes -r requirements-inference.lock -r requirements-cuda-compat.lock \
    && uv pip check --python /app/.venv/bin/python \
    && useradd --uid 10001 --create-home plate \
    && rm -rf /opt/plate/plate_ai
COPY --chown=plate:plate plate_ai ./plate_ai
USER plate
HEALTHCHECK NONE
ENTRYPOINT []
CMD ["/app/.venv/bin/python", "-m", "plate_ai", "--mode", "detection"]
