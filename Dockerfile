FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Use a venv to avoid PEP 668 issues on Ubuntu 24.04
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# PyTorch 2.7.0 with CUDA 12.8 (Blackwell sm_120 support)
RUN pip install --no-cache-dir \
    torch==2.7.0 \
    torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ONNX Runtime with CUDA EP
RUN pip install --no-cache-dir onnxruntime-gpu==1.21.1

# Project dependencies
RUN pip install --no-cache-dir \
    fastapi==0.115.12 \
    uvicorn[standard]==0.34.2 \
    python-multipart==0.0.20 \
    Pillow==11.2.1 \
    numpy==2.2.4 \
    scikit-learn==1.6.1 \
    tensorboard==2.19.0 \
    tqdm==4.67.1 \
    requests==2.32.3 \
    imagehash==4.3.2 \
    onnx==1.17.0 \
    opencv-python-headless==4.10.0.84

# EasyOCR for hero name identification
RUN pip install --no-cache-dir easyocr==1.7.2

# Pre-bake EasyOCR model weights so first container start is fast
RUN python3 -c "import easyocr; easyocr.Reader(['en', 'ch_sim'], gpu=False)"

WORKDIR /workspace
