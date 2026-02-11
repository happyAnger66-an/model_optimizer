#! /bin/bash
set -e

export PATH=$HOME/.local/bin:$PATH

source /opt/openpi/bin/activate

uv pip uninstall torch torchvision torchcodec \
    nvidia-cublas-cu12 \
    nvidia-cuda-cupti-cu12 \
    nvidia-cuda-nvrtc-cu12 \
    nvidia-cuda-runtime-cu12 \
    nvidia-cudnn-cu12 \
    nvidia-cufft-cu12 \
    nvidia-cufile-cu12 \
    nvidia-curand-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-cusparselt-cu12 \
    nvidia-nccl-cu12 \
    nvidia-nvjitlink-cu12 \
    nvidia-nvtx-cu12