import gc

import torch

from transformers.utils import (
    is_torch_bf16_gpu_available,
    is_torch_cuda_available,
    is_torch_mps_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)

def is_accelerator_available() -> bool:
    r"""Check if the accelerator is available."""
    return (
        is_torch_xpu_available() or is_torch_npu_available() or is_torch_mps_available() or is_torch_cuda_available()
    )

def torch_gc() -> None:
    r"""Collect the device memory."""
    gc.collect()
    if is_torch_xpu_available():
        torch.xpu.empty_cache()
    elif is_torch_npu_available():
        torch.npu.empty_cache()
    elif is_torch_mps_available():
        torch.mps.empty_cache()
    elif is_torch_cuda_available():
        torch.cuda.empty_cache()