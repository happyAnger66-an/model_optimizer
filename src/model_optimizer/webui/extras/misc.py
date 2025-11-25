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

def get_current_memory() -> tuple[int, int]:
    r"""Get the available and total memory for the current device (in Bytes)."""
    if is_torch_xpu_available():
        return torch.xpu.mem_get_info()
    elif is_torch_npu_available():
        return torch.npu.mem_get_info()
    elif is_torch_mps_available():
        return torch.mps.current_allocated_memory(), torch.mps.recommended_max_memory()
    elif is_torch_cuda_available():
        return torch.cuda.mem_get_info()
    else:
        return 0, -1