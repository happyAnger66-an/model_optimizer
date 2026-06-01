"""GPU 指针小工具（移植自 FlashRT ``hardware/thor/shared_primitives.py``）。

纯 ctypes + libcudart，无 flash_rt 依赖。供 decoder 编排与校准使用。
"""

from __future__ import annotations

import ctypes

_crt = ctypes.CDLL("libcudart.so")


def gpu_alloc(nbytes: int) -> int:
    """cudaMalloc → 返回 int 指针。"""
    ptr = ctypes.c_void_p()
    _crt.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
    return ptr.value


def gpu_free(ptr: int) -> None:
    _crt.cudaFree(ctypes.c_void_p(ptr))


def gpu_zero(ptr: int, nbytes: int, stream: int = 0) -> None:
    _crt.cudaMemsetAsync(
        ctypes.c_void_p(ptr), 0, ctypes.c_size_t(nbytes), ctypes.c_void_p(stream)
    )


def gpu_copy(dst: int, src: int, nbytes: int, stream: int = 0) -> None:
    """Device-to-device 拷贝（cudaMemcpyKind=3）。"""
    _crt.cudaMemcpyAsync(
        ctypes.c_void_p(dst),
        ctypes.c_void_p(src),
        ctypes.c_size_t(nbytes),
        3,
        ctypes.c_void_p(stream),
    )


def gpu_sync(stream: int = 0) -> None:
    if stream:
        _crt.cudaStreamSynchronize(ctypes.c_void_p(stream))
    else:
        _crt.cudaDeviceSynchronize()


def d2h_float(device_ptr: int) -> float:
    """读单个 float32（D2H）。"""
    val = ctypes.c_float()
    _crt.cudaMemcpy(ctypes.byref(val), ctypes.c_void_p(device_ptr), ctypes.c_size_t(4), 2)
    return val.value


def d2h_floats(device_ptr: int, count: int) -> list[float]:
    """读 float32 数组（D2H）→ host list。"""
    arr = (ctypes.c_float * count)()
    _crt.cudaMemcpy(arr, ctypes.c_void_p(device_ptr), ctypes.c_size_t(count * 4), 2)
    return [float(arr[i]) for i in range(count)]


def measure_scale_gpu(
    fvk_mod, fp16_ptr: int, n_elements: int, d_scale_ptr: int, d_fp8_scratch: int, stream: int = 0
) -> None:
    """GPU 端 amax 测量：absmax → compute_scale → d_scale。

    复用 ``quantize_fp8_device_fp16``（absmax + scale + quantize 三合一），
    量化输出落到 ``d_fp8_scratch``（丢弃），scale 写入 ``d_scale_ptr``。
    """
    fvk_mod.quantize_fp8_device_fp16(fp16_ptr, d_fp8_scratch, d_scale_ptr, n_elements, stream)
