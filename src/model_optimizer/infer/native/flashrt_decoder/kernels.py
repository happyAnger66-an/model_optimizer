"""按 ``.so`` 路径装载 FlashRT 编译出的 kernel 扩展（``fvk``），**不 import flash_rt 包**。

FlashRT 的算子（``gmm_fp16`` / ``fp8_gemm_descale_fp16`` / ``fused_adarms_fp8_static_fp16``
/ ``attention_qkv_fp16`` …）由其 ``csrc`` 在 Thor(SM110) 上 cmake 构建为一个
``flash_rt_kernels*.so`` 扩展。该 CUDA 代码量约 27MB/700+ 文件（含 CUTLASS FMHA / FP8 GEMM），
不可能在仓内重写，故本仓库**复用其编译产物**，但只按文件路径动态加载，不依赖 ``flash_rt`` 的
任何 python 模块。

构建（在 Thor 上，FlashRT 源码树内）::

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j --target flash_rt_kernels
    # 产出 build/flash_rt_kernels*.so（及 libfmha_fp16_strided.so）

加载顺序：
  1) 环境变量 ``MO_FLASHRT_KERNELS_SO``（显式 .so 全路径）；
  2) 环境变量 ``MO_FLASHRT_BUILD_DIR`` 下的 ``flash_rt_kernels*.so``；
  3) 传入的 ``build_dir`` 参数。
"""

from __future__ import annotations

import glob
import importlib.util
import os
from pathlib import Path
from types import ModuleType

_KERNELS_CACHE: ModuleType | None = None


def _find_kernel_so(build_dir: str | None) -> str:
    candidates: list[str] = []
    env_so = os.environ.get("MO_FLASHRT_KERNELS_SO", "").strip()
    if env_so:
        candidates.append(env_so)
    search_dirs = [
        os.environ.get("MO_FLASHRT_BUILD_DIR", "").strip(),
        build_dir or "",
    ]
    for d in search_dirs:
        if not d:
            continue
        candidates.extend(sorted(glob.glob(str(Path(d).expanduser() / "flash_rt_kernels*.so"))))
        candidates.extend(
            sorted(glob.glob(str(Path(d).expanduser() / "**" / "flash_rt_kernels*.so"), recursive=True))
        )
    for c in candidates:
        if c and Path(c).is_file():
            return str(Path(c).resolve())
    raise FileNotFoundError(
        "未找到 flash_rt_kernels*.so。请在 Thor 上构建后通过 MO_FLASHRT_KERNELS_SO "
        "或 MO_FLASHRT_BUILD_DIR 指定，例如：\n"
        "  cmake --build build -j --target flash_rt_kernels\n"
        "  export MO_FLASHRT_BUILD_DIR=/path/to/FlashRT/build\n"
        f"  (已搜索: {[c for c in candidates if c]})"
    )


def load_kernels(build_dir: str | None = None, *, fmha_so: str | None = None) -> ModuleType:
    """加载 ``flash_rt_kernels`` 扩展模块（缓存单例）。

    Args:
        build_dir: 可选 FlashRT cmake build 目录。
        fmha_so: 可选 ``libfmha_fp16_strided.so`` 路径；提供时调用
            ``fvk.load_fmha_strided_library`` 注册 CUTLASS FMHA。

    Returns:
        kernel 扩展模块（即 FlashRT 中的 ``fvk``）。
    """
    global _KERNELS_CACHE
    if _KERNELS_CACHE is not None:
        return _KERNELS_CACHE

    so_path = _find_kernel_so(build_dir)
    # 用唯一模块名按路径加载扩展，避免与任何 flash_rt 包命名冲突。
    spec = importlib.util.spec_from_file_location("mo_flash_rt_kernels", so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从 {so_path} 创建扩展模块 spec")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    fmha = fmha_so or os.environ.get("MO_FLASHRT_FMHA_SO", "").strip()
    if fmha and Path(fmha).is_file() and hasattr(mod, "load_fmha_strided_library"):
        mod.load_fmha_strided_library(str(Path(fmha).resolve()))

    _KERNELS_CACHE = mod
    return mod
